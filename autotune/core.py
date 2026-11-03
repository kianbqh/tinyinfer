"""Pure control logic. Fake-proposer unit checks do not establish LLM quality."""
from __future__ import annotations
import json
import math
import random
import statistics
from bench.v2.kernels.candidate_build import validate_proposal
from llm.native_ops import GEMM_CATALOG


def proposal(identifier):
    c=GEMM_CATALOG[identifier]
    return dict(config_id=identifier,tile=list(c["tile"]),stages=c["stages"],epilogue=c["epilogue"])


def parse_json(raw):
    if not isinstance(raw,str) or len(raw.encode("utf-8"))>16384:raise ValueError("proposal output limit")
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError("duplicate JSON key")
            out[k]=v
        return out
    def invalid(value):raise ValueError("non-finite JSON constant: "+value)
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)
    validate_proposal(value)
    return value


def median(samples):
    if not isinstance(samples,list) or len(samples)<7 or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in samples):
        raise ValueError("need >=7 positive finite per-call samples")
    return statistics.median(samples)


def gate(record,incumbent):
    if record.get("status")!="passed" or record.get("correctness_pass") is not True:return False
    candidate=median(record["samples_ms"]);baseline=median(incumbent["samples_ms"])
    if incumbent.get("correctness_pass") is not True or incumbent.get("status")!="passed":raise ValueError("invalid incumbent")
    return candidate<=0.97*baseline


class RandomProposer:
    """Draw the same exact catalog with replacement; duplicates stay observable."""
    def __init__(self,seed):self.rng=random.Random(seed)
    def __call__(self,history,attempt):return json.dumps(proposal(self.rng.randrange(12)))


def search(proposer,evaluate,baseline,*,proposal_budget,evaluation_budget,save=lambda value:None):
    if type(proposal_budget) is not int or type(evaluation_budget) is not int or not 1<=evaluation_budget<=11 or proposal_budget<evaluation_budget:
        raise ValueError("search budgets")
    if baseline.get("status")!="passed" or baseline.get("correctness_pass") is not True:raise ValueError("baseline correctness failed")
    median(baseline["samples_ms"])
    base_id=validate_proposal(baseline["proposal"]);seen={base_id};incumbent=baseline
    result=dict(status="started",proposal_budget=proposal_budget,evaluation_budget=evaluation_budget,baseline=baseline,
                attempts=[],evaluations=0,accepted=0,invalid=0,duplicates=0,proposal_errors=0,compile_failures=0,test_failures=0,
                valid_compiles=0,test_passes=0,evaluation_errors=0,best=baseline)
    save(result)
    for attempt in range(proposal_budget):
        if result["evaluations"]>=evaluation_budget:break
        row={"attempt":attempt,"status":"proposing"};result["attempts"].append(row);save(result)
        try:
            raw=proposer(result,attempt);row["raw_proposal"]=raw
        except Exception as e:
            row.update(status="proposal_error",error_type=type(e).__name__,error=str(e));result["proposal_errors"]+=1;save(result);continue
        try:
            value=parse_json(raw);identifier=validate_proposal(value);row["proposal"]=value
        except Exception as e:
            row.update(status="invalid",error_type=type(e).__name__,error=str(e));result["invalid"]+=1;save(result);continue
        if identifier in seen:
            row["status"]="duplicate";result["duplicates"]+=1;save(result);continue
        seen.add(identifier);result["evaluations"]+=1;row["status"]="evaluating";save(result)
        try:
            record=evaluate(value,attempt);row["evaluation"]=record
            if record.get("proposal")!=value:raise ValueError("evaluation candidate mismatch")
            status=record.get("status")
            if status=="compile_failed":result["compile_failures"]+=1
            elif status in ("test_failed","passed"):
                result["valid_compiles"]+=1
                if status=="test_failed":result["test_failures"]+=1
                else:
                    if record.get("correctness_pass") is not True:raise ValueError("passing evaluation lacks correctness")
                    if baseline.get("fixture") is not None and record.get("fixture")!=baseline["fixture"]:raise ValueError("candidate fixture differs from initial baseline")
                    median(record["samples_ms"]);result["test_passes"]+=1
                    if gate(record,incumbent):
                        incumbent=record;result["best"]=record;result["accepted"]+=1;row["accepted"]=True
            else:raise ValueError("unknown evaluator status")
            row["status"]=status
        except Exception as e:
            row.update(status="evaluation_error",error_type=type(e).__name__,error=str(e));result["evaluation_errors"]+=1
        save(result)
    result["proposal_attempts"]=len(result["attempts"])
    result["budget_exhausted"]=result["evaluations"]==evaluation_budget
    result["status"]="completed" if result["budget_exhausted"] else "incomplete_proposal_cap"
    result["selection_speedup_vs_initial_baseline"]=median(baseline["samples_ms"])/median(incumbent["samples_ms"])
    result["speedup_scope"]="selection only; independent matched winner/baseline remeasurement required"
    save(result);return result
