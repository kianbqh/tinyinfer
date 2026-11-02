import json
import math
import sys
import unittest
from autotune.core import RandomProposer,gate,median,parse_json,proposal,search


def record(identifier,ms=100,status="passed"):
    return dict(proposal=proposal(identifier),status=status,correctness_pass=status=="passed",samples_ms=[ms]*9)


class GateTests(unittest.TestCase):
    def test_exact_boundary_and_correctness(self):
        base=record(0)
        self.assertTrue(gate(record(1,97),base));self.assertFalse(gate(record(1,97.00001),base))
        self.assertFalse(gate(record(1,20,"test_failed"),base));self.assertTrue(gate(record(1,100),record(0,104)))
    def test_bad_samples_cannot_win(self):
        for bad in ([0]*9,[math.nan]*9,[math.inf]*9,[-1]*9,[True]*9,[1]*6):
            with self.assertRaises(ValueError):median(bad)
    def test_schema_exact_catalog_all_twelve(self):
        for i in range(12):self.assertEqual(parse_json(json.dumps(proposal(i))),proposal(i))
        for raw in ('{}','{"config_id":0,"config_id":1}','NaN','[]','```json\n{}\n```',json.dumps({**proposal(0),"shell":"touch x"})):
            with self.assertRaises(ValueError):parse_json(raw)
        for field,value in (("config_id",True),("tile",[32,32,16,8]),("stages",True),("epilogue","generated code")):
            bad=proposal(0);bad[field]=value
            with self.assertRaises(ValueError):parse_json(json.dumps(bad))
    def test_fake_proposer_accounting(self):
        raws=["invalid",json.dumps(proposal(0)),json.dumps(proposal(1)),json.dumps(proposal(1)),json.dumps(proposal(2)),json.dumps(proposal(3))]
        seen=[]
        def fake(history,i):return raws[i]
        def evaluate(value,i):
            seen.append(value["config_id"])
            return record(value["config_id"],90,{1:"compile_failed",2:"test_failed",3:"passed"}[value["config_id"]])
        result=search(fake,evaluate,record(0),proposal_budget=6,evaluation_budget=3)
        self.assertEqual(seen,[1,2,3]);self.assertEqual(result["evaluations"],3);self.assertEqual(result["proposal_attempts"],6)
        self.assertEqual((result["invalid"],result["duplicates"],result["compile_failures"],result["test_failures"],result["test_passes"],result["accepted"]),(1,2,1,1,1,1))
        self.assertTrue(result["budget_exhausted"])
    def test_incomplete_search_is_not_matched_budget(self):
        result=search(lambda h,i:json.dumps(proposal(0)),lambda v,i:self.fail("duplicate evaluated"),record(0),proposal_budget=4,evaluation_budget=3)
        self.assertEqual(result["status"],"incomplete_proposal_cap");self.assertFalse(result["budget_exhausted"])
    def test_incumbent_updates_require_three_percent_each_time(self):
        result=search(lambda h,i:json.dumps(proposal(i+1)),lambda v,i:record(v["config_id"],[96,95][i]),record(0),proposal_budget=2,evaluation_budget=2)
        self.assertEqual(result["accepted"],1);self.assertEqual(result["best"]["proposal"]["config_id"],1)
    def test_random_reproducibility_and_stdlib_only(self):
        a=RandomProposer(41);b=RandomProposer(41)
        self.assertEqual([a({},i) for i in range(30)],[b({},i) for i in range(30)])
        self.assertNotIn("torch",sys.modules);self.assertNotIn("transformers",sys.modules)
    def test_fixture_mismatch_is_rejected(self):
        base=record(0);base["fixture"]={"bias_hash":"a"}
        def evaluate(v,i):
            out=record(1,50);out["fixture"]={"bias_hash":"b"};return out
        out=search(lambda h,i:json.dumps(proposal(1)),evaluate,base,proposal_budget=1,evaluation_budget=1)
        self.assertEqual(out["accepted"],0);self.assertEqual(out["evaluation_errors"],1)

if __name__=="__main__":unittest.main()
