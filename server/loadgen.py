"""Open-loop Poisson offered arrivals. Scheduler deadlines never depend on replies."""
from __future__ import annotations
import argparse
import asyncio
import bisect
import csv
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from . import protocol
from .metadata import canonical,digest,metadata,now,output_directory,write


def validate(c):
    required={"schema_version","seed","rates_per_second","duration_seconds","max_inflight","timeout_seconds",
              "drain_timeout_seconds","max_schedule_lag_ms","prompt_lengths","new_token_counts","histogram_edges_ms"}
    if set(c)!=required or type(c["schema_version"]) is not int or c["schema_version"]!=1:raise ValueError("load config schema")
    for k in ("seed","max_inflight"):
        if type(c[k]) is not int or c[k]<(0 if k=="seed" else 1):raise ValueError(k)
    for k in ("duration_seconds","timeout_seconds","drain_timeout_seconds","max_schedule_lag_ms"):
        if type(c[k]) not in (int,float) or not math.isfinite(c[k]) or c[k]<=0:raise ValueError(k)
    if c["drain_timeout_seconds"]<c["timeout_seconds"]:raise ValueError("drain must allow every request timeout")
    if not c["rates_per_second"] or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in c["rates_per_second"]):raise ValueError("rates")
    for k in ("prompt_lengths","new_token_counts"):
        if not c[k] or any(type(v) is not int or v<1 for v in c[k]):raise ValueError(k)
    if max(c["prompt_lengths"])+max(c["new_token_counts"])>1024:raise ValueError("context")
    e=c["histogram_edges_ms"]
    if len(e)<2 or e[0]!=0 or any(type(v) not in (int,float) or not math.isfinite(v) for v in e) or any(a>=b for a,b in zip(e,e[1:])):raise ValueError("histogram edges")
    return c


def workload(config,rate):
    # A rate-specific stream makes both off/on invocations byte-identical.
    rng=random.Random(f"{config['seed']}:{rate}");t=0.0;rows=[]
    while True:
        t+=rng.expovariate(rate)
        if t>=config["duration_seconds"]:return rows
        count=rng.choice(config["prompt_lengths"]);tokens=[rng.randrange(50257) for _ in range(count)]
        rows.append(dict(id=len(rows)+1,due_seconds=t,tokens=tokens,max_new_tokens=rng.choice(config["new_token_counts"])))


def percentile(values,p):
    if not values:return None
    ordered=sorted(values);x=(len(ordered)-1)*p/100;lo=math.floor(x);hi=math.ceil(x)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(x-lo)


def distribution(values,edges):
    bins=[0]*(len(edges)-1);overflow=0
    for value in values:
        i=bisect.bisect_right(edges,value)-1
        if i>=len(bins):overflow+=1
        elif i>=0:bins[i]+=1
        else:raise ValueError("negative latency")
    return dict(count=len(values),p50_ms=percentile(values,50),p99_ms=percentile(values,99),p99_9_ms=percentile(values,99.9),
                mean_ms=statistics.mean(values) if values else None,edges_ms=edges,bin_counts=bins,overflow=overflow,
                percentile_definition="linear interpolation at (n-1)*p/100; per-request raw samples, no batch averages")


def arrival_status(lag_ms,inflight,config):
    if lag_ms>config["max_schedule_lag_ms"]:return "dropped_scheduler_lag"
    if inflight>=config["max_inflight"]:return "dropped_client_capacity"
    return "offered"


async def sweep(config,rate,host,port):
    planned=workload(config,rate);records=[];tasks=set();epoch=time.perf_counter();start_ns=time.perf_counter_ns()
    async def issue(row,record):
        writer=None
        try:
            async with asyncio.timeout(config["timeout_seconds"]):
                reader,writer=await asyncio.open_connection(host,port)
                record["connected_ns"]=time.perf_counter_ns()-start_ns
                writer.write(protocol.request(row["id"],row["tokens"],row["max_new_tokens"]));await writer.drain()
                sent=time.perf_counter_ns();record["sent_ns"]=sent-start_ns
                first=None
                while True:
                    response=await protocol.read_response(reader);received=time.perf_counter_ns()
                    if response["id"]!=row["id"]:raise ValueError("response request ID mismatch")
                    if response["kind"]==protocol.FIRST:
                        if first is not None:raise ValueError("duplicate first token")
                        first=response["tokens"][0];record["ttft_ms"]=(received-start_ns-row["due_seconds"]*1e9)/1e6
                        record["send_to_first_ms"]=(received-sent)/1e6
                        continue
                    record["response"]=response;record["received_ns"]=received-start_ns
                    if response["status"]==0:
                        if first is None or len(response["tokens"])!=row["max_new_tokens"] or response["tokens"][0]!=first:raise ValueError("incomplete token result")
                        record["status"]="completed"
                        record["latency_ms"]=(received-start_ns-row["due_seconds"]*1e9)/1e6
                        record["send_to_final_ms"]=(received-sent)/1e6
                        for field in ("queue","first","infer","server"):record["server_"+field+"_ms"]=response[field+"_ns"]/1e6
                        record["socket_residual_ms"]=record["send_to_final_ms"]-record["server_server_ms"]
                    elif response["status"]==2:record["status"]="dropped_server_overload"
                    else:record.update(status="error_server",error_status=response["status"])
                    break
        except asyncio.CancelledError:
            record.update(status="error_drain_timeout");raise
        except Exception as e:record.update(status="error",error_type=type(e).__name__,error=str(e))
        finally:
            if writer:
                writer.close()
                try:await writer.wait_closed()
                except (OSError,asyncio.CancelledError):pass
    for row in planned:
        delay=epoch+row["due_seconds"]-time.perf_counter()
        if delay>0:await asyncio.sleep(delay)
        lag_ms=max(0,(time.perf_counter()-epoch-row["due_seconds"])*1000)
        record={"id":row["id"],"due_seconds":row["due_seconds"],"schedule_lag_ms":lag_ms,"status":"offered",
                "prompt_length":len(row["tokens"]),"max_new_tokens":row["max_new_tokens"]}
        records.append(record)
        record["status"]=arrival_status(lag_ms,len(tasks),config)
        if record["status"]=="offered":
            task=asyncio.create_task(issue(row,record));tasks.add(task);task.add_done_callback(tasks.discard)
    # Preserve the full offered-rate window even when the last arrival is early.
    await asyncio.sleep(max(0,epoch+config["duration_seconds"]-time.perf_counter()))
    if tasks:
        pending=list(tasks)
        done,unfinished=await asyncio.wait(pending,timeout=config["drain_timeout_seconds"])
        for task in unfinished:task.cancel()
        await asyncio.gather(*pending,return_exceptions=True)
    elapsed=time.perf_counter()-epoch
    counts={key:sum(r["status"]==key for r in records) for key in sorted({r["status"] for r in records})}
    completed=[r for r in records if r["status"]=="completed"]
    fields=("latency_ms","ttft_ms","send_to_first_ms","send_to_final_ms","server_queue_ms","server_first_ms","server_infer_ms","server_server_ms")
    summary={"offered":len(planned),"completed":len(completed),"dropped":sum(v for k,v in counts.items() if k.startswith("dropped_")),
             "errors":sum(v for k,v in counts.items() if k.startswith("error")),"status_counts":counts,
             "offered_rate_realized":len(planned)/config["duration_seconds"],"offered_rate_parameter":rate,
             "completed_per_offer_window_second":len(completed)/config["duration_seconds"],
             "completed_per_full_drain_second":len(completed)/elapsed,"full_drain_seconds":elapsed,
             "all_offers_terminal":all(r["status"]!="offered" for r in records),
             "histograms":{key:distribution([r[key] for r in completed],config["histogram_edges_ms"]) for key in fields}}
    if summary["offered"]!=summary["completed"]+summary["dropped"]+summary["errors"]:raise AssertionError("lost request accounting")
    return dict(rate=rate,workload=planned,workload_sha256=canonical(planned),records=records,summary=summary)


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--server-record",required=True)
    p.add_argument("--mode",choices=("off","on"),required=True);p.add_argument("--output",required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];path=Path(a.config).resolve(strict=True);config=validate(json.loads(path.read_text(encoding="utf-8")))
    server_path=Path(a.server_record).resolve(strict=True);server=json.loads(server_path.read_text(encoding="utf-8"));sc=server["config"]
    if (a.mode=="off" and (sc["max_batch"]!=1 or sc["max_wait_us"]!=0)) or (a.mode=="on" and sc["max_batch"]<=1):raise ValueError("batching mode/config mismatch")
    if max(config["prompt_lengths"])+max(config["new_token_counts"])>sc["capacity"] or max(config["new_token_counts"])>sc["max_new_tokens"]:raise ValueError("workload exceeds server limits")
    out=output_directory(a.output,root);start=time.perf_counter()
    journal=dict(status="started",timestamp=now(),command_argv=sys.argv,config=config,config_sha256=digest(path),mode=a.mode,
                 warmups=0,repeats=1,concurrency_limit=config["max_inflight"],warmup_scope="no hidden warmup; first native calls are retained; run independent explicit warmups and record them if changing this protocol",
                 server_record=server,server_record_sha256=digest(server_path),environment=metadata(root),results=[],
                 scope="offered due-time to socket final/first receipt includes scheduling, TCP connect/send, queue, maxwait, worker/native calls, eventfd/epoll and network; server times are monotonic durations and do not require clock alignment; socket_residual is send-to-final minus server accepted-to-complete, not a pure network timer")
    write(out/"load.json",journal)
    try:
        for rate in config["rates_per_second"]:
            result=asyncio.run(sweep(config,rate,sc["host"],sc["port"]))
            directory=out/("rate-"+str(len(journal["results"])));directory.mkdir()
            write(directory/"workload.json",result.pop("workload"));write(directory/"raw.json",result.pop("records"))
            write(directory/"summary.json",result);journal["results"].append(result);write(out/"load.json",journal)
        journal["status"]="completed" if all(r["summary"]["all_offers_terminal"] for r in journal["results"]) else "failed"
    except Exception as e:journal.update(status="failed",error_type=type(e).__name__,error=str(e))
    finally:
        journal.update(finished_timestamp=now(),wall_seconds=time.perf_counter()-start);write(out/"load.json",journal)
    return 0 if journal["status"]=="completed" else 1

if __name__=="__main__":raise SystemExit(main())
