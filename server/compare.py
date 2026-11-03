"""Pair batching-off/on runs only when offered workloads/configuration match."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .metadata import digest,metadata,now,output_directory,write


def compare(off,on):
    if off.get("status")!="completed" or on.get("status")!="completed" or off["mode"]!="off" or on["mode"]!="on":raise ValueError("two completed off/on runs required")
    if off["config"]!=on["config"]:raise ValueError("load configurations differ")
    a,b=off["server_record"],on["server_record"]
    for key in ("binary_sha256","build_receipt_sha256","artifact_manifest_sha256"):
        if a.get(key)!=b.get(key):raise ValueError("server binary/artifact/build differs: "+key)
    ca,cb=dict(a["config"]),dict(b["config"])
    for key in ("port","max_batch","max_wait_us"):ca.pop(key);cb.pop(key)
    if ca!=cb:raise ValueError("server settings other than port/maxbatch/maxwait differ")
    if a["environment"]["git"]["sha"]!=b["environment"]["git"]["sha"]:raise ValueError("Git commit differs")
    if len(off["results"])!=len(on["results"]):raise ValueError("rate counts differ")
    pairs=[]
    for x,y in zip(off["results"],on["results"]):
        if x["rate"]!=y["rate"] or x["workload_sha256"]!=y["workload_sha256"]:raise ValueError("offered arrivals/IDs/tokens differ")
        if not x["summary"]["all_offers_terminal"] or not y["summary"]["all_offers_terminal"]:raise ValueError("run did not drain")
        pairs.append(dict(rate=x["rate"],workload_sha256=x["workload_sha256"],off=x["summary"],on=y["summary"]))
    return dict(status="matched",pairs=pairs,scope="identical offered arrivals and payloads; drops/errors retained in both summaries; latency percentiles describe completed requests and must be read alongside completion/drop/error rates")


def main():
    p=argparse.ArgumentParser();p.add_argument("--off",required=True);p.add_argument("--on",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    off=json.loads(Path(a.off).read_text(encoding="utf-8"));on=json.loads(Path(a.on).read_text(encoding="utf-8"));result=compare(off,on)
    out=output_directory(a.output,Path(__file__).resolve().parents[1]);result.update(timestamp=now(),command_argv=sys.argv,environment=metadata(Path(__file__).resolve().parents[1]),
        off_sha256=digest(a.off),on_sha256=digest(a.on));write(out/"comparison.json",result)

if __name__=="__main__":main()
