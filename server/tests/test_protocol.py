import json
import struct
import unittest
from pathlib import Path
from server import protocol as p
from server.loadgen import arrival_status,distribution,validate,workload
from server.run import validate as validate_server


class ProtocolTests(unittest.TestCase):
    def test_request_roundtrip_layout(self):
        wire=p.request(2**40,[1,42,50256],7)
        self.assertEqual(struct.unpack_from("!I",wire)[0],32)
        self.assertEqual(p.REQUEST.unpack_from(wire,4),(1,1,2**40,7,3))
        self.assertEqual(p.cancellation(3),struct.pack("!IHHQII",20,1,2,3,0,0))
    def test_limits(self):
        for args in ((0,[1],1),(1,[],1),(1,[True],1),(1,[-1],1),(1,[50257],1),(1,[1]*1024,1),(1,[1],0)):
            with self.assertRaises(ValueError):p.request(*args)
    def test_response(self):
        body=p.RESPONSE.pack(1,p.FINAL,8,0,2,3,0,10,20,30,40)+struct.pack("!II",2,3)
        result=p.decode_response(body);self.assertEqual(result["tokens"],[2,3]);self.assertEqual(result["queue_ns"],10)
        for bad in (body[:20],body+b"x",p.RESPONSE.pack(1,p.FIRST,8,0,2,3,0,0,0,0,0)+b"\0"*8):
            with self.assertRaises(ValueError):p.decode_response(bad)
    def test_workload_is_matched_and_poisson_schedule_is_absolute(self):
        cfg=json.loads((Path(__file__).parents[1]/"config_load.json").read_text())
        validate(cfg);a=workload(cfg,10);b=workload(cfg,10)
        self.assertEqual(a,b);self.assertGreater(len(a),1)
        self.assertTrue(all(x["due_seconds"]<y["due_seconds"] for x,y in zip(a,a[1:])))
        self.assertTrue(all(x["due_seconds"]<cfg["duration_seconds"] for x in a))
        self.assertNotEqual(a,workload(cfg,5))
    def test_histograms_use_individual_samples(self):
        d=distribution([0.5,1,2,50],[0,1,2,10])
        self.assertEqual(d["bin_counts"],[1,1,1]);self.assertEqual(d["overflow"],1)
        self.assertEqual(d["p50_ms"],1.5);self.assertAlmostEqual(d["p99_9_ms"],49.856)
        self.assertIsNone(distribution([],[0,1])["p50_ms"])
    def test_config_limits(self):
        cfg=json.loads((Path(__file__).parents[1]/"config_cpu.json").read_text());validate_server(cfg)
        for key,value in (("max_batch",True),("max_batch",33),("capacity",1025),("host","0.0.0.0")):
            copy=dict(cfg);copy[key]=value
            with self.assertRaises(ValueError):validate_server(copy)
    def test_late_arrival_and_capacity_admission(self):
        cfg=json.loads((Path(__file__).parents[1]/"config_load.json").read_text())
        self.assertEqual(arrival_status(11,0,cfg),"dropped_scheduler_lag")
        self.assertEqual(arrival_status(1,cfg["max_inflight"],cfg),"dropped_client_capacity")
        self.assertEqual(arrival_status(1,0,cfg),"offered")

if __name__=="__main__":unittest.main()
