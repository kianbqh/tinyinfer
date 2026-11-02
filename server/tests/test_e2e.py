"""Runs the real compiled CPU server only when TI_SERVER_BINARY is supplied."""
import os
import json
import select
import signal
import socket
import struct
import subprocess
import time
import unittest
from server import protocol as p


def exact(sock,n):
    data=b""
    while len(data)<n:
        chunk=sock.recv(n-len(data))
        if not chunk:raise EOFError("connection closed")
        data+=chunk
    return data


def response(sock):
    size=struct.unpack("!I",exact(sock,4))[0]
    if not p.RESPONSE.size<=size<=p.MAX_RESPONSE:raise ValueError("frame bound")
    return p.decode_response(exact(sock,size))


@unittest.skipUnless(os.environ.get("TI_SERVER_BINARY"),"native server build is a later execution gate")
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        probe=socket.socket();probe.bind(("127.0.0.1",0));cls.port=probe.getsockname()[1];probe.close()
        cls.proc=subprocess.Popen([os.environ["TI_SERVER_BINARY"],"--model","cpu","--port",str(cls.port),
                                   "--max-batch","8","--max-wait-us","100000","--max-pending","32","--per-connection","16","--send-chunk-bytes","7"],
                                  stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        print(json.dumps(dict(event="cpu_oracle_server_started",pid=cls.proc.pid,port=cls.port,argv=cls.proc.args)),flush=True)
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            if cls.proc.poll() is not None:raise RuntimeError(cls.proc.communicate())
            ready,_,_=select.select([cls.proc.stdout],[],[],0.1)
            if ready and '"ready":true' in cls.proc.stdout.readline():return
        cls.proc.terminate();cls.proc.wait(timeout=10);raise TimeoutError("CPU server startup")
    @classmethod
    def tearDownClass(cls):
        cls.proc.send_signal(signal.SIGTERM)
        try:stdout,stderr=cls.proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:cls.proc.kill();stdout,stderr=cls.proc.communicate();raise
        if cls.proc.returncode:raise AssertionError(f"server exit {cls.proc.returncode}: {stderr}")
        if "ThreadSanitizer" in stderr:raise AssertionError(stderr)
        print(json.dumps(dict(event="cpu_oracle_server_stopped",pid=cls.proc.pid,exit_code=cls.proc.returncode,stderr=stderr)),flush=True)
    def connect(self):
        s=socket.create_connection(("127.0.0.1",self.port),timeout=5);s.settimeout(5);return s
    def test_fragmented_and_coalesced_mixed_prompts(self):
        with self.connect() as s:
            frame=p.request(1,[4,5,6],3)
            for byte in frame:s.sendall(bytes([byte]))
            s.sendall(p.request(2,[8],2)+p.request(3,[1,2],1))
            first={};final={}
            while len(final)<3:
                r=response(s);(first if r["kind"]==p.FIRST else final)[r["id"]]=r
            self.assertEqual(final[1]["tokens"],[10,15,21]);self.assertEqual(final[2]["tokens"],[10,13]);self.assertEqual(final[3]["tokens"],[5])
            self.assertEqual(set(first),{1,2,3});self.assertTrue(all(r["status"]==0 for r in final.values()))
    def test_invalid_prefix_disconnect_and_fd_reuse(self):
        with self.connect() as s:
            s.sendall(b"\xff"*4);self.assertEqual(s.recv(1),b"")
        s=self.connect();s.sendall(p.request(1,[1]*512,128));s.close()
        with self.connect() as s:
            s.sendall(p.request(1,[1],1));r=response(s);self.assertEqual(r["kind"],p.FIRST)
            r=response(s);self.assertEqual(r["tokens"],[3]);self.assertEqual(r["status"],0)
    def test_cancellation(self):
        with self.connect() as s:
            s.sendall(p.request(6,[1]*512,128)+p.cancellation(6))
            while True:
                r=response(s)
                if r["kind"]==p.FINAL:break
            self.assertEqual(r["id"],6);self.assertEqual(r["status"],3)
    def test_duplicate_ids_close(self):
        with self.connect() as s:
            s.sendall(p.request(1,[1],1)*2)
            # Request-ID ambiguity is a protocol violation, terminating the
            # connection and cancelling any earlier queued request.
            self.assertEqual(s.recv(1),b"")
    def test_output_backpressure_closes_and_reaps(self):
        probe=socket.socket();probe.bind(("127.0.0.1",0));port=probe.getsockname()[1];probe.close()
        proc=subprocess.Popen([os.environ["TI_SERVER_BINARY"],"--model","cpu","--port",str(port),"--max-output-bytes","64"],
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        print(json.dumps(dict(event="cpu_oracle_backpressure_started",pid=proc.pid,port=port,argv=proc.args)),flush=True)
        try:
            ready,_,_=select.select([proc.stdout],[],[],10)
            self.assertTrue(ready);self.assertIn('"ready":true',proc.stdout.readline())
            with socket.create_connection(("127.0.0.1",port),timeout=5) as s:
                # The 68-byte framed FIRST response exceeds the configured
                # 64-byte bound: deterministic backpressure without a load test.
                s.sendall(p.request(1,[1],1));self.assertEqual(s.recv(1),b"")
        finally:
            proc.terminate()
            try:stdout,stderr=proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.communicate();raise
            print(json.dumps(dict(event="cpu_oracle_backpressure_stopped",pid=proc.pid,exit_code=proc.returncode,stderr=stderr)),flush=True)
            self.assertEqual(proc.returncode,0)

if __name__=="__main__":unittest.main()
