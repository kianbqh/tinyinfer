import copy
import json
import unittest
from pathlib import Path
from autotune.run import RealEvaluator,validate


class ConfigTests(unittest.TestCase):
    def config(self):
        c=json.loads((Path(__file__).parents[1]/"config.example.json").read_text())
        c["llama"]["model_sha256"]="a"*64;c["llama"]["executable_sha256"]="b"*64
        return c
    def test_example_needs_actual_hashes(self):
        c=json.loads((Path(__file__).parents[1]/"config.example.json").read_text())
        with self.assertRaises(ValueError):validate(c)
        validate(self.config())
    def test_three_seeds_and_matching_budgets(self):
        for patch in ({"seeds":[1,2]},{"seeds":[1,1,2]},{"seeds":[True,2,3]},{"proposal_budget":1,"evaluation_budget":2},{"evaluation_budget":12},{"schema_version":True}):
            c=self.config();c.update(patch)
            with self.assertRaises(ValueError):validate(c)
    def test_seed_directory_is_in_candidate_scratch_identity(self):
        first=RealEvaluator(self.config(),Path("/repo/results/v2/test/seed-17/random"),"test")
        second=RealEvaluator(self.config(),Path("/repo/results/v2/test/seed-41/random"),"test")
        self.assertEqual(first.out.relative_to(first.run_root).as_posix(),"seed-17/random")
        self.assertNotEqual(first.out.relative_to(first.run_root),second.out.relative_to(second.run_root))

if __name__=="__main__":unittest.main()
