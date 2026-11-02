"""History capacity/retention regression; fixtures are not model outcomes."""
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

from autotune.core import proposal
from autotune.proposer import LocalLlamaProposer
from server.metadata import digest

COLUMNS = ['attempt', 'config_id', 'status', 'median_ms', 'correctness_pass', 'error_type']
STATUSES = ['passed', 'compile_failed', 'test_failed', 'evaluation_error', 'invalid', 'duplicate', 'proposal_error']


def capacity_history(previous=64, error_characters=2048):
    """Conservative one-extra-row history beyond the 64-proposal cap."""
    samples = [0.032768001556396484 + i * 0.000000001 for i in range(101)]
    baseline = dict(proposal=proposal(0), status='passed', correctness_pass=True, samples_ms=list(samples))
    attempts = []
    for index in range(previous):
        status = STATUSES[index % len(STATUSES)]
        row = dict(attempt=index, status=status)
        if status not in ('invalid', 'proposal_error'):
            row['proposal'] = proposal(index % 12)
        if status in ('passed', 'compile_failed', 'test_failed', 'evaluation_error'):
            row['evaluation'] = dict(correctness_pass=status == 'passed', samples_ms=list(samples),
                                     error_type='UnicodeDecodeError', error='TRACEBACK_BODY_' + 'x' * error_characters)
        if status in ('invalid', 'proposal_error', 'evaluation_error'):
            row.update(error_type='JSONDecodeError', error='UNFAVORABLE_ERROR_' + 'y' * error_characters)
        attempts.append(row)
    attempts.append(dict(attempt=previous, status='proposing'))
    return dict(baseline=baseline, attempts=attempts)


def wide_capacity_history():
    history = capacity_history()
    history['baseline']['samples_ms'] = [1.7976931348623157e308] * 101
    for row in history['attempts'][:-1]:
        row['status'] = 'evaluation_error'
        row['error_type'] = 'LongNativeEvaluationExceptionXXX'
        row['evaluation'] = dict(samples_ms=[1.7976931348623157e308] * 101, correctness_pass=False)
    return history


def capture_prompt(history):
    """Use the actual proposer/file/subprocess interface with no model."""
    with tempfile.TemporaryDirectory(prefix='tinyinfer-history-') as temporary:
        root = Path(temporary)
        executable = root / 'completion-fixture'
        executable.write_text('#!' + sys.executable + '\nimport sys\nsys.stdout.write(' +
                              repr(json.dumps(proposal(1))) + ')\n', encoding='utf-8')
        executable.chmod(0o755)
        model = root / 'fixture-not-a-model.txt';model.write_text('fixture only', encoding='utf-8')
        config = dict(executable=str(executable), model=str(model), executable_sha256=digest(executable),
                      model_sha256=digest(model), max_output_tokens=384, context_tokens=8192, timeout_seconds=10)
        proposer = LocalLlamaProposer(config, root / 'proposals', seed=17)
        attempt = history['attempts'][-1]['attempt']
        proposer(history, attempt)
        return (root / 'proposals' / ('proposal-%04d' % attempt) / 'prompt.txt').read_text(encoding='utf-8')


@unittest.skipUnless(os.name == 'posix', 'supported completion executable runs on Linux')
class ProposerHistoryTests(unittest.TestCase):
    def table(self, prompt):
        line = next(s for s in prompt.splitlines() if s.startswith('PREVIOUS_RESULTS='))
        value = json.loads(line.removeprefix('PREVIOUS_RESULTS='))
        self.assertIsInstance(value, dict)
        self.assertEqual(value['columns'], COLUMNS)
        return value['rows']

    def test_all_prior_attempts_and_real_statistics_are_retained(self):
        history = capacity_history()
        untouched = copy.deepcopy(history)
        prompt = capture_prompt(history)
        rows = self.table(prompt)
        self.assertEqual(history, untouched, 'raw evidence/history must not be mutated')
        self.assertEqual(len(rows), 65)
        self.assertEqual(rows[0], [-1, 0, 'initial_baseline', statistics.median(history['baseline']['samples_ms']), True, None])
        self.assertEqual([row[0] for row in rows[1:]], list(range(64)))
        for actual, source in zip(rows[1:], history['attempts'][:-1]):
            evaluation = source.get('evaluation', {})
            measured = statistics.median(evaluation['samples_ms']) if evaluation.get('samples_ms') else None
            self.assertEqual(actual, [source['attempt'], source.get('proposal', {}).get('config_id'), source['status'],
                                      measured, evaluation.get('correctness_pass'),
                                      source.get('error_type', evaluation.get('error_type'))])
        self.assertNotIn('TRACEBACK_BODY_', prompt)
        self.assertNotIn('UNFAVORABLE_ERROR_', prompt)
        self.assertNotIn('samples_ms', prompt)
        self.assertIn('Lower median_ms is better', prompt)

    def test_capacity_does_not_grow_with_raw_samples_or_long_error_bodies(self):
        first = capacity_history(error_characters=1)
        second = copy.deepcopy(first)
        for row in second['attempts'][:-1]:
            if 'error' in row:
                row['error'] = 'different full raw error body' * 8192
            if 'evaluation' in row:
                row['evaluation']['error'] = 'different full native traceback' * 8192
                middle = statistics.median(row['evaluation']['samples_ms'])
                row['evaluation']['samples_ms'] = [middle / 2] * 50 + [middle] + [middle * 2] * 50
        self.assertEqual(capture_prompt(first), capture_prompt(second))

    def test_conservative_wide_rows_keep_capacity_bounded(self):
        # Full-width finite floats and a conservative 32-character error type;
        # actual tokenizer validation is a separate same-GGUF CPU-only gate.
        history = wide_capacity_history()
        prompt = capture_prompt(history)
        self.assertEqual(len(self.table(prompt)), 65)
        self.assertLess(len(prompt.encode('utf-8')), 9000, 'byte budget only; actual model tokens are checked separately')


if __name__ == '__main__':
    unittest.main(verbosity=2)
