"""Exercise completion stdout/stderr transport with a real CPU subprocess."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from autotune.core import parse_json, proposal
from autotune import proposer as proposer_module
from autotune.proposer import LocalLlamaProposer
from server.metadata import digest

# Pinned CLI program framing, distinct from generated model text:
# https://github.com/ggml-org/llama.cpp/blob/ddf48f85c933c1f438212a02eb21e7b001d2d37d/tools/completion/completion.cpp#L927-L949
CLI_TRAILER = ' [end of text]\n\n\n'


@unittest.skipUnless(os.name == 'posix', 'the supported completion CLI runs on Linux')
class ProposerTransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='tinyinfer-proposer-transport-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.executable = self.directory / 'completion-fixture'
        # The supported CLI routes generated text through its logger; pausing
        # that logger suppresses generation as well as diagnostics. This is an
        # executable transport fixture, not local-LLM quality evidence.
        self.executable.write_text('#!' + sys.executable + '\n' + r'''
import json
from pathlib import Path
import sys

args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1]
assert value('-ngl') == '0'
assert value('--temp') == '0.7'
assert '--no-display-prompt' in args and '--no-conversation' in args
prompt = Path(value('-f')).read_text(encoding='utf-8')
catalog = json.loads(next(line.removeprefix('CATALOG=') for line in prompt.splitlines() if line.startswith('CATALOG=')))
history = json.loads(next(line.removeprefix('PREVIOUS_RESULTS=') for line in prompt.splitlines() if line.startswith('PREVIOUS_RESULTS=')))
assert len(catalog) == 12
assert history['columns'] == ['attempt', 'config_id', 'status', 'median_ms', 'correctness_pass', 'error_type']
assert history['rows'] == [[-1, 0, 'initial_baseline', 1.0, None, None]]
assert prompt.endswith('\nJSON:\n')
generated = Path(value('-m')).read_bytes().decode('utf-8')
if '--log-disable' not in args:
    sys.stderr.write('completion fixture: prompt read from -f; CPU diagnostics\n')
    sys.stdout.write(generated)
''', encoding='utf-8')
        self.executable.chmod(0o755)

    def invoke(self, generated, expected=None):
        model = self.directory / 'fixture-output.txt'
        model.write_text(generated, encoding='utf-8')
        config = dict(executable=str(self.executable), model=str(model),
                      executable_sha256=digest(self.executable), model_sha256=digest(model),
                      max_output_tokens=64, context_tokens=2048, timeout_seconds=10)
        proposer = LocalLlamaProposer(config, self.directory / 'proposals', seed=17)
        history = dict(baseline=dict(proposal=proposal(0), samples_ms=[1.0] * 7),
                       attempts=[dict(attempt=3, status='proposing')])
        returned = proposer(history, 3)
        out = self.directory / 'proposals/proposal-0003'
        receipt = json.loads((out / 'proposal.json').read_text(encoding='utf-8'))
        self.assertEqual(receipt['exit_code'], 0)
        self.assertEqual(receipt['status'], 'returned_unvalidated')
        self.assertEqual(receipt['prompt_sha256'], digest(out / 'prompt.txt'))
        self.assertEqual(receipt['stdout.txt_sha256'], digest(out / 'stdout.txt'))
        self.assertEqual(receipt['stderr.txt_sha256'], digest(out / 'stderr.txt'))
        self.assertEqual(receipt['argv'][receipt['argv'].index('--seed') + 1], '20')
        self.assertEqual((out / 'stdout.txt').read_text(encoding='utf-8'), generated)
        self.assertEqual((out / 'stderr.txt').read_text(encoding='utf-8'),
                         'completion fixture: prompt read from -f; CPU diagnostics\n')
        self.assertEqual(returned, generated.strip() if expected is None else expected)
        self.assertNotIn('CPU diagnostics', returned)
        self.receipt = receipt
        return returned

    def assert_transport_receipt(self, returned, removed):
        self.assertEqual(self.receipt['stdout_transport'], dict(
            contract='llama.cpp_completion_eog_stdout_v1',
            upstream_revision='ddf48f85c933c1f438212a02eb21e7b001d2d37d',
            exact_terminal_suffix=CLI_TRAILER, maximum_suffix_removals=1,
            cli_framing_removed=removed))
        self.assertEqual(self.receipt['returned_payload_sha256'],
                         hashlib.sha256(returned.encode('utf-8')).hexdigest())

    def test_completion_stdout_survives_and_diagnostics_stay_separate(self):
        generated = '  ' + json.dumps(proposal(1), separators=(', ', ': ')) + '\n'
        returned = self.invoke(generated)
        self.assertEqual(parse_json(returned), proposal(1))

    def test_non_json_generation_is_preserved_for_strict_rejection(self):
        generated = json.dumps(proposal(1)) + ' [end of text]\n'
        returned = self.invoke(generated)
        with self.assertRaises(json.JSONDecodeError):
            parse_json(returned)

    def test_exact_cli_trailer_allows_strict_json_without_changing_raw_stdout(self):
        payload = '  ' + json.dumps(proposal(1))
        returned = self.invoke(payload + CLI_TRAILER, expected=payload.strip())
        self.assertEqual(parse_json(returned), proposal(1))
        self.assert_transport_receipt(returned, True)

    def test_actual_probe_extra_key_stays_rejected_after_cli_deframing(self):
        payload = ' {"configuration_id": 2, "config_id": 2, "tile": [64, 64, 32, 8], "stages": 2, "epilogue": "fused"}'
        returned = self.invoke(payload + CLI_TRAILER, expected=payload.strip())
        with self.assertRaisesRegex(ValueError, 'proposal schema'):
            parse_json(returned)
        self.assert_transport_receipt(returned, True)

    def test_model_marker_and_cli_marker_remove_only_last_once(self):
        payload = json.dumps(proposal(1)) + CLI_TRAILER
        returned = self.invoke(payload + CLI_TRAILER, expected=payload.strip())
        self.assertTrue(returned.endswith('[end of text]'))
        with self.assertRaises(json.JSONDecodeError):
            parse_json(returned)
        self.assert_transport_receipt(returned, True)

    def test_token_limit_truncation_without_cli_trailer_stays_unchanged(self):
        generated = '{"config_id": 2'
        returned = self.invoke(generated)
        with self.assertRaises(json.JSONDecodeError):
            parse_json(returned)
        self.assert_transport_receipt(returned, False)

    def test_decoder_leaves_middle_incomplete_and_crlf_trailers_untouched(self):
        decoder = proposer_module.decode_completion_stdout
        for text in ['plain text', '{"config_id": 2',
                     'before' + CLI_TRAILER + 'after',
                     'payload [end of text]', 'payload [end of text]\n',
                     'payload [end of text]\n\n',
                     'payload [end of text]\r\n\r\n\r\n',
                     'payload' + CLI_TRAILER + '\n']:
            with self.subTest(text=repr(text)):
                self.assertEqual(decoder(text), (text, False))

    def test_decoder_preserves_payload_whitespace_and_never_recurses(self):
        decoder = proposer_module.decode_completion_stdout
        payload = '  payload\r\n with whitespace  '
        self.assertEqual(decoder(payload + CLI_TRAILER), (payload, True))
        twice = payload + CLI_TRAILER + CLI_TRAILER
        self.assertEqual(decoder(twice), (payload + CLI_TRAILER, True))


if __name__ == '__main__':
    unittest.main(verbosity=2)
