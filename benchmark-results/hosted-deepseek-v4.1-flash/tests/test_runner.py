import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runner
import validate_results


SCHEMA = {
    "type": "object",
    "properties": {"entities": {"type": "array", "items": {
        "type": "object",
        "properties": {"start_token": {"type": "integer"}},
        "required": ["start_token"],
        "additionalProperties": False,
    }}},
    "required": ["entities"],
    "additionalProperties": False,
}


class OpenRouterRunnerTests(unittest.TestCase):
    def test_payload_preserves_only_exact_prompt_and_original_schema(self):
        prompt = "# entities v2\n\nPASSAGE TOKENS:\n[0] Jane"
        payload = runner.request_payload("entities", prompt, SCHEMA)
        self.assertEqual(payload["model"], "deepseek/deepseek-v4.1-flash")
        self.assertEqual(payload["messages"], [{"role": "user", "content": prompt}])
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 16384)
        self.assertEqual(payload["reasoning"], {"effort": "none"})
        self.assertFalse(payload["include_reasoning"])
        self.assertEqual(payload["response_format"]["json_schema"]["schema"], SCHEMA)
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(payload["provider"]["order"], ["DeepInfra"])
        self.assertFalse(payload["provider"]["allow_fallbacks"])
        self.assertTrue(payload["provider"]["require_parameters"])
        self.assertNotIn("gold", json.dumps(payload).lower())

    def test_cache_identity_is_stable_and_separates_changed_inputs(self):
        first = runner.cache_identity("entities", "book", "prompt", SCHEMA, "test")
        self.assertEqual(first, runner.cache_identity("entities", "book", "prompt", SCHEMA, "test"))
        self.assertNotEqual(first, runner.cache_identity("entities", "book", "changed", SCHEMA, "test"))
        self.assertNotEqual(first, runner.cache_identity("quotes", "book", "prompt", SCHEMA, "test"))
        self.assertNotEqual(first, runner.cache_identity("entities", "book", "prompt", SCHEMA, "smoke"))

    def test_prompt_must_append_the_exact_old_token_serialization(self):
        template = "# entities v2\nIdentify entities"
        suffix = "\n\nPASSAGE TOKENS (the integers are the only valid token IDs):\n[0] Jane"
        self.assertTrue(runner.prompt_uses_exact_template(template + suffix, template))
        self.assertFalse(runner.prompt_uses_exact_template(template + " revised" + suffix, template))
        self.assertFalse(runner.prompt_uses_exact_template(template + "\n[0] Jane", template))

    def test_transport_429_retries_then_returns_the_single_completion(self):
        response_body = {"id": "gen-safe", "provider": "DeepInfra", "model": runner.MODEL,
                         "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                         "usage": {"prompt_tokens": 12, "completion_tokens": 2}}
        good_response = mock.MagicMock()
        good_response.__enter__.return_value.read.return_value = json.dumps(response_body).encode()
        calls = [0]

        def fake_urlopen(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 1:
                raise HTTPError("https://openrouter.ai/api/v1/chat/completions", 429,
                                "rate limit", {}, io.BytesIO(b"private error details"))
            return good_response

        waits = []
        with mock.patch.object(runner.urllib.request, "urlopen", side_effect=fake_urlopen):
            body, attempts = runner.call_openrouter("sentinel-secret", {}, wait=waits.append)
        self.assertEqual(body["id"], "gen-safe")
        self.assertEqual(calls[0], 2)
        self.assertEqual(waits, [1])
        self.assertEqual([item["http_status"] for item in attempts], [429, 200])
        self.assertNotIn("sentinel-secret", repr(attempts))

    def test_nonretryable_api_error_does_not_expose_error_body_or_retry(self):
        calls = []

        def fake_urlopen(*args, **kwargs):
            calls.append(1)
            raise HTTPError("https://openrouter.ai/api/v1/chat/completions", 400,
                            "bad request", {}, io.BytesIO(b"sentinel-secret"))

        with mock.patch.object(runner.urllib.request, "urlopen", side_effect=fake_urlopen):
            with self.assertRaises(runner.TransportFailure) as raised:
                runner.call_openrouter("sentinel-secret", {}, wait=lambda _: None)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("sentinel-secret", str(raised.exception))
        self.assertEqual(raised.exception.attempts[0]["http_status"], 400)

    def test_atomic_json_replaces_a_complete_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            runner.atomic_json(path, {"stage": 1})
            runner.atomic_json(path, {"stage": 2})
            self.assertEqual(json.loads(path.read_text()), {"stage": 2})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_split_audit_respects_coref_tenfold_union(self):
        validation = {f"book-{index}" for index in range(10)}
        test = validation | {f"book-{index}" for index in range(10, 100)}
        self.assertEqual(validate_results._validate_split_overlap("coref", validation, test), 10)
        with self.assertRaisesRegex(AssertionError, "coref"):
            validate_results._validate_split_overlap("coref", validation, test - {"book-0"})

    def test_split_audit_rejects_holdout_overlap_for_other_tasks(self):
        with self.assertRaisesRegex(AssertionError, "entities"):
            validate_results._validate_split_overlap("entities", {"book-a"}, {"book-a"})
        self.assertEqual(validate_results._validate_split_overlap("quotes", {"book-a"}, {"book-b"}), 0)


if __name__ == "__main__":
    unittest.main()
