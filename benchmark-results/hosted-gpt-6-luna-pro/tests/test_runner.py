import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runner


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


class LunaProRunnerTests(unittest.TestCase):
    def test_payload_keeps_old_prompt_schema_and_protocol(self):
        prompt = "# entities v2\n\nPASSAGE TOKENS:\n[0] Jane"
        payload = runner.request_payload("entities", prompt, SCHEMA)
        self.assertEqual(payload["model"], "openai/gpt-6-luna-pro")
        self.assertEqual(payload["messages"], [{"role": "user", "content": prompt}])
        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["max_tokens"], 16384)
        self.assertEqual(payload["reasoning"], {"effort": "none"})
        self.assertFalse(payload["include_reasoning"])
        self.assertEqual(payload["response_format"]["json_schema"]["schema"], SCHEMA)
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(payload["provider"]["order"], ["OpenAI"])
        self.assertFalse(payload["provider"]["allow_fallbacks"])
        self.assertTrue(payload["provider"]["require_parameters"])
        self.assertNotIn("gold", json.dumps(payload).lower())

    def test_metadata_selects_default_openai_endpoint_not_flex(self):
        supported = ["max_tokens", "reasoning_effort", "response_format", "structured_outputs"]
        model = {"id": runner.MODEL, "name": "GPT-6 Luna Pro", "supported_parameters": supported}
        endpoints = [
            {"provider_name": "OpenAI", "tag": "openai/flex", "status": 0,
             "supported_parameters": supported, "pricing": {"prompt": "flex"}},
            {"provider_name": "OpenAI", "tag": "openai", "status": 0,
             "supported_parameters": supported, "pricing": {"prompt": "default"}},
        ]
        with mock.patch.object(runner, "_json_get", side_effect=[{"data": [model]},
                                                                  {"data": {"endpoints": endpoints}}]):
            metadata = runner.fetch_model_metadata()
        self.assertEqual(metadata["default_provider_endpoint"]["tag"], "openai")
        self.assertEqual(metadata["default_provider_endpoint"]["pricing"]["prompt"], "default")

    def test_cache_identity_changes_with_task_prompt_or_split(self):
        first = runner.cache_identity("entities", "book", "prompt", SCHEMA, "test")
        self.assertEqual(first, runner.cache_identity("entities", "book", "prompt", SCHEMA, "test"))
        self.assertNotEqual(first, runner.cache_identity("entities", "book", "changed", SCHEMA, "test"))
        self.assertNotEqual(first, runner.cache_identity("quotes", "book", "prompt", SCHEMA, "test"))
        self.assertNotEqual(first, runner.cache_identity("entities", "book", "prompt", SCHEMA, "validation"))

    def test_prompt_must_use_the_frozen_token_serialization(self):
        template = "# entities v2\nIdentify entities"
        suffix = "\n\nPASSAGE TOKENS (the integers are the only valid token IDs):\n[0] Jane"
        self.assertTrue(runner.prompt_uses_exact_template(template + suffix, template))
        self.assertFalse(runner.prompt_uses_exact_template(template + " revised" + suffix, template))
        self.assertFalse(runner.prompt_uses_exact_template(template + "\n[0] Jane", template))

    def test_transport_429_retries_without_logging_secrets_or_error_body(self):
        body = {"id": "gen-safe", "provider": "OpenAI", "model": runner.MODEL,
                "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2}}
        good = mock.MagicMock()
        good.__enter__.return_value.read.return_value = json.dumps(body).encode()
        calls = [0]

        def fake_urlopen(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 1:
                raise HTTPError("https://openrouter.ai/api/v1/chat/completions", 429,
                                "rate limit", {}, io.BytesIO(b"private error details"))
            return good

        waits = []
        with mock.patch.object(runner.urllib.request, "urlopen", side_effect=fake_urlopen):
            received, attempts = runner.call_openrouter("sentinel-secret", {}, wait=waits.append)
        self.assertEqual(received["id"], "gen-safe")
        self.assertEqual(calls[0], 2)
        self.assertEqual(waits, [1])
        self.assertEqual([item["http_status"] for item in attempts], [429, 200])
        self.assertNotIn("sentinel-secret", repr(attempts))

    def test_nonretryable_error_is_not_retried_or_exposed(self):
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

    def test_atomic_cache_write_replaces_completed_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            runner.atomic_json(path, {"stage": 1})
            runner.atomic_json(path, {"stage": 2})
            self.assertEqual(json.loads(path.read_text()), {"stage": 2})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
