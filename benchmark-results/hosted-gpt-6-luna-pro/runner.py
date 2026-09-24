"""Run the frozen BookNLP-style test cohorts against GPT-6 Luna Pro.

The licensed documents and gold labels live in the private local experiment
checkout. This script imports that checkout's prompt, split, parser, and scorer
code so the public repository never needs to redistribute those materials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL = "openai/gpt-6-luna-pro"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_URL = "https://openrouter.ai/api/v1/models/openai/gpt-6-luna-pro/endpoints"
BENCHMARK_VERSION = "booknlp-local-llm-2026"
PROVIDER = "OpenAI"
CONTEXT_TOKENS = 32768
MAX_OUTPUT_TOKENS = 16384
REASONING_EFFORT = "none"
KEY_FILE = Path("/home/drew/.config/openrouter/api_key")
RETRYABLE_HTTP = {408, 425, 429, 500, 502, 503, 504}
TASK_ORDER = ("entities", "events", "coref", "quotes", "speakers", "supersenses")
TASK_SIZES = {"entities": 10, "events": 30, "coref": 100,
              "quotes": 10, "speakers": 10, "supersenses": 35}
BUDGET_USD = 5.0


class TransportFailure(RuntimeError):
    def __init__(self, message: str, attempts: list[dict[str, Any]]):
        super().__init__(message)
        self.attempts = attempts


def schema_response_format(task: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap the original task schema in OpenRouter strict JSON Schema mode."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "booknlp_" + re.sub(r"[^a-zA-Z0-9_-]", "_", task),
            "strict": True,
            "schema": schema,
        },
    }


def request_payload(task: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "reasoning": {"effort": REASONING_EFFORT},
        "include_reasoning": False,
        "response_format": schema_response_format(task, schema),
        "provider": {
            "order": [PROVIDER],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        "usage": {"include": True},
    }


def prompt_uses_exact_template(prompt: str, template: str) -> bool:
    return prompt.startswith(template + "\n\nPASSAGE TOKENS (the integers are the only valid token IDs):\n")


def cache_identity(task: str, doc_id: str, prompt: str, schema: dict[str, Any], split: str) -> str:
    body = request_payload(task, prompt, schema)
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    material = "\0".join((BENCHMARK_VERSION, MODEL, split, task, doc_id, canonical))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key and KEY_FILE.is_file():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("OpenRouter credentials are not configured")
    return key


def _json_get(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def fetch_model_metadata() -> dict[str, Any]:
    """Confirm the exact slug and a pinned provider's required capabilities."""
    models = _json_get("https://openrouter.ai/api/v1/models").get("data", [])
    model = next((item for item in models if item.get("id") == MODEL), None)
    if model is None:
        raise RuntimeError(f"OpenRouter no longer lists requested model {MODEL}")
    endpoint_data = _json_get(MODEL_URL).get("data", {})
    endpoints = endpoint_data.get("endpoints", []) if isinstance(endpoint_data, dict) else []
    # The OpenAI model has multiple endpoint tags (flex/default/fast). The
    # benchmark does not send service_tier, so capability-check the default
    # endpoint rather than whichever OpenAI endpoint happens to be listed first.
    endpoint = next((item for item in endpoints
                     if item.get("provider_name", "").casefold() == PROVIDER.casefold()
                     and item.get("tag") == "openai"), None)
    if endpoint is None:
        raise RuntimeError(f"OpenRouter has no {PROVIDER} endpoint for {MODEL}")
    supported = set(endpoint.get("supported_parameters", []))
    required = {"max_tokens", "reasoning_effort", "response_format"}
    missing = required - supported
    if missing:
        raise RuntimeError(f"Pinned provider lacks required API parameters: {', '.join(sorted(missing))}")
    if endpoint.get("status") not in (0, None):
        raise RuntimeError(f"Pinned provider reports unavailable status {endpoint.get('status')}")
    if "structured_outputs" not in supported:
        raise RuntimeError("Pinned provider does not advertise structured-output enforcement")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {key: model.get(key) for key in (
            "id", "name", "context_length", "architecture", "pricing",
            "supported_parameters", "created")},
        "default_provider_endpoint": {
            key: endpoint.get(key) for key in (
                "provider_name", "name", "tag", "status", "context_length",
                "max_completion_tokens", "pricing", "supported_parameters")},
    }


def _usage(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage") or {}
    keep = ("prompt_tokens", "completion_tokens", "total_tokens", "cost",
            "cost_details", "prompt_tokens_details", "completion_tokens_details")
    return {key: usage[key] for key in keep if key in usage}


def call_openrouter(api_key: str, payload: dict[str, Any], *, timeout: int = 1200,
                    max_attempts: int = 4, wait=time.sleep) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Retry transport/provider errors only; any received model answer is final."""
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            attempts.append({"attempt": attempt, "status": "response_received",
                             "http_status": 200, "latency_seconds": time.monotonic() - started})
            return body, attempts
        except urllib.error.HTTPError as exc:
            retry = exc.code in RETRYABLE_HTTP and attempt < max_attempts
            attempts.append({"attempt": attempt, "status": "http_error", "http_status": exc.code,
                             "latency_seconds": time.monotonic() - started,
                             "retrying": retry})
            # Never persist/print provider error bodies; they may echo request data.
            if not retry:
                raise TransportFailure(f"OpenRouter HTTP {exc.code}; transport request failed", attempts) from None
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError,
                json.JSONDecodeError) as exc:
            retry = attempt < max_attempts
            attempts.append({"attempt": attempt, "status": "transport_error",
                             "error_type": type(exc).__name__,
                             "latency_seconds": time.monotonic() - started,
                             "retrying": retry})
            if not retry:
                raise TransportFailure(f"OpenRouter transport failed ({type(exc).__name__})", attempts) from None
        if retry:
            wait(min(2 ** (attempt - 1), 30))
    raise AssertionError("unreachable")


def _load_experiment(root: Path):
    sys.path.insert(0, str(root / "src"))
    from booknlp_experiment import benchmark
    if benchmark.ROOT.resolve() != root.resolve():
        raise RuntimeError("Imported benchmark package from a different experiment checkout")
    return benchmark


def _verify_prompts_and_splits(benchmark, repository_dir: Path, task_docs: dict[str, list[dict]], split: str):
    base = repository_dir.parent / "booknlp-local-llm-2026"
    split_manifest = json.loads((base / "metadata" / "splits.json").read_text(encoding="utf-8"))
    selections = {}
    for task in TASK_ORDER:
        prompt_path = base / "prompts" / f"{task}.md"
        local_prompt_path = benchmark.ROOT / "prompts" / f"{task}.md"
        if local_prompt_path.read_bytes() != prompt_path.read_bytes():
            raise RuntimeError(f"Local {task} prompt differs from the frozen public benchmark prompt")
        docs, split_info = benchmark.select_task_split(task, task_docs[task], split)
        expected_ids = split_manifest[task][f"{split}_ids"]
        actual_ids = [doc["doc_id"] for doc in docs]
        if actual_ids != expected_ids:
            raise RuntimeError(f"{task} {split} document IDs/order differ from frozen manifest")
        if len(actual_ids) != TASK_SIZES[task] if split == "test" else len(actual_ids) < 1:
            raise RuntimeError(f"Unexpected {task} {split} cohort size: {len(actual_ids)}")
        if len(set(actual_ids)) != len(actual_ids):
            raise RuntimeError(f"Duplicate document ID in {task} {split} cohort")
        selections[task] = (docs, split_info, hashlib.sha256(prompt_path.read_bytes()).hexdigest())
    return selections


def _task_docs(benchmark) -> dict[str, list[dict]]:
    docs = benchmark.load_docs()
    semcor_path = benchmark.ROOT / "datasets/normalized/semcor.jsonl"
    if not semcor_path.exists():
        from booknlp_experiment.data import save_semcor_normalized
        save_semcor_normalized(semcor_path)
    semcor = [json.loads(line) for line in semcor_path.read_text(encoding="utf-8").splitlines() if line]
    return {task: semcor if task == "supersenses" else docs for task in TASK_ORDER}


def _read_cached_row(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _resolve_completion(response: dict[str, Any]) -> tuple[str | None, str | None]:
    try:
        choice = response["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content, choice.get("finish_reason")
        return None, choice.get("finish_reason") or "missing_content"
    except (KeyError, IndexError, TypeError):
        return None, "missing_choice"


def _evaluate_response(benchmark, task: str, doc: dict[str, Any], cached: dict[str, Any]) -> dict[str, Any]:
    response = cached["response"]
    content, finish_reason = _resolve_completion(response)
    parsed: dict[str, Any] = {}
    valid_json = False
    valid_schema = False
    error = None
    if content is None:
        error = f"no textual completion (finish_reason={finish_reason})"
        prediction = []
    else:
        try:
            parsed = json.loads(content)
            valid_json = True
            valid_schema = benchmark.validate_generated_shape(task, parsed)
            prediction = benchmark.clean_preds(task, parsed, len(doc["tokens"]))
            if not valid_schema:
                error = "generated JSON did not match the required task schema"
        except (json.JSONDecodeError, TypeError):
            prediction = []
            error = "generated completion was not parseable JSON"
    field = {"entities": "entities", "events": "events", "coref": "mentions",
             "quotes": "quotes", "speakers": "speakers", "supersenses": "supersenses"}[task]
    raw_values = parsed.get(field, []) if isinstance(parsed, dict) else []
    raw_count = len(raw_values) if isinstance(raw_values, list) else 0
    score = benchmark.metric_for(task, doc["gold"], prediction)
    usage = cached.get("usage", {})
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "model": MODEL,
        "model_runtime": "OpenRouter chat completions API",
        "task": task,
        "split": cached["split"],
        "doc_id": doc["doc_id"],
        "prompt_version": cached["prompt_version"],
        "prompt_hash": cached["prompt_hash"],
        "request_hash": cached["request_hash"],
        "prediction": prediction,
        "valid_json": valid_json,
        "valid_schema": valid_schema,
        "discarded_items": max(0, raw_count - len(prediction)),
        "error": error,
        "finish_reason": finish_reason,
        "provider": response.get("provider"),
        "service_tier": response.get("service_tier"),
        "resolved_model": response.get("model"),
        "generation_id": response.get("id"),
        "latency_seconds": cached.get("latency_seconds"),
        "usage": usage,
        "metrics": score,
        "retry_count": max(0, len(cached.get("attempts", [])) - 1),
    }


def _sum_numeric(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        vals = [row.get("usage", {}).get(key) for row in rows]
        vals = [float(value) for value in vals if isinstance(value, (int, float))]
        out[key] = sum(vals)
    return out


def _task_summary(benchmark, task: str, docs: list[dict], rows: list[dict]) -> dict[str, Any]:
    scores = [row["metrics"] for row in rows]
    result: dict[str, Any] = {
        "task": task,
        "documents_expected": len(docs),
        "documents_scored": len(rows),
        "schema_valid": sum(bool(row["valid_schema"]) for row in rows),
        "valid_json": sum(bool(row["valid_json"]) for row in rows),
        "invalid_or_failed": sum(not bool(row["valid_schema"]) for row in rows),
        "mean_latency_seconds": sum(float(row.get("latency_seconds") or 0) for row in rows) / len(rows) if rows else 0,
        "macro_metrics": {},
        "reported_usage": _sum_numeric(rows, ("prompt_tokens", "completion_tokens", "total_tokens", "cost")),
        "cache_read_tokens": sum(float(row.get("usage", {}).get("cost_details", {}).get("cache_read") or
                                         row.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens") or 0)
                                 for row in rows),
    }
    if not rows:
        return result
    metric_keys = scores[0].keys()
    for key in metric_keys:
        if isinstance(scores[0][key], dict):
            result["macro_metrics"][key] = {
                subkey: sum(score[key][subkey] for score in scores) / len(scores)
                for subkey in scores[0][key]
            }
        elif isinstance(scores[0][key], (int, float)):
            result["macro_metrics"][key] = sum(score[key] for score in scores) / len(scores)
    if task in {"entities", "events", "quotes", "supersenses"}:
        matched = sum(score["matched"] for score in scores)
        predicted = sum(score["predicted"] for score in scores)
        gold = sum(score["gold"] for score in scores)
        precision = matched / predicted if predicted else (1.0 if not gold else 0.0)
        recall = matched / gold if gold else (1.0 if not predicted else 0.0)
        result["pooled_metrics"] = {
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "matched": matched, "predicted": predicted, "gold": gold,
        }
    if task == "coref":
        pred_by_id = {row["doc_id"]: row["prediction"] for row in rows}
        scorer_dir = Path(rows[0]["_scorer_dir"])
        official = benchmark.score_conll(docs, pred_by_id, scorer_dir)
        official.pop("raw_output", None)
        result["official_scorer_metrics"] = official
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_dir = Path(__file__).resolve().parent
    experiment_root = Path(args.experiment_root).resolve()
    benchmark = _load_experiment(experiment_root)
    task_docs = _task_docs(benchmark)
    split = "validation" if args.smoke else "test"
    selections = _verify_prompts_and_splits(benchmark, repo_dir, task_docs, split)
    model_metadata = fetch_model_metadata()
    # Credential lookup is intentionally quiet; no key value enters files, logs, or output.
    api_key = load_api_key()
    output_root = Path(args.output_dir).resolve() if args.output_dir else experiment_root / "results" / "openrouter_gpt_6_luna_pro"
    run_root = output_root / ("smoke" if args.smoke else "test")
    cache_root = output_root / "cache"
    run_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    atomic_json(run_root / "openrouter_metadata.json", model_metadata)
    config = {
        "benchmark_version": BENCHMARK_VERSION,
        "model": MODEL,
        "endpoint": API_URL,
        "provider_pinned": PROVIDER,
        "split": split,
        "tasks": ["entities"] if args.smoke else list(TASK_ORDER),
        "expected_documents": {"entities": 1} if args.smoke else TASK_SIZES,
        "reasoning": {"effort": REASONING_EFFORT, "include_reasoning": False},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "legacy_context_setting": CONTEXT_TOKENS,
        "provider_context_length": model_metadata["default_provider_endpoint"].get("context_length"),
        "temperature": None,
        "temperature_note": "The OpenAI endpoint does not advertise temperature; omitted, leaving the provider default.",
        "context_limit_note": "OpenRouter does not expose an Ollama-style num_ctx setting; the default OpenAI endpoint advertises its provider context length. The complete unchunked prompts are sent as in the baseline.",
        "structured_output": "response_format=json_schema, strict=true; routed only to provider endpoint supporting required parameters",
        "prompt_serialization": "exact frozen local prompt + newline + blank line + PASSAGE TOKENS header + every [zero_based_id] token, matching benchmark.prompt_for",
        "split_protocols": {task: selections[task][1] for task in ("entities",) if args.smoke} if args.smoke else {task: selections[task][1] for task in TASK_ORDER},
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "budget_limit_usd": BUDGET_USD,
    }
    atomic_json(run_root / "run_config.json", config)
    attempt_log = run_root / "attempts.jsonl"
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    actual_cost = 0.0
    requested_tasks = ("entities",) if args.smoke else TASK_ORDER
    for task in requested_tasks:
        docs, _, prompt_hash = selections[task]
        if args.smoke:
            docs = docs[:1]
        task_file = run_root / f"{task}.json"
        existing: dict[str, dict[str, Any]] = {}
        if task_file.exists():
            try:
                for row in json.loads(task_file.read_text(encoding="utf-8")):
                    existing[row["doc_id"]] = row
            except (json.JSONDecodeError, KeyError, TypeError):
                raise RuntimeError(f"Existing {task} results file is malformed: {task_file}") from None
        rows: list[dict[str, Any]] = []
        for doc in docs:
            prompt = benchmark.prompt_for(task, doc)
            phash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            template = (experiment_root / "prompts" / f"{task}.md").read_text(encoding="utf-8")
            template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
            if template_hash != prompt_hash or not prompt_uses_exact_template(prompt, template):
                raise RuntimeError(f"{task} prompt changed after startup verification")
            request_hash = cache_identity(task, doc["doc_id"], prompt, benchmark.SCHEMAS[task], split)
            cache_file = cache_root / f"{request_hash}.json"
            cached = _read_cached_row(cache_file)
            if cached is None:
                payload = request_payload(task, prompt, benchmark.SCHEMAS[task])
                try:
                    response, attempts = call_openrouter(api_key, payload)
                except TransportFailure as error:
                    for item in getattr(error, "attempts", []):
                        append_jsonl(attempt_log, {"task": task, "doc_id": doc["doc_id"],
                                                   "request_hash": request_hash, **item})
                    raise
                latency = sum(item.get("latency_seconds", 0) for item in attempts)
                usage = _usage(response)
                choices = []
                for choice in response.get("choices", []):
                    message = choice.get("message") or {}
                    safe_message = {key: message[key] for key in ("content", "refusal") if key in message}
                    choices.append({"finish_reason": choice.get("finish_reason"), "message": safe_message})
                cached = {
                    "benchmark_version": BENCHMARK_VERSION, "model": MODEL,
                    "task": task, "split": split, "doc_id": doc["doc_id"],
                    "prompt_version": prompt.splitlines()[0].lstrip("# ").strip(),
                    "prompt_hash": phash, "request_hash": request_hash,
                    "latency_seconds": latency, "attempts": attempts,
                    "usage": usage, "response": {
                        **{key: response.get(key) for key in ("id", "model", "provider", "created", "service_tier")
                           if key in response},
                        "choices": choices,
                        "usage": usage,
                    },
                }
                atomic_json(cache_file, cached)
                for item in attempts:
                    append_jsonl(attempt_log, {"task": task, "doc_id": doc["doc_id"],
                                               "request_hash": request_hash, **item})
            else:
                if cached.get("request_hash") != request_hash or cached.get("model") != MODEL:
                    raise RuntimeError(f"Cache identity mismatch for {task}/{doc['doc_id']}")
            row = _evaluate_response(benchmark, task, doc, cached)
            actual_cost += float(row.get("usage", {}).get("cost") or 0)
            if actual_cost > BUDGET_USD:
                raise RuntimeError(f"Reported OpenRouter cost exceeded ${BUDGET_USD:.2f}; inference halted")
            row["_scorer_dir"] = str(run_root / "coref_official") if task == "coref" else None
            existing[doc["doc_id"]] = row
            ordered_rows = [existing[doc_item["doc_id"]] for doc_item in docs if doc_item["doc_id"] in existing]
            atomic_json(run_root / f"{task}.json", ordered_rows)
            rows.append(row)
            print(json.dumps({"task": task, "doc_id": doc["doc_id"],
                              "provider": row.get("provider"), "resolved_model": row.get("resolved_model"),
                              "valid_schema": row.get("valid_schema"),
                              "prompt_tokens": row.get("usage", {}).get("prompt_tokens"),
                              "completion_tokens": row.get("usage", {}).get("completion_tokens"),
                              "reported_cost_usd": row.get("usage", {}).get("cost"),
                              "cumulative_reported_cost_usd": round(actual_cost, 6)},
                             separators=(",", ":")), flush=True)
        rows_by_task[task] = rows

        # Existing scorer reads private gold/token inputs but never sends them to the model.
        summary_rows = [{**row, "_scorer_dir": str(run_root / "coref_official")} for row in rows]
        summary = _task_summary(benchmark, task, docs, summary_rows)
        atomic_json(run_root / f"{task}_summary.json", summary)

    summaries = {task: json.loads((run_root / f"{task}_summary.json").read_text(encoding="utf-8"))
                 for task in requested_tasks}
    all_rows = [row for task_rows in rows_by_task.values() for row in task_rows]
    full_summary = {
        **config,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "requests_attempted": sum(1 for line in attempt_log.read_text(encoding="utf-8").splitlines() if line.strip()) if attempt_log.exists() else 0,
        "requests_with_responses": len(all_rows),
        "successful_schema_valid_responses": sum(bool(row["valid_schema"]) for row in all_rows),
        "schema_invalid_or_failed_responses": sum(not bool(row["valid_schema"]) for row in all_rows),
        "prompt_tokens": sum(float(row.get("usage", {}).get("prompt_tokens") or 0) for row in all_rows),
        "completion_tokens": sum(float(row.get("usage", {}).get("completion_tokens") or 0) for row in all_rows),
        "cached_input_tokens": sum(float(row.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens") or 0)
                                    for row in all_rows),
        "total_tokens": sum(float(row.get("usage", {}).get("total_tokens") or 0) for row in all_rows),
        "reported_cost_usd": sum(float(row.get("usage", {}).get("cost") or 0) for row in all_rows),
        "average_reported_cost_per_request_usd": sum(float(row.get("usage", {}).get("cost") or 0) for row in all_rows) / len(all_rows) if all_rows else 0,
        "latency_seconds": sum(float(row.get("latency_seconds") or 0) for row in all_rows),
        "task_summaries": summaries,
        "provider_counts": dict(Counter(row.get("provider") or "not reported" for row in all_rows)),
        "resolved_model_counts": dict(Counter(row.get("resolved_model") or "not reported" for row in all_rows)),
    }
    atomic_json(run_root / "summary.json", full_summary)
    return full_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True,
                        help="private local BookNLP LLM experiment checkout (contains src/, prompts/, datasets/)")
    parser.add_argument("--output-dir", help="local output/cache directory; defaults under experiment-root/results")
    parser.add_argument("--smoke", action="store_true", help="one validation entities document only")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: result[key] for key in (
        "model", "split", "requests_attempted", "requests_with_responses",
        "successful_schema_valid_responses", "schema_invalid_or_failed_responses",
        "prompt_tokens", "completion_tokens", "total_tokens", "reported_cost_usd",
        "provider_counts", "resolved_model_counts")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
