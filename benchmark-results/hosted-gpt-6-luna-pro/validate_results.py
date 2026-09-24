"""Independently validate a completed GPT-6 Luna Pro test run against the baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import runner


def _equal_numbers(left, right, path="metric"):
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise AssertionError(f"{path}: key mismatch")
        for key in left:
            _equal_numbers(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"{path}: {left!r} != {right!r}")
    elif left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")


def _validate_split_overlap(task: str, validation_ids: set[str], test_ids: set[str]) -> int:
    """Enforce the frozen split's real semantics, including coref CV folds."""
    overlap = validation_ids & test_ids
    if task == "coref":
        # The benchmark defines coref test as the union of all ten held-out
        # folds, while its smoke/dev selection is fold 0's dev set. Those 10
        # dev documents are therefore necessarily in the pooled test union.
        if overlap != validation_ids or len(validation_ids) != 10:
            raise AssertionError("coref: unexpected validation/test overlap for the frozen CV protocol")
    elif overlap:
        raise AssertionError(f"{task}: validation/test document leakage")
    return len(overlap)


def validate(experiment_root: Path, results_dir: Path) -> dict:
    benchmark = runner._load_experiment(experiment_root)
    task_docs = runner._task_docs(benchmark)
    selections = runner._verify_prompts_and_splits(benchmark, Path(__file__).resolve().parent,
                                                   task_docs, "test")
    checked = {}
    failures = []
    expected_request_hashes = set()
    for task in runner.TASK_ORDER:
        docs, _, template_hash = selections[task]
        path = results_dir / f"{task}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        expected_ids = [doc["doc_id"] for doc in docs]
        actual_ids = [row.get("doc_id") for row in rows]
        if actual_ids != expected_ids:
            raise AssertionError(f"{task}: result IDs, order, or coverage differ from frozen manifest")
        if len(set(actual_ids)) != len(actual_ids):
            raise AssertionError(f"{task}: duplicate document result")
        by_id = {doc["doc_id"]: doc for doc in docs}
        for row in rows:
            doc = by_id[row["doc_id"]]
            prompt = benchmark.prompt_for(task, doc)
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if hashlib.sha256((experiment_root / "prompts" / f"{task}.md").read_bytes()).hexdigest() != template_hash:
                raise AssertionError(f"{task}: source prompt template changed")
            if row.get("prompt_hash") != prompt_hash:
                raise AssertionError(f"{task}/{doc['doc_id']}: prompt hash mismatch")
            expected_request_hash = runner.cache_identity(task, doc["doc_id"], prompt,
                                                          benchmark.SCHEMAS[task], "test")
            expected_request_hashes.add(expected_request_hash)
            if row.get("request_hash") != expected_request_hash:
                raise AssertionError(f"{task}/{doc['doc_id']}: request cache identity mismatch")
            if row.get("model") != runner.MODEL or row.get("resolved_model") != runner.MODEL:
                raise AssertionError(f"{task}/{doc['doc_id']}: model slug mismatch")
            if row.get("provider") != runner.PROVIDER:
                raise AssertionError(f"{task}/{doc['doc_id']}: provider mismatch")
            cache_path = results_dir.parent / "cache" / f"{expected_request_hash}.json"
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            replayed = runner._evaluate_response(benchmark, task, doc, cached)
            for key in ("valid_json", "valid_schema", "prediction", "metrics", "finish_reason", "usage"):
                if row.get(key) != replayed.get(key):
                    _equal_numbers(row.get(key), replayed.get(key), f"{task}/{doc['doc_id']}.{key}")
            if row.get("valid_schema") and not row.get("valid_json"):
                raise AssertionError(f"{task}/{doc['doc_id']}: schema valid while JSON invalid")
            # Validate every retained predicted span with the same bounds cleaner.
            field = {"entities": "entities", "events": "events", "coref": "mentions",
                     "quotes": "quotes", "speakers": "speakers", "supersenses": "supersenses"}[task]
            if benchmark.clean_preds(task, {field: row["prediction"]}, len(doc["tokens"])) != row["prediction"]:
                raise AssertionError(f"{task}/{doc['doc_id']}: prediction contains impossible token IDs")
            if row.get("split") != "test" or row.get("prompt_version") != prompt.splitlines()[0].lstrip("# ").strip():
                raise AssertionError(f"{task}/{doc['doc_id']}: split or prompt version mismatch")
        summary_rows = [{**row, "_scorer_dir": str(results_dir / "coref_official")}
                        for row in rows]
        recomputed = runner._task_summary(benchmark, task, docs, summary_rows)
        saved_summary = json.loads((results_dir / f"{task}_summary.json").read_text(encoding="utf-8"))
        for key in ("documents_expected", "documents_scored", "schema_valid", "valid_json", "invalid_or_failed"):
            if saved_summary.get(key) != recomputed.get(key):
                raise AssertionError(f"{task}: saved {key} differs from independent recomputation")
        for key in ("macro_metrics", "pooled_metrics", "official_scorer_metrics"):
            if key in recomputed:
                _equal_numbers(saved_summary.get(key), recomputed[key], f"{task}.{key}")
        checked[task] = {
            "documents": len(rows),
            "schema_valid": sum(bool(row["valid_schema"]) for row in rows),
            "valid_json": sum(bool(row["valid_json"]) for row in rows),
            "reported_cost_usd": sum(float(row.get("usage", {}).get("cost") or 0) for row in rows),
            "summary": recomputed,
        }

    total_rows = sum(item["documents"] for item in checked.values())
    if total_rows != sum(runner.TASK_SIZES.values()):
        raise AssertionError(f"expected {sum(runner.TASK_SIZES.values())} rows, found {total_rows}")
    total_cost = sum(item["reported_cost_usd"] for item in checked.values())
    if total_cost > runner.BUDGET_USD:
        raise AssertionError(f"reported spend ${total_cost:.4f} exceeds ${runner.BUDGET_USD:.2f}")

    attempt_path = results_dir / "attempts.jsonl"
    attempts = [json.loads(line) for line in attempt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    attempt_hashes = [item.get("request_hash") for item in attempts]
    if len(attempt_hashes) != total_rows or len(set(attempt_hashes)) != total_rows:
        raise AssertionError("attempt ledger does not contain exactly one unique HTTP attempt per paid request")
    if set(attempt_hashes) != expected_request_hashes:
        raise AssertionError("attempt ledger request identities do not exactly match the frozen test rows")
    if any(item.get("http_status") != 200 or item.get("status") != "response_received" for item in attempts):
        raise AssertionError("attempt ledger contains a retry or transport failure requiring separate accounting")

    run_summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    expected_valid = sum(item["schema_valid"] for item in checked.values())
    expected_tokens = {key: sum(float(row.get("usage", {}).get(source_key) or 0)
                                for task in runner.TASK_ORDER
                                for row in json.loads((results_dir / f"{task}.json").read_text(encoding="utf-8"))
                                for source_key in (key,))
                       for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    expected_cached_tokens = sum(float(row.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens") or 0)
                                 for task in runner.TASK_ORDER
                                 for row in json.loads((results_dir / f"{task}.json").read_text(encoding="utf-8")))
    if run_summary.get("requests_attempted") != len(attempts) or run_summary.get("requests_with_responses") != total_rows:
        raise AssertionError("run summary request counts do not match result rows and attempt ledger")
    if run_summary.get("successful_schema_valid_responses") != expected_valid:
        raise AssertionError("run summary schema-valid count does not match per-document replay")
    for key, value in expected_tokens.items():
        if not math.isclose(float(run_summary.get(key, -1)), value, abs_tol=1e-9):
            raise AssertionError(f"run summary {key} does not match per-document API usage")
    if not math.isclose(float(run_summary.get("cached_input_tokens", -1)), expected_cached_tokens, abs_tol=1e-9):
        raise AssertionError("run summary cached-token total does not match per-document API usage")
    if not math.isclose(float(run_summary.get("reported_cost_usd", -1)), total_cost, abs_tol=1e-10):
        raise AssertionError("run summary API cost does not match per-document API usage")
    if run_summary.get("provider_counts") != {runner.PROVIDER: total_rows} or run_summary.get("resolved_model_counts") != {runner.MODEL: total_rows}:
        raise AssertionError("run summary provider/model counts do not match all responses")

    # The source split definitions must keep validation documents out of each test cohort.
    overlap_counts = {}
    for task in runner.TASK_ORDER:
        val, _ = benchmark.select_task_split(task, task_docs[task], "validation")
        test, _ = benchmark.select_task_split(task, task_docs[task], "test")
        overlap_counts[task] = _validate_split_overlap(
            task, {d["doc_id"] for d in val}, {d["doc_id"] for d in test})

    audit = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": runner.MODEL,
        "provider": runner.PROVIDER,
        "total_documents": total_rows,
        "total_requests_with_responses": total_rows,
        "http_attempts": len(attempts),
        "duplicate_or_retried_requests": 0,
        "schema_valid": sum(item["schema_valid"] for item in checked.values()),
        "valid_json": sum(item["valid_json"] for item in checked.values()),
        "invalid_or_failed": total_rows - sum(item["schema_valid"] for item in checked.values()),
        "reported_cost_usd": total_cost,
        "validation_test_overlap_counts": overlap_counts,
        "tasks": checked,
        "checks": [
            "all six frozen test cohorts have exact expected IDs in manifest order",
            "each result prompt hash and cache key recompute from exact frozen prompt/token serialization",
            "all rows resolve to requested GPT-6 Luna Pro slug and pinned DeepInfra provider",
            "attempt ledger has one successful HTTP request per distinct cached task/document identity",
            "attempt hashes exactly cover the frozen rows and summary token/cost totals match per-response API usage",
            "each cached API response independently reparses to the saved prediction and task score",
            "pooled and official CoNLL metrics independently recomputed",
            "all predictions have valid token offsets after the original cleaner",
            "validation/test overlap follows each frozen split definition (coref fold-0 dev is part of the tenfold test union)",
            "per-task totals aggregate documents independently, avoiding cross-document token-ID collisions",
        ],
    }
    runner.atomic_json(results_dir / "validation_audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--results-dir", required=True,
                        help="local test result folder, not the public repository export")
    args = parser.parse_args()
    audit = validate(Path(args.experiment_root).resolve(), Path(args.results_dir).resolve())
    print(json.dumps({key: audit[key] for key in (
        "model", "provider", "total_documents", "schema_valid", "valid_json",
        "invalid_or_failed", "reported_cost_usd")}, indent=2))


if __name__ == "__main__":
    main()
