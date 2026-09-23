"""Model-free regression checks for token-ID and output-schema safeguards."""
from pathlib import Path
import importlib.util
import os
import sys

EXP = Path(os.environ.get("BOOKNLP_EXPERIMENT_ROOT", "/home/drew/booknlp_llm_experiment"))
sys.path.insert(0, str(EXP / "src"))
SCRIPT = Path(__file__).parents[1] / "scripts" / "exhaustive.py"
spec = importlib.util.spec_from_file_location("exhaustive_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

def test_batches_visit_each_token_once_in_order():
    flat = [x for batch in runner.batches(103) for x in batch]
    assert flat == list(range(103))

def test_model_output_labels_map_back_to_stable_token_ids():
    ids = [0, 7, 101]
    out = runner.validate_labels(ids, [0, 1, 0], "events")
    assert [(x["token_id"], x["label"]) for x in out] == [(0,"NO"),(7,"EVENT"),(101,"NO")]

def test_missing_or_bad_labels_raise_instead_of_silently_dropping():
    import pytest
    with pytest.raises(ValueError): runner.validate_labels([4,5], [1], "events")
    with pytest.raises(ValueError): runner.validate_labels([4], [9], "events")
    with pytest.raises(ValueError): runner.validate_labels([4], [True], "events")

def test_quote_boundary_pairing_keeps_local_token_indices():
    doc={"tokens":["“","Hello",",","Mary","!","”","she","said","." ]}
    labels=["OPEN","NO","NO","NO","NO","CLOSE","NO","NO","NO"]
    decisions={i:{"label":x} for i,x in enumerate(labels)}
    spans=runner.quote_spans(doc,decisions,{i:1.0 for i in range(len(labels))})
    assert [(x["start_token"],x["end_token"]) for x in spans] == [(0,5)]

def test_existing_event_metric_is_document_local_before_pooling():
    from booknlp_experiment.metrics import token_prf, pool_token_prf
    # Both books restart token IDs at zero; pooled token sets would collide.
    a=token_prf([{"start_token":1,"end_token":1}], [{"start_token":1,"end_token":1}])
    b=token_prf([{"start_token":1,"end_token":1}], [])
    pooled=pool_token_prf([a,b])
    assert pooled["gold"] == 2 and pooled["matched"] == 1
    assert pooled["recall"] == 0.5

def test_teacher_curve_keeps_restarted_token_offsets_document_scoped():
    rows=[
        ({"doc_id":"a","gold":{"events":[{"start_token":1,"end_token":1}]}},
         {"teacher_candidates":[{"start_token":1,"end_token":1,"confidence":1.0}]}),
        ({"doc_id":"b","gold":{"events":[{"start_token":2,"end_token":2}]}},
         {"teacher_candidates":[{"start_token":1,"end_token":1,"confidence":1.0}]})]
    curve=runner.teacher_curve("events",rows)
    full=curve["points"][-1]
    assert full["precision"] == 0.5
    assert full["gold_recall"] == 0.5
