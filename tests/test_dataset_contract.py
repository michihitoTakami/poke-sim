import json
from dataclasses import asdict

import pytest

from pokesim.data.curate_battles import Battle, sha
from pokesim.data.dataset import DatasetScope, require_training_rows, validate_artifact, write_metadata
from pokesim.data.prior_bundle import summaries


def scope_at(tmp_path):
    battles = []
    for i, split in enumerate(["train", "development", "test", "train", "train"]):
        path = tmp_path / f"{i}.json"
        path.write_text(json.dumps({"value": i}))
        b = Battle(f"showdown:battle-{i}", "showdown", f"20261015-00000{i}", "unknown",
                   [{"path": path.name, "sha": sha(path), "key": str(i), "kind": "showdown"}],
                   None, None, [])
        if i == 3:
            b.merge_pending = ["showdown:battle-0"]
        if i == 4:
            b.opponent_kind = "bot_confirmed"
        battles.append((b, split))
    ledger, manifest = tmp_path / "ledger.jsonl", tmp_path / "split.json"
    ledger.write_text("\n".join(json.dumps(asdict(b)) for b, _ in battles))
    manifest.write_text(json.dumps({"test_from": "20261015-000000",
                                    "battles": {b.battle_id: s for b, s in battles}}))
    return DatasetScope(ledger, manifest, tmp_path)


def test_primary_records_quarantine_and_role_indices(tmp_path):
    scope = scope_at(tmp_path)
    assert scope.records("train") == [("showdown:battle-0", {"value": 0})]
    assert scope.indices(["showdown:battle-0:p2", "showdown:battle-1:p2"], "development").tolist() == [1]
    scope.battles[0].records.append({"path": "missing-secondary.json"})
    assert len(scope.records("train")) == 1


def test_raw_changes_fail_closed(tmp_path):
    scope = scope_at(tmp_path)
    (tmp_path / "0.json").write_text('{}')
    with pytest.raises(ValueError, match="record changed"):
        scope.records("train")


def test_artifacts_require_provenance_and_reject_changes(tmp_path):
    scope = scope_at(tmp_path)
    artifact = tmp_path / "data.npz"
    artifact.write_bytes(b"data")
    with pytest.raises(ValueError, match="unversioned"):
        validate_artifact(artifact, scope)
    write_metadata(artifact, scope.metadata("train"))
    assert validate_artifact(artifact, scope)["split"] == "train"
    scope.battles[0].records[0]["sha"] = "changed"
    with pytest.raises(ValueError, match="train records changed"):
        validate_artifact(artifact, scope)
    scope = scope_at(tmp_path)
    artifact.write_bytes(b"modified")
    with pytest.raises(ValueError, match="modified"):
        validate_artifact(artifact, scope)


@pytest.mark.parametrize("train,dev", [([], [1]), ([1], [])])
def test_empty_real_development_never_falls_back_to_random_split(train, dev):
    with pytest.raises(ValueError, match="nonempty"):
        require_training_rows(train, dev)


def test_prior_summary_round_trip_preserves_counter_keys():
    sm = summaries({"garchomp": {"species_id": "garchomp", "teams": 2, "moves": {"earthquake": 2}}})
    assert sm["garchomp"].moves.most_common() == [("earthquake", 2)]
