"""Frozen split contract for offline data builders. Live collectors do not read or modify this contract."""

import hashlib
import json
from pathlib import Path

from pokesim.config import ROOT
from pokesim.data.curate_battles import CURATED, load, sha
from pokesim.data.split_manifest import MANIFEST, Splits
from pokesim.runtime.bundle import file_hash, git_commit

SCHEMA = 2


def observation_code() -> dict:
    paths = [
        "rl/human.py",
        "rl/features.py",
        "rl/myteam.py",
        "rl/items.py",
        "rl/threats.py",
        "sim/damage.py",
        "sim/effectiveness.py",
        "live/session.py",
    ]
    return {p: file_hash(ROOT / "src/pokesim" / p) for p in paths}


class DatasetScope:
    def __init__(self, ledger: Path = CURATED / "battles.jsonl", manifest: Path = MANIFEST, root: Path = ROOT):
        self.ledger_path, self.manifest_path, self.root = Path(ledger), Path(manifest), Path(root)
        self.battles, self.splits = load(self.ledger_path), Splits(self.manifest_path)

    def select(self, split: str):
        if split not in {"train", "development", "calibration", "test"}:
            raise ValueError(f"unsupported split: {split}")
        # Ambiguous duplicates are quarantined, not silently merged or counted twice. Unknown players stay usable.
        return [
            b
            for b in self.battles
            if self.splits.split_of(b) == split and not b.merge_pending and b.opponent_kind != "bot_confirmed"
        ]

    def records(self, split: str) -> list[tuple[str, dict]]:
        out = []
        for b in self.select(split):
            primary = b.records[0]
            path = self.root / primary["path"]
            if sha(path) != primary["sha"]:
                raise ValueError(f"record changed since curation: {b.battle_id}; refresh the ledger explicitly")
            record = json.loads(path.read_text())
            out.append((b.battle_id, record))
        return out

    def metadata(self, split: str) -> dict:
        records = {b.battle_id: b.records[0]["sha"] for b in self.select(split)}
        return {
            "schema_version": SCHEMA,
            "code": git_commit(),
            "split": split,
            "split_manifest_sha": file_hash(self.manifest_path),
            "ledger_sha": file_hash(self.ledger_path),
            "records": records,
            "records_sha": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()[:16],
            "reader_sha": file_hash(ROOT / "src/pokesim/rl/human.py"),
            "features_sha": file_hash(ROOT / "src/pokesim/rl/features.py"),
            "observation_code": observation_code(),
        }

    def indices(self, ids, split: str):
        import numpy as np

        allowed = {b.battle_id for b in self.select(split)}
        return np.flatnonzero([str(key).rsplit(":", 1)[0] in allowed for key in ids])


def sidecar(path: Path) -> Path:
    return Path(str(path) + ".manifest.json")


def write_metadata(path: Path, metadata: dict) -> None:
    sidecar(path).write_text(json.dumps(dict(metadata, artifact_sha=file_hash(path)), indent=2))


def validate_artifact(path: Path, scope: DatasetScope | None = None) -> dict:
    if not sidecar(path).exists():
        raise ValueError(f"unversioned artifact: {path}; regenerate it with the current builder")
    meta = json.loads(sidecar(path).read_text())
    if meta.get("schema_version") != SCHEMA or meta.get("artifact_sha") != file_hash(path):
        raise ValueError(f"stale or modified artifact: {path}")
    if (
        meta.get("reader_sha") != file_hash(ROOT / "src/pokesim/rl/human.py")
        or meta.get("features_sha") != file_hash(ROOT / "src/pokesim/rl/features.py")
        or meta.get("observation_code") != observation_code()
    ):
        raise ValueError(f"observation code changed: regenerate {path}")
    if scope and meta.get("split_manifest_sha") != file_hash(scope.manifest_path):
        raise ValueError(f"artifact uses another split manifest: {path}")
    if scope and meta.get("records") != scope.metadata("train")["records"]:
        raise ValueError(f"train records changed: rebuild {path}")
    if (
        scope
        and "development_records" in meta
        and meta["development_records"] != scope.metadata("development")["records"]
    ):
        raise ValueError(f"development records changed: rebuild {path}")
    return meta


def require_training_rows(train, development) -> None:
    if not len(train) or not len(development):
        raise ValueError("need nonempty train AND real-battle development decisions; rebuild data before training")


def add_scope_args(parser) -> None:
    parser.add_argument("--ledger", type=Path, default=CURATED / "battles.jsonl")
    parser.add_argument("--split-manifest", type=Path, default=MANIFEST)


def scope_from(args) -> DatasetScope:
    return DatasetScope(args.ledger, args.split_manifest)
