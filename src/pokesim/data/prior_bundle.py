"""Build an offline-only prior snapshot from train records. Never overwrite live priors or the split manifest."""
import argparse
import json
from collections import Counter
from dataclasses import fields
from pathlib import Path

from pokesim.config import PROCESSED_DIR
from pokesim.data.dataset import DatasetScope, add_scope_args, scope_from, validate_artifact, write_metadata
from pokesim.data.summarize_sets import SpeciesSummary

DEFAULT = PROCESSED_DIR / "scoped/priors.json"


def summaries(data: dict) -> dict[str, SpeciesSummary]:
    return {sid: SpeciesSummary(**{k: Counter(v) if isinstance(v, dict) else v for k, v in sm.items()})
            for sid, sm in data.items()}


def sampler(bundle: dict, seed: int = 0):
    from pokesim.search.lookahead import SetSampler

    return SetSampler(["M-6"], seed=seed, observed={}, summary=summaries(bundle["sets"]))


def load_bundle(path: Path = DEFAULT, scope: DatasetScope | None = None) -> dict:
    meta = validate_artifact(path, scope)
    if meta.get("split") != "train":
        raise ValueError("priors must be fitted on train only")
    if scope and meta["records_sha"] != scope.metadata("train")["records_sha"]:
        raise ValueError("train records changed: rebuild the prior bundle")
    return json.loads(Path(path).read_text())


def build(scope: DatasetScope, out: Path = DEFAULT, allow_empty: bool = False) -> dict:
    """allow_empty: build from the published sets alone when the ledger has no train battles yet (a fresh
    collection); the bundle is still versioned against the (empty) train split."""
    from pokesim.live.observed import collect
    from pokesim.rl.items import build_priors as item_priors
    from pokesim.rl.myteam import COUNTER_ABILITIES, COUNTER_GROUPS
    from pokesim.rl.threats import build_priors as threat_priors
    from pokesim.search.lookahead import SetSampler

    records = scope.records("train")
    if not records and not allow_empty:
        raise ValueError("no eligible train records")
    engine = SetSampler(["M-6"], observed=collect([r for _, r in records]))
    items, _ = item_priors(engine)
    threats = threat_priors(engine)
    counters = {}
    for sid, sm in engine.summary.items():
        total = max(sm.teams, 1)
        counters[sid] = {g: min(1.0, sum(sm.moves.get(m, 0) for m in moves) / total)
                         for g, moves in COUNTER_GROUPS.items()}
        ab = sum(sm.abilities.values()) or 1
        counters[sid]["unaware"] = sum(v for k, v in sm.abilities.items() if k in COUNTER_ABILITIES) / ab
    bundle = {"sets": {sid: {f.name: dict(value) if isinstance(value := getattr(sm, f.name), Counter) else value
                            for f in fields(sm)} for sid, sm in engine.summary.items()},
              "items": items, "threats": threats, "counter": counters}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, sort_keys=True))
    write_metadata(out, scope.metadata("train"))
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    add_scope_args(parser)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--allow-empty", action="store_true", help="no train battles yet: published sets only")
    args = parser.parse_args()
    build(scope_from(args), args.out, args.allow_empty)
    print(f"train-only priors -> {args.out}")


if __name__ == "__main__":
    main()
