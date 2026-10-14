"""Which battles each step may use: one split of the battle ledger (pokesim.data.curate_battles) for every process.

  train        imitation, policy / value training, set priors, positions to practise
  development  choosing depth, risk settings, league shares, candidates (tuned on: not a final result)
  calibration  calibrating win estimates and opponent predictions of a fixed candidate
  test         the final comparison, opened once after everything is fixed

Battles are grouped (same opponent on Showdown; possible duplicates in `merge_pending` together) and whole groups are
assigned in order of their first battle: the earliest groups to train, then development, then calibration. A group
never spans splits. When a group's later battles fall after the next split began, they are "purged" (left out):
keeping them would put a later battle in an earlier split.

Test is forward: battles that start at or after `test_from` (default: when the manifest was made) are test, so the
final comparison is on battles nobody has looked at. Battles not in the manifest are assigned by the same rule
(split_of).

  uv run python -m pokesim.data.split_manifest [--shares 0.6 0.2 0.2] [--test-from 20261016-000000]
"""
import argparse
import json
import time
from pathlib import Path

from pokesim.data.curate_battles import CURATED, Battle, load

MANIFEST = CURATED / "split_manifest.json"
SPLITS = ("train", "development", "calibration")


def groups(battles: list[Battle]) -> list[list[Battle]]:
    """Battles that must share a split: same opponent group, linked by merge_pending."""
    parent = {b.battle_id: b.battle_id for b in battles}

    def root(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, c: str) -> None:
        if a in parent and c in parent:
            parent[root(a)] = root(c)

    by_group: dict[str, str] = {}
    for b in battles:
        if b.opponent_group:
            if b.opponent_group in by_group:
                union(b.battle_id, by_group[b.opponent_group])
            else:
                by_group[b.opponent_group] = b.battle_id
        for other in b.merge_pending:
            union(b.battle_id, other)
    out: dict[str, list[Battle]] = {}
    for b in battles:
        out.setdefault(root(b.battle_id), []).append(b)
    return sorted((sorted(g, key=lambda b: b.started) for g in out.values()), key=lambda g: (g[0].started,
                                                                                            g[0].battle_id))


def make(battles: list[Battle], shares: tuple[float, float, float] = (0.6, 0.2, 0.2),
         test_from: str | None = None) -> dict:
    test_from = test_from or time.strftime("%Y%m%d-%H%M%S")
    before = [b for b in battles if b.started < test_from]
    gs = groups(before)
    n = sum(len(g) for g in gs)
    bounds = [sum(shares[:k + 1]) / sum(shares) * n for k in range(len(SPLITS))]
    assign: dict[str, str] = {}
    starts: dict[str, str] = {}  # first battle of each split
    done = 0
    for g in gs:
        k = next(i for i, bound in enumerate(bounds) if done < bound or i == len(SPLITS) - 1)
        split = SPLITS[k]
        starts.setdefault(split, g[0].started)
        for b in g:
            assign[b.battle_id] = split
        done += len(g)
    for b in before:  # a battle after the next split's first battle would leak forward in time
        k = SPLITS.index(assign[b.battle_id])
        nxt = next((starts[s] for s in SPLITS[k + 1:] if s in starts), None)
        if nxt is not None and b.started >= nxt:
            assign[b.battle_id] = "purged"
    for b in battles:
        if b.started >= test_from:
            assign[b.battle_id] = "test"
    counts = {s: sum(v == s for v in assign.values()) for s in (*SPLITS, "test", "purged")}
    return {"made": time.strftime("%Y-%m-%d %H:%M:%S"), "test_from": test_from, "shares": list(shares),
            "split_starts": starts, "counts": counts, "battles": assign}


class Splits:
    """Lookups on a saved manifest."""

    def __init__(self, path: Path = MANIFEST):
        self.data = json.loads(Path(path).read_text())
        self.battles: dict[str, str] = self.data["battles"]

    def split_of(self, battle: Battle) -> str:
        if battle.battle_id in self.battles:
            return self.battles[battle.battle_id]
        if battle.started >= self.data["test_from"]:
            return "test"
        return "unassigned"  # an older battle missing from the manifest: rebuild it rather than guess

    def ids(self, split: str) -> set[str]:
        return {k for k, v in self.battles.items() if v == split}

    def record_keys(self, ledger: list[Battle], split: str) -> set[str]:
        """Record keys (as in pokesim.rl.human: "replayed/<stem>", "ladder/<stem>") of the split's battles; every
        record of a battle (live and rebuilt) gets the battle's split."""
        return {r["key"] for b in ledger if self.split_of(b) == split for r in b.records}


def as_of(battles: list[Battle], started: str) -> list[Battle]:
    """The battles that had already started before `started` (what was known at that time)."""
    return [b for b in battles if b.started < started]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shares", type=float, nargs=3, default=[0.6, 0.2, 0.2], metavar=("TRAIN", "DEV", "CAL"))
    ap.add_argument("--test-from", help="YYYYMMDD-HHMMSS: battles from then on are test (default: now)")
    ap.add_argument("--out", default=str(MANIFEST))
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest (moves its test start)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists: the split is fixed once made (use --force to remake it)")
    manifest = make(load(), tuple(args.shares), args.test_from)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"-> {out}: {manifest['counts']} (test from {manifest['test_from']})")


if __name__ == "__main__":
    main()
