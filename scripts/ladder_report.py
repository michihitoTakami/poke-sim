"""Summarise ladder records (scripts/ladder.py) to find problems with the models.

  uv run python scripts/ladder_report.py [--since 20261014-2153] [--losses]

Per policy (A/B runs alternate battles): wins / losses with a 95% interval, decisions that fell back to a random
move, advice time. Over all battles: how often the opponent prediction had the opponent's actual action first /
in its top 3, how the policy's win estimate at the first turn relates to the result, our lineups, the opponents'
revealed items that the item priors (pokesim.rl.items) gave little chance, and (--losses) one line per loss.
Records have us as p1.
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from pokesim.config import RAW_DIR
from pokesim.rl.items import item_class, load_priors
from pokesim.rl.myteam import to_id

LADDER = RAW_DIR / "ladder"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n)


def label(line: str) -> str:
    parts = line.split("|")
    if parts[1] == "move":
        return "move:" + to_id(parts[3])
    return "switch:" + to_id(parts[3].split(",")[0])


def opponent_action(log: list[str], turn: int) -> str | None:
    """The opponent's (p2) first move or switch in `turn` (turn 0: the lead)."""
    start = 0 if turn == 0 else next((i for i, ln in enumerate(log) if ln == f"|turn|{turn}"), None)
    if start is None:
        return None
    for line in log[start + 1:]:
        if line.startswith("|turn|"):
            return None
        if line.startswith(("|move|p2", "|switch|p2")) and "[from]" not in line:
            return label(line)
    return None


def match(pred: str, actual: str) -> bool:
    """Mega variants and formes count as the same action."""
    p, a = pred.split(":"), actual.split(":")
    return p[0] == a[0] and (p[1] == a[1] or p[1].startswith(a[1]) or a[1].startswith(p[1]))


def revealed_items(log: list[str]) -> dict[str, str]:
    """Opponent species -> item it was seen with (|-item|, |-enditem|, [from] item:)."""
    out = {}
    for line in log:
        parts = line.split("|")
        if len(parts) < 4 or not parts[2].startswith(("p2a:", "p2:")):
            continue
        mon = to_id(parts[2].split(":", 1)[1])
        if parts[1] in ("-item", "-enditem"):
            out.setdefault(mon, to_id(parts[3]))
        elif m := re.search(r"\[from\] item: ([^|]+)", line):
            out.setdefault(mon, to_id(m.group(1)))
    return out


def species_of(log: list[str]) -> dict[str, str]:
    """Opponent nickname id -> species id."""
    out = {}
    for line in log:
        parts = line.split("|")
        if parts[1:2] in (["switch"], ["drag"]) and parts[2].startswith("p2a:"):
            out[to_id(parts[2].split(":", 1)[1])] = to_id(parts[3].split(",")[0])
    return out


def calibration(recs: list[dict]) -> None:
    """The terminal search's own objective for the played action (as a win chance) against the real result:
    bins, Brier against a constant, by turn bucket. A constant beating the search means its rollouts mislead it."""
    from pokesim.evaluation.positions import objective_of

    rows = []
    for rec in recs:
        rho = ((rec.get("versions") or {}).get("planner") or {}).get("rho", .3)
        for a in rec.get("advice", []):
            if a.get("force"):
                continue
            value = objective_of(a, rho)
            if value is not None:
                rows.append(((value + 1) / 2, rec["result"] == "win", a["turn"]))
    if not rows:
        return
    won = sum(w for _, w, _ in rows) / len(rows)
    brier = sum((p - w) ** 2 for p, w, _ in rows) / len(rows)
    constant = sum((won - w) ** 2 for _, w, _ in rows) / len(rows)
    print(f"search objective of the played action -> actual result ({len(rows)} decisions): "
          f"Brier {brier:.4f} vs constant {constant:.4f} ({'worse' if brier > constant else 'better'})")
    bins = defaultdict(lambda: [0, 0])
    for p, w, _ in rows:
        bins[min(int(p * 5), 4)][0] += w
        bins[min(int(p * 5), 4)][1] += 1
    for b in sorted(bins):
        k, n = bins[b]
        print(f"  {b * 20:3d}-{b * 20 + 20}%: {k}/{n} = {k / n:.0%}")
    for name, lo, hi in (("early (T<=3)", 0, 3), ("middle (T4-8)", 3, 8), ("late (T>8)", 8, 10 ** 6)):
        part = [(p, w) for p, w, t in rows if lo < t <= hi]
        if part:
            print(f"  {name:14s} predicted {sum(p for p, _ in part) / len(part):.2f} "
                  f"actual {sum(w for _, w in part) / len(part):.2f} (n={len(part)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(LADDER))
    ap.add_argument("--since", default="", help="only records started at or after this stamp (YYYYMMDD-HHMM)")
    ap.add_argument("--losses", action="store_true", help="one line per loss")
    args = ap.parse_args()

    recs = []
    for p in sorted(Path(args.dir).glob("*.json")):
        rec = json.loads(p.read_text())
        if rec.get("started", "") >= args.since and rec.get("result") in ("win", "loss"):
            recs.append(rec)
    print(f"{len(recs)} battles")

    by_policy = defaultdict(list)
    for rec in recs:
        by_policy[rec.get("policy") or "?"].append(rec)
    for pol, rs in sorted(by_policy.items()):
        won = sum(r["result"] == "win" for r in rs)
        lo, hi = wilson(won, len(rs))
        advice = [a for r in rs for a in r.get("advice", [])]
        random = sum(a.get("chosen") == "random" for a in advice)
        secs = [a["secs"] for a in advice if "secs" in a]
        print(f"  {pol:24s} {won}-{len(rs) - won} ({won / len(rs):.0%}, 95% {lo:.0%}-{hi:.0%})  "
              f"decisions {len(advice)}, random fallbacks {random}, advice "
              f"{(sum(secs) / len(secs) if secs else 0):.1f}s mean / {max(secs, default=0):.1f}s max")

    hits1 = hits3 = n = 0
    for rec in recs:
        for a in rec.get("advice", []):
            if a.get("force") or not a.get("opponent"):
                continue
            actual = opponent_action(rec["log"], a["turn"])
            if actual is None:
                continue
            preds = [lab for lab, _ in a["opponent"]]
            n += 1
            hits1 += match(preds[0], actual)
            hits3 += any(match(p, actual) for p in preds[:3])
    if n:
        print(f"opponent prediction: top-1 {hits1 / n:.0%}, top-3 {hits3 / n:.0%} over {n} turns")

    calibration(recs)

    buckets = defaultdict(lambda: [0, 0])
    for rec in recs:
        first = next((a for a in rec.get("advice", []) if a.get("win") is not None), None)
        if first:
            b = min(int(first["win"] * 5), 4)
            buckets[b][0] += rec["result"] == "win"
            buckets[b][1] += 1
    if buckets:
        print("win estimate at the first decision -> actual win rate:")
        for b in sorted(buckets):
            w, t = buckets[b]
            print(f"  {b * 20:3d}-{b * 20 + 20}%: {w}/{t}")

    lineups = defaultdict(lambda: [0, 0])
    for rec in recs:
        key = " / ".join(rec.get("lineup") or [])
        lineups[key][0] += rec["result"] == "win"
        lineups[key][1] += 1
    print("lineups:")
    for key, (w, t) in sorted(lineups.items(), key=lambda kv: -kv[1][1]):
        print(f"  {w}/{t}  {key}")

    priors = load_priors()
    surprises, seen = Counter(), Counter()
    for rec in recs:
        names = species_of(rec["log"])
        for mon, item in revealed_items(rec["log"]).items():
            sid = names.get(mon, mon)
            cls = item_class(item, set())
            seen[cls] += 1
            prior = priors.get(sid, {}).get(cls, 0.0)
            if prior < 0.1:
                surprises[(sid, item, round(prior, 2))] += 1
    if seen:
        print("opponents' revealed item classes:", dict(seen.most_common()))
        print("revealed items the priors gave < 10%:")
        for (sid, item, prior), k in surprises.most_common(15):
            print(f"  {sid} @ {item} (prior {prior:.0%}) x{k}")

    if args.losses:
        print("losses:")
        for rec in recs:
            if rec["result"] != "loss":
                continue
            log = rec["log"]
            brought = list(dict.fromkeys(to_id(ln.split("|")[3].split(",")[0]) for ln in log
                                         if ln.startswith(("|switch|p2", "|drag|p2"))))
            items = revealed_items(log)
            turns = sum(ln.startswith("|turn|") for ln in log)
            print(f"  [{rec.get('policy')}] {rec['started']} {turns}T us {'/'.join(rec.get('lineup') or [])} "
                  f"vs {'/'.join(brought)}  items {items}")


if __name__ == "__main__":
    main()
