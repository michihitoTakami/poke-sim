"""Score the advisor's opponent predictions on our own recorded battles (data/raw/live/logs).

  uv run python scripts/replay_frames.py /data/poke-sim/obs_frames/* --records data/processed/replayed
  uv run python scripts/eval_records.py query [--out data/processed/eval_cache.jsonl]   # run the search per turn
  uv run python scripts/eval_records.py score [--br-weight 0.5 --tau 0.1]                # score cached results
  uv run python scripts/eval_records.py bc                    # cache the cloned human policy's predictions
  uv run python scripts/eval_records.py score --bc-weight 1   # its share in the prediction (rest: the search)

`query` rebuilds each battle from its protocol log and, at every turn start, stores the search's payoff matrices
with what the opponent (and we) actually did that turn. `score` turns them into opponent predictions under an
opponent model and reports top-1 / top-3 hit rates and log-likelihood. Re-run `query` after changing the set
sampler (its samples are in the cache); opponent-model changes only need `score`.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from pokesim.config import ROOT
from pokesim.live.advisor import Advisor
from pokesim.live.observed import collect
from pokesim.live.session import BattleSession
from pokesim.rl.myteam import to_id
from pokesim.search.lookahead import SetSampler, aggregate

CACHE = ROOT / "data" / "processed" / "eval_cache.jsonl"
BC_CACHE = ROOT / "data" / "processed" / "eval_bc.jsonl"


def records(logs: Path) -> list[tuple[str, dict]]:
    out = []
    for p in sorted(logs.glob("*.json")):
        rec = json.loads(p.read_text())
        if rec.get("preview") and rec.get("lineup") and rec.get("log"):
            out.append((p.stem, rec))
    return out


def action(line: str) -> str:
    parts = line.split("|")
    if parts[1] == "move":
        return "move:" + to_id(parts[3])
    return "switch:" + to_id(parts[3].split(",")[0])


def turn_actions(log: list[str], start: int) -> tuple[str | None, str | None]:
    """The first move/switch of each side after log[start] (a |turn| line), before the next turn."""
    mine = theirs = None
    for line in log[start + 1:]:
        if line.startswith("|turn|"):
            break
        if line.startswith(("|move|", "|switch|")) and "[from]" not in line:
            side = line.split("|")[2][:2]
            if side == "p1" and mine is None:
                mine = action(line)
            if side == "p2" and theirs is None:
                theirs = action(line)
    return mine, theirs


def turns(args, recs, on_turn) -> int:
    """Rebuild each battle; at every turn start call on_turn(stem, rec, session, battle, mine, theirs)."""
    sets = json.loads(Path(args.team).read_text())
    n = 0
    for stem, rec in recs:
        session = BattleSession(sets)
        session.preview(rec["preview"]["species"])
        session.select([next(i for i, s in enumerate(sets) if to_id(s["species"]) == to_id(x)) for x in rec["lineup"]])
        log = rec["log"]
        start = next(i for i, line in enumerate(log) if line.startswith("|teampreview")) + 1
        for i in range(start, len(log)):
            session.apply([log[i]])
            if not log[i].startswith("|turn|"):
                continue
            mine, theirs = turn_actions(log, i)
            battle = session.decision()
            if theirs is None or battle.active_pokemon is None or battle.opponent_active_pokemon is None:
                continue
            on_turn(stem, rec, session, battle, mine, theirs)
            n += 1
            print(stem, battle.turn, theirs, flush=True)
    return n


def visible(other: dict, other_stem: str, rec: dict, stem: str, scope: str) -> bool:
    """Whether another battle's revealed sets may inform the predictions in this one.
    before: only battles that started earlier (what was known at the time; the default)
    others: every other battle, later ones included (the old setting: it can see the future)"""
    if other_stem == stem:
        return False
    if scope == "others":
        return True
    return (other.get("started") or other_stem) < (rec.get("started") or stem)


def query(args) -> None:
    sets = json.loads(Path(args.team).read_text())
    advisor = Advisor(sets, policy=None, opponent_policy=None)
    out = open(args.out, "w")
    recs = records(Path(args.logs))
    current = {}

    def on_turn(stem, rec, session, battle, mine, theirs):
        if current.get("stem") != stem:  # sets seen in other battles: by default only those played before this one
            others = [r for s2, r in recs if visible(r, s2, rec, stem, args.observed_scope)] if args.observed else []
            advisor.engine.sampler = SetSampler(["M-6"], observed=collect(others), utility=args.utility)
            current["stem"] = stem
        results = advisor.engine.query(battle, advisor.own_sets)
        out.write(json.dumps({"battle": stem, "turn": battle.turn, "opponent": theirs, "mine": mine,
                              "opp_active": battle.opponent_active_pokemon.species, "results": results}) + "\n")

    n = turns(args, recs, on_turn)
    advisor.close()
    print(f"{n} decisions -> {args.out}")


def bc(args) -> None:
    from pokesim.live.opponent import OpponentPolicy

    predictor = OpponentPolicy(args.bc_policy)
    out = open(args.bc_out, "w")

    def on_turn(stem, rec, session, battle, mine, theirs):
        out.write(json.dumps({"battle": stem, "turn": battle.turn, "opponent": theirs,
                              "dist": dict(predictor.predict(session.log))}) + "\n")

    n = turns(args, records(Path(args.logs)), on_turn)
    print(f"{n} decisions -> {args.bc_out}")


def score(args) -> dict:
    model = {"br_weight": args.br_weight, "tau": args.tau, "repeat": args.repeat, "bc_weight": args.bc_weight}
    hit1 = hit3 = 0
    ll = []
    rows = [json.loads(line) for line in open(args.out)]
    bc_dist = {}
    if args.bc_weight > 0:
        bc_dist = {(r["battle"], r["turn"]): r["dist"] for r in map(json.loads, open(args.bc_out))}
        rows = [r for r in rows if (r["battle"], r["turn"]) in bc_dist]
    prev: dict[tuple, str] = {}  # (battle, opponent active) -> its previous move
    for r in rows:
        key = (r["battle"], r["opp_active"])
        opp = aggregate(r["results"], model, prev.get(key)).opponent
        if args.bc_weight > 0:
            merged = {":".join(k.split(":")[:2]): 0.0 for k in opp}
            for k, p in opp.items():
                merged[":".join(k.split(":")[:2])] += (1 - args.bc_weight) * p
            for k, p in bc_dist[(r["battle"], r["turn"])].items():
                merged[k] = merged.get(k, 0.0) + args.bc_weight * p
            opp = merged
        if r["opponent"].startswith("move:"):
            prev[key] = r["opponent"]
        else:
            prev.pop(key, None)
        dist = {k.split(":")[0] + ":" + k.split(":")[1]: 0.0 for k in opp}  # merge "move:x:mega" into "move:x"
        for k, p in opp.items():
            key = ":".join(k.split(":")[:2])
            dist[key] = dist.get(key, 0.0) + p
        ranked = sorted(dist, key=lambda k: -dist[k])
        hit1 += ranked[:1] == [r["opponent"]]
        hit3 += r["opponent"] in ranked[:3]
        ll.append(math.log(max(dist.get(r["opponent"], 0.0), 1e-3)))
        if args.verbose:
            print(r["battle"], r["turn"], r["opponent"], [(k, round(dist[k], 2)) for k in ranked[:3]])
    n = len(rows)
    res = {"n": n, "top1": hit1 / n, "top3": hit3 / n, "loglik": float(np.mean(ll))}
    print(f"{model}  n={n}  top1 {hit1}/{n} ({hit1 / n:.0%})  top3 {hit3}/{n} ({hit3 / n:.0%})  "
          f"mean log-lik {res['loglik']:.2f}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["query", "score", "bc"])
    ap.add_argument("--team", default=str(ROOT / "data/my_teams/baton.json"))
    ap.add_argument("--out", default=str(CACHE))
    ap.add_argument("--logs", default=str(ROOT / "data/processed/replayed"),
                    help="battle records (default: rebuilt from saved frames by replay_frames.py --records)")
    ap.add_argument("--br-weight", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--repeat", type=float, default=0.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--bc-weight", type=float, default=0.0, help="share of the cloned human policy (needs bc)")
    ap.add_argument("--bc-policy", default=str(ROOT / "checkpoints/bc_human.zip"))
    ap.add_argument("--bc-out", default=str(BC_CACHE))
    ap.add_argument("--observed", action=argparse.BooleanOptionalAction, default=True,
                    help="blend in the sets seen in our other battles")
    ap.add_argument("--utility", type=float, default=0.03, help="pseudo-share for learnable utility moves")
    ap.add_argument("--observed-scope", choices=["before", "others"], default="before",
                    help="before: sets from battles played earlier only; others: all other battles (sees the future)")
    args = ap.parse_args()
    {"query": query, "score": score, "bc": bc}[args.cmd](args)


if __name__ == "__main__":
    main()
