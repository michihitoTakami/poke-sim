"""Score the advisor's opponent predictions on our own recorded battles (data/raw/live/logs).

  uv run python scripts/replay_frames.py /data/poke-sim/obs_frames/* --records data/processed/replayed
  uv run python scripts/eval_records.py query [--out data/processed/eval_cache.jsonl]   # run the search per turn
  uv run python scripts/eval_records.py score [--br-weight 0.5 --tau 0.1]                # score cached results

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


def query(args) -> None:
    sets = json.loads(Path(args.team).read_text())
    advisor = Advisor(sets, policy=None)
    out = open(args.out, "w")
    n = 0
    recs = records(Path(args.logs))
    for stem, rec in recs:
        # sets seen in the *other* battles only (leave-one-battle-out), as they would be known before this one
        others = [r for s2, r in recs if s2 != stem] if args.observed else []
        advisor.engine.sampler = SetSampler(["M-6"], observed=collect(others), utility=args.utility)
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
            results = advisor.engine.query(battle, advisor.own_sets)
            out.write(json.dumps({"battle": stem, "turn": battle.turn, "opponent": theirs, "mine": mine,
                                  "opp_active": battle.opponent_active_pokemon.species,
                                  "results": results}) + "\n")
            n += 1
            print(stem, battle.turn, theirs, flush=True)
    advisor.close()
    print(f"{n} decisions -> {args.out}")


def score(args) -> dict:
    model = {"br_weight": args.br_weight, "tau": args.tau, "repeat": args.repeat}
    hit1 = hit3 = 0
    ll = []
    rows = [json.loads(line) for line in open(args.out)]
    prev: dict[tuple, str] = {}  # (battle, opponent active) -> its previous move
    for r in rows:
        key = (r["battle"], r["opp_active"])
        opp = aggregate(r["results"], model, prev.get(key)).opponent
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
    ap.add_argument("cmd", choices=["query", "score"])
    ap.add_argument("--team", default=str(ROOT / "data/my_teams/baton.json"))
    ap.add_argument("--out", default=str(CACHE))
    ap.add_argument("--logs", default=str(ROOT / "data/processed/replayed"),
                    help="battle records (default: rebuilt from saved frames by replay_frames.py --records)")
    ap.add_argument("--br-weight", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--repeat", type=float, default=0.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--observed", action=argparse.BooleanOptionalAction, default=True,
                    help="blend in the sets seen in our other battles")
    ap.add_argument("--utility", type=float, default=0.03, help="pseudo-share for learnable utility moves")
    args = ap.parse_args()
    query(args) if args.cmd == "query" else score(args)


if __name__ == "__main__":
    main()
