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
from pokesim.live.session import BattleSession
from pokesim.rl.myteam import to_id
from pokesim.search.lookahead import aggregate

CACHE = ROOT / "data" / "processed" / "scoped/eval_cache.jsonl"
BC_CACHE = ROOT / "data" / "processed" / "scoped/eval_bc.jsonl"


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
    from pokesim.data.dataset import scope_from, write_metadata
    from pokesim.data.prior_bundle import load_bundle, sampler
    from pokesim.runtime.bundle import file_hash

    scope = scope_from(args)
    bundle = load_bundle(args.prior_bundle, scope)
    sets = json.loads(Path(args.team).read_text())
    advisor = Advisor(sets, policy=None, opponent_policy=None, prior_bundle=bundle, sampler=sampler(bundle))
    advisor.engine.sampler.utility = args.utility
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = open(args.out, "w")
    recs = scope.records(args.split)
    if not recs:
        advisor.close()
        out.close()
        raise ValueError(f"no eligible {args.split} records")

    def on_turn(stem, rec, session, battle, mine, theirs):
        results = advisor.engine.query(battle, advisor.own_sets)
        out.write(json.dumps({"battle": stem, "turn": battle.turn, "opponent": theirs, "mine": mine,
                              "opp_active": battle.opponent_active_pokemon.species, "results": results}) + "\n")

    n = turns(args, recs, on_turn)
    advisor.close()
    out.close()
    write_metadata(Path(args.out), dict(scope.metadata("train"), target_split=args.split,
                   target_records=scope.metadata(args.split)["records"], prior_bundle_sha=file_hash(args.prior_bundle)))
    print(f"{n} decisions -> {args.out}")


def bc(args) -> None:
    from pokesim.data.dataset import scope_from, write_metadata
    from pokesim.data.prior_bundle import load_bundle, sampler
    from pokesim.live.opponent import OpponentPolicy
    from pokesim.runtime.bundle import file_hash

    scope = scope_from(args)
    predictor = OpponentPolicy(args.bc_policy, sampler=sampler(load_bundle(args.prior_bundle, scope)))
    Path(args.bc_out).parent.mkdir(parents=True, exist_ok=True)
    out = open(args.bc_out, "w")

    def on_turn(stem, rec, session, battle, mine, theirs):
        out.write(json.dumps({"battle": stem, "turn": battle.turn, "opponent": theirs,
                              "dist": dict(predictor.predict(session.log))}) + "\n")

    recs = scope.records(args.split)
    if not recs:
        out.close()
        raise ValueError(f"no eligible {args.split} records")
    n = turns(args, recs, on_turn)
    out.close()
    write_metadata(Path(args.bc_out), dict(scope.metadata("train"), target_split=args.split,
                   target_records=scope.metadata(args.split)["records"], prior_bundle_sha=file_hash(args.prior_bundle),
                   policy_sha=file_hash(args.bc_policy)))
    print(f"{n} decisions -> {args.bc_out}")


def score(args) -> dict:
    from pokesim.data.dataset import scope_from, validate_artifact

    scope = scope_from(args)
    meta = validate_artifact(Path(args.out), scope)
    if meta.get("target_split") != args.split or meta.get("target_records") != scope.metadata(args.split)["records"]:
        raise ValueError("evaluation cache belongs to another target split")
    model = {"br_weight": args.br_weight, "tau": args.tau, "repeat": args.repeat, "bc_weight": args.bc_weight}
    hit1 = hit3 = 0
    ll = []
    rows = [json.loads(line) for line in open(args.out)]
    bc_dist = {}
    if args.bc_weight > 0:
        bc_meta = validate_artifact(Path(args.bc_out), scope)
        if any(bc_meta.get(k) != meta.get(k) for k in ("target_split", "target_records", "prior_bundle_sha")):
            raise ValueError("BC and search caches use different data or priors")
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
    if not n:
        raise ValueError("evaluation has no scored decisions")
    res = {"n": n, "top1": hit1 / n, "top3": hit3 / n, "loglik": float(np.mean(ll))}
    print(f"{model}  n={n}  top1 {hit1}/{n} ({hit1 / n:.0%})  top3 {hit3}/{n} ({hit3 / n:.0%})  "
          f"mean log-lik {res['loglik']:.2f}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["query", "score", "bc"])
    ap.add_argument("--team", default=str(ROOT / "data/my_teams/baton.json"))
    ap.add_argument("--out", default=str(CACHE))
    from pokesim.data.dataset import add_scope_args
    from pokesim.data.prior_bundle import DEFAULT

    add_scope_args(ap)
    ap.add_argument("--prior-bundle", type=Path, default=DEFAULT)
    ap.add_argument("--split", choices=["development", "test"], default="development")
    ap.add_argument("--br-weight", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--repeat", type=float, default=0.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--bc-weight", type=float, default=0.0, help="share of the cloned human policy (needs bc)")
    ap.add_argument("--bc-policy", default=str(ROOT / "checkpoints/bc_human.zip"))
    ap.add_argument("--bc-out", default=str(BC_CACHE))
    ap.add_argument("--utility", type=float, default=0.03, help="pseudo-share for learnable utility moves")
    args = ap.parse_args()
    {"query": query, "score": score, "bc": bc}[args.cmd](args)


if __name__ == "__main__":
    main()
