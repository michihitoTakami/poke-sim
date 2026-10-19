"""Compare relay policies on identical preview candidates, hypothetical sets and chance streams."""

import argparse
import copy
import json
import time
from pathlib import Path

from eval_relay_tempo import ReplayWorker

from pokesim.belief.team_catalog import TeamCatalog
from pokesim.config import ROOT
from pokesim.data.prior_bundle import load_bundle, sampler
from pokesim.evaluation.branch_matches import atomic_json, file_sha, hash_seed, tree_sha
from pokesim.live.lineup_search import LineupSearch
from pokesim.live.opponent_selection import OpponentLineups, pick_rows
from pokesim.live.preview_matchups import PreviewCoverage
from pokesim.live.selection import LadderLineupModel, PickModel, ladder_rows
from pokesim.search.lookahead import Lookahead


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, default=ROOT / "data/raw/ladder")
    ap.add_argument("--priors", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, default=ROOT / "data/processed/top_teams.json")
    ap.add_argument("--pick-rates", type=Path, default=ROOT / "data/processed/pick_rates.json")
    ap.add_argument("--lineup-prior", type=Path, default=ROOT / "data/lineup_prior.json")
    ap.add_argument("--baseline-root", type=Path, required=True)
    ap.add_argument("--cutoff", required=True, help="last completed record timestamp, YYYYMMDD-HHMMSS")
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    hash_seed()
    root = ROOT
    out = args.out
    records = args.records
    cutoff = args.cutoff + "-\uffff"
    sets = json.loads((root / "data/my_teams/baton.json").read_text())
    roster = [s["species"] for s in sets]
    prior = sampler(load_bundle(args.priors))
    prior.catalog = TeamCatalog.load(prior.base_id, path=args.catalog)
    after = Lookahead(prior)
    before = Lookahead(None)
    before.worker.kill()
    before.worker = ReplayWorker(args.baseline_root / "services/sim-tools/lookahead.js")
    before.query_timeout = after.query_timeout = 120
    rows = [r for r in ladder_rows(records, roster) if r["name"] <= cutoff]
    model = LadderLineupModel(rows, prior=json.loads((args.lineup_prior).read_text()))
    picks_model = PickModel(args.pick_rates)
    opps = OpponentLineups(
        [r for r in pick_rows(records, roster) if r["record"] <= cutoff], picks_model, signature=args.cutoff
    )
    selection = LineupSearch(sets, after)
    coverage = PreviewCoverage(sets, after)
    report = {
        "scope": "paired development preview diagnostics; fixed history cutoff; not causal or calibrated win rates",
        "cutoff": cutoff,
        "catalog_sha": file_sha(args.catalog),
        "worker_sha": {
            "before": tree_sha(args.baseline_root / "services/sim-tools", {".js"}),
            "after": tree_sha(root / "services/sim-tools", {".js"}),
        },
        "cases": [],
    }
    try:
        for stamp in list(dict.fromkeys(c["stamp"] for c in json.loads(args.cases.read_text()))):
            paths = list(records.glob(stamp + "*.json"))
            if not paths:
                continue
            path = sorted(paths)[0]
            rec = json.loads(path.read_text())
            preview = rec["preview"]["species"]
            if any(x["preview"] == preview for x in report["cases"]):
                continue
            picks = opps.predict(picks_model.predict(preview))
            ranking = model.rank(roster, picks, coverage.profile(picks))
            captured = []
            original = after._query

            def capture(req):
                if "lineupMatches" in req:
                    captured.append(copy.deepcopy(req))
                return original(req)

            after._query = capture
            t = time.monotonic()
            ranked, info = selection.rerank(ranking, picks)
            elapsed = time.monotonic() - t
            after._query = original
            req = captured[0]
            atomic_json(out / (stamp + ".request.json"), req)
            old = before._query(req)
            new = selection.cache[next(reversed(selection.cache))]
            variants = {}
            for name, reply in [("before", old), ("after", new)]:
                scores = []
                for i, trio in enumerate(req["lineupMatches"]["ours"]):
                    trials = [t for t in reply["trials"] if t["candidate"] == i]
                    nom = [t for t in trials if t["world"] < 4]
                    mean = sum(t["target"] or 0 for t in nom) / len(nom)
                    worst = [
                        min(t["target"] or 0 for t in trials if t["world"] == w)
                        for w in range(len(req["lineupMatches"]["worlds"]))
                    ]
                    lineup = [m["species"] for m in trio]
                    evidence = next(e for e in info['candidates'] if e['lineup'] == lineup)
                    simulation = 0.7 * mean + 0.3 * sum(worst) / len(worst)
                    history_weight = evidence['history_weight']
                    scores.append(
                        {
                            "lineup": lineup,
                            "utility": (1 - history_weight) * simulation + history_weight * evidence['history_mean'],
                            "simulation_utility": simulation,
                            "history_weight": history_weight,
                            "wins": sum(t["outcome"] == "win" for t in trials),
                            "losses": sum(t["outcome"] == "loss" for t in trials),
                            "unfinished": sum(t["outcome"] == "unfinished" for t in trials),
                            "failed": sum(t["outcome"] == "failed" for t in trials),
                            "payload_games": sum(bool(t.get("relay", {}).get("transfers")) for t in trials),
                            "two_hop_games": sum(len(t.get("relay", {}).get("transfers", [])) >= 2 for t in trials),
                        }
                    )
                scores.sort(key=lambda r: -r["utility"])
                variants[name] = {"ranked": scores, "trials": len(reply["trials"])}
            assert variants["after"]["ranked"][0]["lineup"] == ranked[0][0]
            atomic_json(out / (stamp + ".trials.json"), {"before": old, "after": new})
            report["cases"].append(
                {
                    "file": path.name,
                    "sha256": file_sha(path),
                    "preview": preview,
                    "variants": variants,
                    "new_seconds": elapsed,
                    "candidates": len(req["lineupMatches"]["ours"]),
                    "worlds": len(req["lineupMatches"]["worlds"]),
                    "stress_worlds": info['stress_worlds'],
                    "policies": req["lineupMatches"]["policies"],
                }
            )
            atomic_json(out / "report.json", report)
            print(path.name, {k: v["ranked"][0] for k, v in variants.items()}, "seconds", elapsed, flush=True)
    finally:
        before.close()
        after.close()


if __name__ == "__main__":
    main()
