"""How well the set belief predicts what the battle shows next: each observation scored before it is seen.

For every damage / speed observation k of an opponent Pokémon in our records, the candidates are drawn from the
prior given what was revealed at that moment (as the lookahead draws them), then
  prior       P(observation k) with the candidates weighted equally (what the lookahead assumed so far)
  posterior   P(observation k) weighted by the observations of that Pokémon before k
Both use the same candidates and floor (pokesim.belief.sets). Higher mean log P is better. Battles are the unit
for the interval (resampled whole). The priors are the train-only bundle (pokesim.data.prior_bundle).

  uv run python -m pokesim.belief.evaluate --split development
  uv run python -m pokesim.belief.evaluate --since 20261015-033700 --until 20261015-094500
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pokesim.belief.sets import SetBelief, Worker, likelihood
from pokesim.config import ROOT
from pokesim.evaluation.metrics import bootstrap


def score_battle(belief: SetBelief, log: list[str]) -> list[dict]:
    belief.feed(log)
    rows = []
    for k, obs in enumerate(belief.observations):
        cands = belief.candidates(obs.species, *obs.revealed)
        prior = np.full(len(cands), 1 / len(cands))
        post = belief.weights(obs.species, cands, before=k)
        res = belief.results(obs, cands)
        lik = np.array([(1 - belief.eps) * likelihood(obs, r, c) + belief.eps for r, c in zip(res, cands)])
        rows.append({"kind": obs.kind, "attacker": obs.attacker, "species": obs.species,
                     "seen_before": sum(o.species == obs.species for o in belief.observations[:k]),
                     "prior": float(np.log(prior @ lik)), "posterior": float(np.log(post @ lik))})
    return rows


def summarize(rows: list[dict], groups: list[str]) -> dict:
    out = {}
    sel = {"all": lambda r: True,
           "damage taken (it attacks us)": lambda r: r["kind"] == "damage" and r["attacker"] == "p2",
           "damage dealt (we attack it)": lambda r: r["kind"] == "damage" and r["attacker"] == "p1",
           "speed": lambda r: r["kind"] == "speed",
           "two moves, no Choice item": lambda r: r["kind"] == "nochoice",
           "with earlier observations": lambda r: r["seen_before"] > 0}
    for name, f in sel.items():
        idx = [i for i, r in enumerate(rows) if f(r)]
        if not idx:
            continue
        prior = np.array([rows[i]["prior"] for i in idx])
        post = np.array([rows[i]["posterior"] for i in idx])
        g = [groups[i] for i in idx]
        diff = post - prior
        lo, hi = bootstrap(lambda j: float(diff[j].mean()), g)
        out[name] = {"n": len(idx), "battles": len(set(g)), "prior mean log P": round(float(prior.mean()), 3),
                     "posterior mean log P": round(float(post.mean()), 3),
                     "gain": round(float(diff.mean()), 3), "gain 95%": [round(lo, 3), round(hi, 3)]}
    return out


def records(args) -> list[tuple[str, dict]]:
    if args.split:
        from pokesim.data.dataset import DatasetScope

        return DatasetScope().records(args.split)
    out = []
    for p in sorted((ROOT / "data/raw/ladder").glob("*.json")):
        if (args.since and p.stem < args.since) or (args.until and p.stem >= args.until):
            continue
        out.append((p.stem, json.loads(p.read_text())))
    return out


def main() -> None:
    from pokesim.data.prior_bundle import DEFAULT, load_bundle, sampler

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "development", "calibration", "test"])
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--team", default=str(ROOT / "data/my_teams/baton.json"))
    ap.add_argument("--prior-bundle", type=Path, default=DEFAULT)
    ap.add_argument("--candidates", type=int, default=48)
    ap.add_argument("--out", type=Path, help="per-observation rows (jsonl)")
    args = ap.parse_args()
    sets = json.loads(Path(args.team).read_text())
    smp = sampler(load_bundle(args.prior_bundle))
    worker = Worker()
    rows, groups = [], []
    try:
        for stem, rec in records(args):
            if not rec.get("log"):
                continue
            try:
                got = score_battle(SetBelief(sets, smp, worker, n=args.candidates), rec["log"])
            except Exception as e:  # noqa: BLE001 - a log the session cannot follow: skip the battle
                print("skip", stem, repr(e)[:120], file=sys.stderr, flush=True)
                continue
            rows += got
            groups += [stem] * len(got)
    finally:
        worker.close()
    if args.out:
        args.out.write_text("".join(json.dumps(dict(r, battle=g)) + "\n" for r, g in zip(rows, groups)))
    print(json.dumps(summarize(rows, groups), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
