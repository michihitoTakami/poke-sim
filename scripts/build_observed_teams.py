"""Opponent teams from our own battles -> teams_observed.json and teams_mix.json for training (pokesim.rl.train).

Each battle's opponent 6 (team preview) become full sets: what that battle revealed (moves, item, ability) is kept,
the rest is sampled from the set data blended with everything we have seen (pokesim.live.observed). Several samples
per battle cover the unrevealed parts. teams_mix.json = the published pool + the observed teams repeated --weight
times, so `--season mix` draws real opponents often without forgetting the wider pool.

  uv run python scripts/build_observed_teams.py [--samples 6 --weight 3]
"""
import argparse
import json
import subprocess
from pathlib import Path

from pokesim.config import FORMAT, PROCESSED_DIR, ROOT, SHOWDOWN_DIR
from pokesim.live.observed import collect
from pokesim.rl.myteam import to_id


def validate(teams: list[list[dict]]) -> list[dict]:
    proc = subprocess.run(["node", str(ROOT / "services/sim-tools/validate_teams.js"), str(SHOWDOWN_DIR), FORMAT],
                          input=json.dumps(teams), capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def main() -> None:
    ap = argparse.ArgumentParser()
    from pokesim.data.dataset import add_scope_args, scope_from, write_metadata
    from pokesim.data.prior_bundle import DEFAULT, load_bundle
    from pokesim.data.prior_bundle import sampler as frozen_sampler

    add_scope_args(ap)
    ap.add_argument("--prior-bundle", type=Path, default=DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=PROCESSED_DIR / "scoped")
    ap.add_argument("--samples", type=int, default=6, help="teams sampled per battle")
    ap.add_argument("--weight", type=int, default=3, help="copies of each observed team in teams_mix.json")
    args = ap.parse_args()

    scope = scope_from(args)
    records = scope.records("train")
    sampler = frozen_sampler(load_bundle(args.prior_bundle, scope))
    candidates, meta = [], []
    for battle_id, rec in records:
        seen = collect([rec])
        species = [to_id(s) for s in rec["preview"]["species"]]
        for k in range(args.samples):
            for _attempt in range(20):  # item clause / one Mega stone: resample until legal-looking
                team = []
                for sid in species:
                    rev = seen.get(sid, {})
                    item = next(iter(rev.get("items", {})), None)
                    ability = next(iter(rev.get("abilities", {})), None)
                    team.append(sampler.sample(sid, list(rev.get("moves", {})), item, ability))
                items = [s["item"] for s in team if s["item"]]
                megas = [i for i in items if i.endswith("ite") or i.endswith("ite X") or i.endswith("ite Y")]
                if len(items) == len(set(items)) and len(megas) <= 1:
                    break
            candidates.append(team)
            meta.append({"battle": battle_id, "sample": k, "result": rec.get("result")})
    results = validate(candidates)
    teams = [{**m, "sets": t, "packed": r["packed"]} for m, t, r in zip(meta, candidates, results) if r["ok"]]
    rejected = [r["problems"][:2] for r in results if not r["ok"]]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    observed_path = args.out_dir / "teams_observed.json"
    observed_path.write_text(json.dumps(teams, ensure_ascii=False, indent=1))
    pool = json.loads((PROCESSED_DIR / "teams_all.json").read_text())
    mix = pool + teams * args.weight
    mix_path = args.out_dir / "teams_mix.json"
    mix_path.write_text(json.dumps(mix, ensure_ascii=False))
    from pokesim.runtime.bundle import file_hash

    metadata = dict(scope.metadata("train"), prior_bundle_sha=file_hash(args.prior_bundle),
                    public_pool_sha=file_hash(PROCESSED_DIR / "teams_all.json"))
    write_metadata(observed_path, metadata)
    write_metadata(mix_path, metadata)
    print(f"{len(records)} battles -> {len(teams)}/{len(candidates)} legal observed teams; "
          f"teams_mix.json = {len(pool)} pool + {len(teams)} x {args.weight} observed")
    for problem in rejected[:5]:
        print("  rejected:", problem)


if __name__ == "__main__":
    main()
