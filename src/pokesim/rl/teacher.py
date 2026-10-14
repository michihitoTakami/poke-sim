"""Positions from our battles against people, answered by a slow lookahead: a check (and later a teacher) for our
fixed-team policy on the positions people actually lead us into.

build: replays each of our records (pokesim.rl.human.record_logs: Switch, Showdown ladder; us = p1) from our side
with our real sets and, at every turn decision, runs the lookahead with many set samples. Kept per position: the
policy observation (embed + TeamFeatures with items and threats), the action mask, the lookahead's equilibrium
mix and the value of each action (a win counts 3, a Pokemon's HP 1), and what we played.

The opponent's set priors are frozen on the manifest's train records. The default target is development.
Values use the terminal planner's ranking units, not calibrated win probabilities or the old HP/+3 scale.

report: for saved policies, how often their top choice is the lookahead's best action, and how often it is a
blunder (value more than --margin below the best), on those positions.

  uv run python -m pokesim.rl.teacher build [--samples 16 --seeds 3]
  uv run python -m pokesim.rl.teacher report checkpoints/ppo_baton_v2.zip checkpoints/ppo_baton_threats.zip
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from poke_env.environment import SinglesEnv

from pokesim.config import PROCESSED_DIR, ROOT
from pokesim.live.advisor import OPPONENT_MODEL
from pokesim.rl.features import embed
from pokesim.rl.human import Sides, decisions, record_logs
from pokesim.rl.myteam import TeamFeatures, to_id
from pokesim.search.lookahead import Lookahead, label_to_action

OUT = PROCESSED_DIR / "scoped/teacher.npz"
N_ACTIONS = 26


class OurSides(Sides):
    """Our side (p1) plays our real sets; the opponent's side as in Sides."""

    def __init__(self, team: list[dict], seed: int = 0, sampler=None):
        super().__init__(seed, sampler=sampler)
        self.team = {to_id(s["species"]): s for s in team}

    def sets(self, log: str, role: str) -> list[dict]:
        if role != "p1":
            return super().sets(log, role)
        preview = [ln.split("|")[3].split(",")[0] for ln in log.split("\n") if ln.startswith("|poke|p1|")]
        out = []
        for name in preview:
            key = to_id(name)
            out.append(self.team.get(key) or next(s for k, s in self.team.items() if key.startswith(k)))
        return out


def build(args) -> None:
    from pokesim.data.dataset import scope_from, write_metadata
    from pokesim.data.prior_bundle import load_bundle, sampler
    from pokesim.live.advisor import PLANNER, PLANNER_SEARCH
    from pokesim.runtime.bundle import file_hash
    from pokesim.search.planner import plan

    scope = scope_from(args)
    bundle = load_bundle(args.prior_bundle, scope)
    team = json.loads(open(args.team).read())
    roster = [s["species"] for s in team]
    extra = TeamFeatures(roster, items=True, threats=True, prior_bundle=bundle)
    settings = dict(PLANNER_SEARCH, seeds=args.seeds, rollout=args.rollout, deadlineMs=None)
    engine = Lookahead(sampler(bundle), args.samples, args.seeds, args.rollout, OPPONENT_MODEL, search=settings)
    own = {engine.sampler.base_id(to_id(s["species"])): s for s in team}
    sides = OurSides(team, sampler=sampler(bundle))
    rows = {k: [] for k in ("obs", "masks", "prob", "value", "played", "battle", "turn")}
    start = time.time()

    def probe(battle, sets):
        if battle.force_switch or not battle.active_pokemon or not battle.opponent_active_pokemon:
            return None
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
        advice = plan(engine.query(battle, own), OPPONENT_MODEL, rho=PLANNER["rho"])
        prob, value = np.zeros(N_ACTIONS, np.float32), np.full(N_ACTIONS, np.nan, np.float32)
        for label, p, v in advice.ranked():
            a = label_to_action(label, battle)
            if a is not None and mask[a]:
                prob[a] += p
                value[a] = v if np.isnan(value[a]) else max(value[a], v)
        if not np.isfinite(value).any():
            return None
        return np.concatenate([embed(battle), extra(battle)]), mask, prob, value

    logs = record_logs(records=scope.records(args.split))
    if not logs:
        engine.close()
        raise ValueError(f"no eligible {args.split} records")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for k, (stem, log) in enumerate(logs):
        try:
            ds, _ = decisions(log, "p1", sides, probe=probe, keep_failed=True)
        except Exception as e:  # noqa: BLE001 - a record our session cannot follow
            print("skip", stem, repr(e)[:120], flush=True)
            continue
        for d in ds:
            if d.extra is None or d.kind != "turn":
                continue
            obs, mask, prob, value = d.extra
            for key, v in zip(("obs", "masks", "prob", "value", "played", "battle", "turn"),
                              (obs, mask, prob, value, d.action, stem, d.turn)):
                rows[key].append(v)
        print(f"[{k + 1}/{len(logs)}] {stem}: {len(rows['obs'])} positions, {time.time() - start:.0f}s", flush=True)
        np.savez_compressed(args.out, **{k: np.array(v) for k, v in rows.items()})
    engine.close()
    write_metadata(Path(args.out), dict(scope.metadata("train"), target_split=args.split,
                   target_records=scope.metadata(args.split)["records"], search=settings,
                   prior_bundle_sha=file_hash(args.prior_bundle)))
    print(f"wrote {len(rows['obs'])} positions to {args.out}")


def report(args) -> None:
    from sb3_contrib import MaskablePPO

    from pokesim.data.dataset import scope_from, validate_artifact
    from pokesim.rl.advise import action_probs

    scope = scope_from(args)
    meta = validate_artifact(Path(args.data), scope)
    if meta.get("target_records") != scope.metadata(meta["target_split"])["records"]:
        raise ValueError("teacher target records changed: rebuild it")
    data = np.load(args.data)
    value, masks = data["value"], data["masks"]
    best = np.nanmax(value, axis=1)
    print(f"{len(value)} positions from {len(set(data['battle']))} battles")
    played = data["played"]
    rows = [("what we played", played)]
    for path in args.policies:
        model = MaskablePPO.load(path, device="cpu")
        n = model.observation_space.shape[0]
        obs = data["obs"][:, :n] if data["obs"].shape[1] >= n else None
        if obs is None:
            print(f"  {path}: needs {n} inputs, the data has {data['obs'].shape[1]}")
            continue
        top = np.array([int(np.argmax(action_probs(model, o, m))) for o, m in zip(obs, masks)])
        rows.append((path, top))
    rows.append(("lookahead mix (top)", np.argmax(data["prob"], axis=1)))
    for name, choice in rows:
        v = value[np.arange(len(value)), choice]
        v = np.where(np.isnan(v), best - 10, v)
        regret = best - v
        print(f"  {name:45s} best {np.mean(regret < 1e-6):5.1%}  blunder {np.mean(regret > args.margin):5.1%}  "
              f"mean regret {regret.mean():.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--team", default=str(ROOT / "data/my_teams/baton.json"))
    b.add_argument("--samples", type=int, default=16)
    b.add_argument("--seeds", type=int, default=3)
    b.add_argument("--rollout", type=int, default=3)
    b.add_argument("--out", default=str(OUT))
    from pokesim.data.dataset import add_scope_args
    from pokesim.data.prior_bundle import DEFAULT

    add_scope_args(b)
    b.add_argument("--prior-bundle", type=Path, default=DEFAULT)
    b.add_argument("--split", choices=["train", "development", "test"], default="development")
    r = sub.add_parser("report")
    add_scope_args(r)
    r.add_argument("policies", nargs="*")
    r.add_argument("--data", default=str(OUT))
    r.add_argument("--margin", type=float, default=0.15, help="planner ranking gap; not win probability")
    args = ap.parse_args()
    build(args) if args.cmd == "build" else report(args)


if __name__ == "__main__":
    main()
