"""Positions from our battles against people, answered by a slow lookahead: a check (and later a teacher) for our
fixed-team policy on the positions people actually lead us into.

build: replays each of our records (pokesim.rl.human.record_logs: Switch, Showdown ladder; us = p1) from our side
with our real sets and, at every turn decision, runs the lookahead with many set samples. Kept per position: the
policy observation (embed + TeamFeatures with items and threats), the action mask, the lookahead's equilibrium
mix and the value of each action (a win counts 3, a Pokemon's HP 1), and what we played.

The opponent's set priors only blend in sets seen in our battles played before the one being replayed
(--observed-scope before; "all" is the old setting, which lets a battle see its own and later opponents' sets).

report: for saved policies, how often their top choice is the lookahead's best action, and how often it is a
blunder (value more than --margin below the best), on those positions.

  uv run python -m pokesim.rl.teacher build [--samples 16 --seeds 3]
  uv run python -m pokesim.rl.teacher report checkpoints/ppo_baton_v2.zip checkpoints/ppo_baton_threats.zip
"""
import argparse
import json
import time

import numpy as np
from poke_env.environment import SinglesEnv

from pokesim.config import PROCESSED_DIR, ROOT
from pokesim.live.advisor import OPPONENT_MODEL
from pokesim.rl.features import embed
from pokesim.rl.human import Sides, decisions, record_logs
from pokesim.rl.myteam import TeamFeatures, to_id
from pokesim.search.lookahead import Lookahead, SetSampler, label_to_action

OUT = PROCESSED_DIR / "teacher.npz"
N_ACTIONS = 26


class OurSides(Sides):
    """Our side (p1) plays our real sets; the opponent's side as in Sides."""

    def __init__(self, team: list[dict], seed: int = 0):
        super().__init__(seed)
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
    team = json.loads(open(args.team).read())
    roster = [s["species"] for s in team]
    extra = TeamFeatures(roster, items=True, threats=True)
    from pokesim.live.observed import collect, load_records

    records = load_records() if args.observed_scope == "before" else []

    def sampler_before(stem: str) -> SetSampler:
        if args.observed_scope == "all":
            return SetSampler(["M-6"])
        started = stem.split("/")[-1][:15]
        earlier = [r for r in records if r.get("started", "") < started]
        return SetSampler(["M-6"], observed=collect(earlier))

    engine = Lookahead(SetSampler(["M-6"], observed={}), args.samples, args.seeds, args.rollout, OPPONENT_MODEL)
    own = {engine.sampler.base_id(to_id(s["species"])): s for s in team}
    sides = OurSides(team)
    rows = {k: [] for k in ("obs", "masks", "prob", "value", "played", "battle", "turn")}
    start = time.time()

    def probe(battle, sets):
        if battle.force_switch or not battle.active_pokemon or not battle.opponent_active_pokemon:
            return None
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
        advice = engine.advise(battle, own)
        prob, value = np.zeros(N_ACTIONS, np.float32), np.full(N_ACTIONS, np.nan, np.float32)
        for label, p, v in advice.ranked():
            a = label_to_action(label, battle)
            if a is not None and mask[a]:
                prob[a] += p
                value[a] = v if np.isnan(value[a]) else max(value[a], v)
        if not np.isfinite(value).any():
            return None
        return np.concatenate([embed(battle), extra(battle)]), mask, prob, value

    logs = record_logs()
    for k, (stem, log) in enumerate(logs):
        engine.sampler = sampler_before(stem)
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
    print(f"wrote {len(rows['obs'])} positions to {args.out}")


def report(args) -> None:
    from sb3_contrib import MaskablePPO

    from pokesim.rl.advise import action_probs

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
    b.add_argument("--observed-scope", choices=["before", "none", "all"], default="before",
                   help="sets seen in our battles: before = only battles played earlier; all = every battle")
    r = sub.add_parser("report")
    r.add_argument("policies", nargs="*")
    r.add_argument("--data", default=str(OUT))
    r.add_argument("--margin", type=float, default=1.0)
    args = ap.parse_args()
    build(args) if args.cmd == "build" else report(args)


if __name__ == "__main__":
    main()
