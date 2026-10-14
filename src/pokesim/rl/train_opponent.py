"""Train an opponent against our team: the learner plays the opponents' teams, starting from the policy cloned from
human players (pokesim.rl.human), so it stays close to how people play while it learns to beat this team.

The learner only sees what a player sees (pokesim.rl.features.embed: our 6 at team preview, then what we
reveal). So that it cannot simply memorise our sets, our side varies between battles:
  - with probability --p-real our real team (data/my_teams/baton.json), piloted by our trained policy
    (--our-policy) or, with probability --p-script, the scripted Baton Pass plan;
  - otherwise the same 6 species with sets sampled from the set data, piloted by the heuristic bot.
Our 3 are the usual lineup with probability --p-usual-lineup, else 3 at random. The learner's own 3 are picked by
counter_preview (it prefers members that stop set-up), and its teams come from --season (default: the published
pool plus the teams we faced, scripts/build_observed_teams.py).

Rewards use the "win" preset (the sparse preset scaled to a ±1 win), the scale of the cloned policy's value head.

  uv run python -m pokesim.rl.train_opponent --init checkpoints/bc_human.zip --name opp_baton --steps 3000000
"""
import argparse
import json
import random
import subprocess
from functools import partial
from pathlib import Path

from poke_env.player import Player, SimpleHeuristicsPlayer
from poke_env.teambuilder import Teambuilder
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from pokesim.config import FORMAT, ROOT, SHOWDOWN_DIR
from pokesim.rl.baton import BatonScript, PolicyPilot
from pokesim.rl.bc import DEFAULT_LINEUP
from pokesim.rl.myteam import CounterPreview, FixedTeam, load_my_team
from pokesim.rl.train import CKPT_DIR, WinRateCallback, make_env
from pokesim.sim.pool import TeamPool, load_teams
from pokesim.sim.server import ShowdownServer


def variant_teams(sets: list[dict], n: int, seed: int = 0, prior_bundle: dict | None = None) -> list[str]:
    """`n` packed teams with our 6 species and sets sampled from the set data (legal per the validator)."""
    from pokesim.data.prior_bundle import sampler as frozen_sampler
    from pokesim.search.lookahead import SetSampler

    sampler = SetSampler(["M-6"], seed=seed) if prior_bundle is None else frozen_sampler(prior_bundle, seed)
    cands = []
    for _ in range(n * 2):
        for _attempt in range(20):
            team = [sampler.sample(sampler.base_id(s["species"].lower().replace("-", ""))) for s in sets]
            items = [s["item"] for s in team if s["item"]]
            if len(items) == len(set(items)):
                break
        cands.append(team)
    proc = subprocess.run(["node", str(ROOT / "services/sim-tools/validate_teams.js"), str(SHOWDOWN_DIR), FORMAT],
                          input=json.dumps(cands), capture_output=True, text=True, check=True)
    ok = [r["packed"] for r in json.loads(proc.stdout) if r["ok"]]
    return ok[:n]


class OurTeams(Teambuilder):
    """Our real team with probability p_real, else a variant; remembers which it handed out last."""

    def __init__(self, real: str, variants: list[str], p_real: float, seed: int = 0):
        self.real, self.variants, self.p_real = real, variants, p_real
        self.rng = random.Random(seed)
        self.last_real = True

    def yield_team(self) -> str:
        self.last_real = not self.variants or self.rng.random() < self.p_real
        return self.real if self.last_real else self.rng.choice(self.variants)


class UsualLineup:
    """Our team preview (picklable): the usual 3 (the first 3 of the team) with probability p, else 3 at random."""

    def __init__(self, p: float, seed: int = 0):
        self.p, self.rng = p, random.Random(seed)

    def __call__(self, battle) -> str:
        picks = [0, 1, 2] if self.rng.random() < self.p else self.rng.sample(range(6), 3)
        return "/team " + "".join(str(i + 1) for i in picks + [i for i in range(6) if i not in picks])


class OurSide(Player):
    """Our side as the environment's opponent: policy / script on the real team, heuristic bot on variants.
    `team` must be the OurTeams the environment hands out (it says whether the battle's team is the real one)."""

    def __init__(self, *args, policy: str, roster: list[str], p_script: float = 0.2, seed: int = 0,
                 prior_bundle: dict | None = None, **kwargs):
        fmt = kwargs.get("battle_format")
        self.rng, self.p_script = random.Random(seed), p_script
        self.pilot = PolicyPilot(battle_format=fmt, policy=policy, roster=roster, deterministic=False,
                                 start_listening=False, prior_bundle=prior_bundle)
        self.script = BatonScript(battle_format=fmt, start_listening=False)
        self.heuristic = SimpleHeuristicsPlayer(battle_format=fmt, start_listening=False)
        self.assigned: dict[str, object] = {}
        super().__init__(*args, **kwargs)

    def strategy(self, battle):
        if battle.battle_tag not in self.assigned:
            if self._team.last_real:
                self.assigned[battle.battle_tag] = self.script if self.rng.random() < self.p_script else self.pilot
            else:
                self.assigned[battle.battle_tag] = self.heuristic
        return self.assigned[battle.battle_tag]

    def choose_move(self, battle):
        try:
            return self.strategy(battle).choose_move(battle)
        except Exception:  # noqa: BLE001 - keep the battle going
            return self.choose_random_move(battle)


def main() -> None:
    from pokesim.data.dataset import add_scope_args, scope_from, write_metadata
    from pokesim.data.prior_bundle import DEFAULT, load_bundle

    ap = argparse.ArgumentParser()
    add_scope_args(ap)
    ap.add_argument("--prior-bundle", type=Path, default=DEFAULT)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--name", default="opp_baton")
    ap.add_argument("--init", default=str(CKPT_DIR / "bc_human.zip"))
    ap.add_argument("--team", default=str(ROOT / "data/my_teams/baton.json"))
    ap.add_argument("--lineup", nargs=3, default=DEFAULT_LINEUP, help="our usual 3 (lead first)")
    ap.add_argument("--our-policy", default=str(CKPT_DIR / "ppo_baton_any.zip"))
    ap.add_argument("--season", default="mix", help="the learner's teams (teams_<season>.json)")
    ap.add_argument("--p-real", type=float, default=0.6)
    ap.add_argument("--p-script", type=float, default=0.2)
    ap.add_argument("--p-usual-lineup", type=float, default=0.6)
    ap.add_argument("--p-counter", type=float, default=0.7)
    ap.add_argument("--variants", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--segment", type=int, default=100_000)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=1024)
    args = ap.parse_args()

    scope = scope_from(args)
    bundle = load_bundle(args.prior_bundle, scope)
    sets = load_my_team(args.team, args.lineup)
    roster = [s["species"] for s in sets]
    real = FixedTeam(sets).packed
    variants = variant_teams(sets, args.variants, prior_bundle=bundle)
    print(f"{len(variants)} variant teams of our 6", flush=True)
    teams = load_teams(args.season, scope=scope)
    callback, model, done, seg = WinRateCallback(), None, 0, 0
    while done < args.steps:
        with ShowdownServer(args.port) as server:
            factories = []
            for rank in range(args.n_envs):
                seed = seg * 100 + rank
                ours = OurTeams(real, variants, args.p_real, seed=seed)
                opponent = partial(OurSide, policy=args.our_policy, roster=roster, p_script=args.p_script,
                                   seed=seed, prior_bundle=bundle)
                kwargs = {"lineup": CounterPreview(args.p_counter, seed=seed),
                          "opponent_preview": UsualLineup(args.p_usual_lineup, seed=seed)}
                factories.append(partial(make_env, opponent, TeamPool(teams, 1 + seed), ours,
                                         server.configuration, "win", kwargs, args.n_envs > 1))
            env = SubprocVecEnv(factories, start_method="spawn") if args.n_envs > 1 else factories[0]()
            if model is None:
                model = MaskablePPO.load(args.init, env=env, device="cuda", tensorboard_log=str(ROOT / "runs"),
                                         custom_objects={"n_steps": args.n_steps, "learning_rate": args.lr,
                                                         "ent_coef": args.ent_coef, "gamma": 0.995})
            else:
                model.set_env(env)
            steps = min(args.segment, args.steps - done)
            model.learn(total_timesteps=steps, callback=callback, tb_log_name=args.name,
                        reset_num_timesteps=model.num_timesteps == 0)
            done += steps
            seg += 1
            env.close()
        model.save(CKPT_DIR / args.name)
        from pokesim.runtime.bundle import file_hash

        write_metadata(CKPT_DIR / f"{args.name}.zip", dict(scope.metadata("train"),
                       prior_bundle_sha=file_hash(args.prior_bundle), reward="win"))
        model.save(CKPT_DIR / f"{args.name}_seg{seg:03d}")
        print(f"segment {seg} done ({done} steps), saved {Path(CKPT_DIR / args.name)}.zip", flush=True)


if __name__ == "__main__":
    main()
