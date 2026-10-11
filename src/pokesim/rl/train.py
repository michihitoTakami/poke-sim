"""Train a masked-PPO policy against a baseline bot on the season's article teams.

Runs its own Showdown server (port 8001) and restarts it every --segment steps, because the
server's battle process keeps growing (~45MB/min) during long runs.

With --league, a snapshot is saved after every segment and later segments play against a mix of the heuristic
bot and recent snapshots (plus the --init model), so the policy also learns against opponents that set up and heal.

With --my-team, our side always plays that team (lineup = the first 3 of --lineup, fixed), observations add
pokesim.rl.myteam.TeamFeatures, and opponents pick their 3 after seeing it (counter_preview). League opponents are
then the heuristic bot and --opp-snapshots (general policies), not snapshots of this team-specific policy.

Usage: uv run python -m pokesim.rl.train --steps 300000 [--opponent heuristic] [--reward sparse]
       uv run python -m pokesim.rl.train --steps 3000000 --league --init checkpoints/ppo_m6_status.zip
       uv run python -m pokesim.rl.train --my-team data/my_teams/baton.json --season all --reward sparse \
           --init checkpoints/bc_baton.zip --league --opp-snapshots checkpoints/ppo_m6_league3m.zip --name ppo_baton
"""

import argparse
import random
from functools import partial

from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from pokesim.config import ROOT
from pokesim.rl.bc import DEFAULT_LINEUP
from pokesim.rl.env import REWARDS, MaskedSinglesEnv
from pokesim.rl.myteam import FixedTeam, TeamFeatures, counter_preview, load_my_team
from pokesim.rl.opponents import LeaguePlayer
from pokesim.sim.pool import TeamPool, load_teams
from pokesim.sim.server import ShowdownServer

OPPONENTS = {"random": RandomPlayer, "maxbp": MaxBasePowerPlayer, "heuristic": SimpleHeuristicsPlayer}
CKPT_DIR = ROOT / "checkpoints"


class WinRateCallback(BaseCallback):
    """Logs the win rate over the last `window` finished battles."""

    def __init__(self, window: int = 200):
        super().__init__()
        self.window, self.results = window, []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "won" in info:
                self.results.append(info["won"])
        if self.results and self.n_calls % 5000 == 0:
            recent = self.results[-self.window :]
            self.logger.record("battle/win_rate", sum(recent) / len(recent))
            self.logger.record("battle/battles", len(self.results))
            print(
                f"step {self.num_timesteps}: win rate {sum(recent) / len(recent):.1%} "
                f"(last {len(recent)} of {len(self.results)} battles)",
                flush=True,
            )
        return True


def new_model(env) -> MaskablePPO:
    return MaskablePPO(
        "MlpPolicy",
        env,
        device="cuda",
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        learning_rate=3e-4,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=str(ROOT / "runs"),
        verbose=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--opponent", default="heuristic", choices=list(OPPONENTS))
    parser.add_argument("--season", default="M-6")
    parser.add_argument("--name", default="ppo_m6")
    parser.add_argument("--segment", type=int, default=100_000, help="restart the server every N steps")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--reward", default="dense", choices=list(REWARDS))
    parser.add_argument("--init", help="continue training from this checkpoint")
    parser.add_argument("--league", action="store_true", help="play against past snapshots and the heuristic bot")
    parser.add_argument("--league-size", type=int, default=5, help="number of recent snapshots kept as opponents")
    parser.add_argument("--p-heuristic", type=float, default=0.4)
    parser.add_argument("--my-team", help="train this fixed team (JSON sets, e.g. data/my_teams/baton.json)")
    parser.add_argument("--lineup", nargs=3, default=DEFAULT_LINEUP)
    parser.add_argument("--p-counter", type=float, default=0.7, help="chance the opponent counter-picks its 3")
    parser.add_argument("--opp-snapshots", nargs="*", default=[], help="general policies used as league opponents")
    args = parser.parse_args()

    teams = load_teams(args.season)
    CKPT_DIR.mkdir(exist_ok=True)
    league_dir = CKPT_DIR / f"{args.name}_league"
    snapshots = [args.init] if args.init and not args.my_team else []
    env_kwargs, my_team = {}, None
    if args.my_team:
        sets = load_my_team(args.my_team, args.lineup)
        my_team, snapshots = FixedTeam(sets), list(args.opp_snapshots)
        preview_rng = random.Random(0)
        env_kwargs = {"extra": TeamFeatures([s["species"] for s in sets]), "lineup": "123",
                      "opponent_preview": lambda battle: counter_preview(battle, preview_rng, args.p_counter)}
    callback, model, done, seg = WinRateCallback(), None, 0, 0
    while done < args.steps:
        with ShowdownServer(args.port) as server:
            if args.league:
                opponent = partial(LeaguePlayer, snapshots=snapshots[:1] + snapshots[1:][-args.league_size:],
                                   p_heuristic=args.p_heuristic, seed=seg)
            else:
                opponent = OPPONENTS[args.opponent]
            env = Monitor(
                MaskedSinglesEnv(
                    opponent,
                    my_team or TeamPool(teams, 1 + seg),
                    TeamPool(teams, 100 + seg),
                    server_configuration=server.configuration,
                    reward=args.reward,
                    **env_kwargs,
                ),
                info_keywords=("won",),
            )
            if model is None and args.init:
                model = MaskablePPO.load(args.init, env=env, device="cuda", tensorboard_log=str(ROOT / "runs"))
            elif model is None:
                model = new_model(env)
            else:
                model.set_env(env)
            steps = min(args.segment, args.steps - done)
            model.learn(
                total_timesteps=steps,
                callback=callback,
                tb_log_name=args.name,
                reset_num_timesteps=model.num_timesteps == 0,
            )
            done += steps
            seg += 1
            env.close()
        model.save(CKPT_DIR / args.name)
        if args.league and not args.my_team:
            league_dir.mkdir(exist_ok=True)
            model.save(league_dir / f"seg{seg:03d}")
            snapshots.append(str(league_dir / f"seg{seg:03d}.zip"))
        elif args.my_team:
            model.save(CKPT_DIR / f"{args.name}_seg{seg:03d}")
        print(f"segment {seg} done ({done} steps), saved {CKPT_DIR / args.name}.zip", flush=True)


if __name__ == "__main__":
    main()
