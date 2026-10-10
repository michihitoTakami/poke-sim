"""Train a masked-PPO policy against a baseline bot on the season's article teams.

Runs its own Showdown server (port 8001) and restarts it every --segment steps, because the
server's battle process keeps growing (~45MB/min) during long runs.

Usage: uv run python -m pokesim.rl.train --steps 300000 [--opponent heuristic]
"""

import argparse

from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from pokesim.config import ROOT
from pokesim.rl.env import MaskedSinglesEnv
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--opponent", default="heuristic", choices=list(OPPONENTS))
    parser.add_argument("--season", default="M-6")
    parser.add_argument("--name", default="ppo_m6")
    parser.add_argument("--segment", type=int, default=100_000, help="restart the server every N steps")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    teams = load_teams(args.season)
    CKPT_DIR.mkdir(exist_ok=True)
    callback, model, done, seg = WinRateCallback(), None, 0, 0
    while done < args.steps:
        with ShowdownServer(args.port) as server:
            env = Monitor(
                MaskedSinglesEnv(
                    OPPONENTS[args.opponent],
                    TeamPool(teams, 1 + seg),
                    TeamPool(teams, 100 + seg),
                    server_configuration=server.configuration,
                ),
                info_keywords=("won",),
            )
            if model is None:
                model = MaskablePPO(
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
        print(f"segment {seg} done ({done} steps), saved {CKPT_DIR / args.name}.zip", flush=True)


if __name__ == "__main__":
    main()
