"""Train a masked-PPO policy against a baseline bot on the season's article teams.

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
            recent = self.results[-self.window:]
            self.logger.record("battle/win_rate", sum(recent) / len(recent))
            self.logger.record("battle/battles", len(self.results))
            print(f"step {self.num_timesteps}: win rate {sum(recent) / len(recent):.1%} "
                  f"(last {len(recent)} of {len(self.results)} battles)", flush=True)
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--opponent", default="heuristic", choices=list(OPPONENTS))
    parser.add_argument("--season", default="M-6")
    parser.add_argument("--name", default="ppo_m6")
    args = parser.parse_args()

    teams = load_teams(args.season)
    env = Monitor(MaskedSinglesEnv(OPPONENTS[args.opponent], TeamPool(teams, 1), TeamPool(teams, 2)),
                  info_keywords=("won",))
    model = MaskablePPO("MlpPolicy", env, device="cuda", n_steps=2048, batch_size=256, gamma=0.995,
                        learning_rate=3e-4, ent_coef=0.01, policy_kwargs={"net_arch": [256, 256]},
                        tensorboard_log=str(ROOT / "runs"), verbose=0)
    model.learn(total_timesteps=args.steps, callback=WinRateCallback(), tb_log_name=args.name)
    CKPT_DIR.mkdir(exist_ok=True)
    model.save(CKPT_DIR / args.name)
    print(f"saved {CKPT_DIR / args.name}.zip")


if __name__ == "__main__":
    main()
