"""Play our team on a Showdown server with the advisor (policy + lookahead), saving every battle as a record.

  # public server (needs a registered account): credentials in ~/.config/poke-sim/showdown.env
  #   PS_USERNAME=...   PS_PASSWORD=...
  uv run python scripts/ladder.py --battles 30
  # local test against a random player
  uv run python scripts/ladder.py --local --battles 2

Team preview: the lineup ranked best by the matchup model for the opponent's 6. Each turn: the advisor's top
legal action. Records go to data/raw/ladder/<time>.json in the format of data/raw/live/logs (opponent preview,
lineup, protocol log, result), so pokesim.live.observed and scripts/eval_records.py can use them.
Showdown is not the Switch game: mechanics follow Showdown's Champions format and opponents are Showdown players.
"""
import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from poke_env import AccountConfiguration, LocalhostServerConfiguration, ShowdownServerConfiguration
from poke_env.environment import SinglesEnv
from poke_env.player import Player, RandomPlayer

from pokesim.config import FORMAT, RAW_DIR, ROOT
from pokesim.live.advisor import Advisor
from pokesim.live.selection import MatchupModel, PickModel
from pokesim.rl.myteam import FixedTeam, to_id
from pokesim.search.lookahead import label_to_action
from pokesim.sim.pool import TeamPool, load_teams

OUT = RAW_DIR / "ladder"
CREDENTIALS = Path.home() / ".config" / "poke-sim" / "showdown.env"


def credentials() -> tuple[str, str] | None:
    env = dict(os.environ)
    if CREDENTIALS.exists():
        for line in CREDENTIALS.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    if env.get("PS_USERNAME") and env.get("PS_PASSWORD"):
        return env["PS_USERNAME"], env["PS_PASSWORD"]
    return None


class AdvisorPlayer(Player):
    def __init__(self, sets: list[dict], policy: str | None, out: Path = OUT, **kwargs):
        self.sets, self.out = sets, out
        self.advisor = Advisor(sets, policy=policy)
        self.picks, self.matchups = PickModel(), MatchupModel()
        self.lineups: dict[str, list[str]] = {}
        self.logs: dict[str, list[str]] = {}  # battle tag -> protocol lines
        super().__init__(team=FixedTeam(sets), **kwargs)

    async def _handle_battle_message(self, split_messages):
        tag = split_messages[0][0].lstrip(">").strip() if split_messages and split_messages[0] else ""
        if tag:
            self.logs.setdefault(tag, []).extend("|".join(m) for m in split_messages[1:]
                                                 if len(m) > 1 and m[1] not in ("request", "t:", "j", "l", "c"))
        await super()._handle_battle_message(split_messages)

    def teampreview(self, battle) -> str:
        opp = [m.species for m in battle.teampreview_opponent_team]
        ranking = self.matchups.rank([s["species"] for s in self.sets], self.picks.predict(opp))
        lineup = ranking[0][0] if ranking else [s["species"] for s in self.sets[:3]]
        self.lineups[battle.battle_tag] = lineup
        order = [next(i for i, s in enumerate(self.sets) if to_id(s["species"]) == to_id(x)) for x in lineup]
        order += [i for i in range(len(self.sets)) if i not in order]
        return "/team " + "".join(str(i + 1) for i in order)

    def last_opponent_move(self, battle) -> str | None:
        role = battle.player_role
        for line in reversed(self.logs.get(battle.battle_tag, [])):
            ev = line.split("|")
            if len(ev) > 3 and ev[1] in ("switch", "drag") and not ev[2].startswith(role):
                return None
            if len(ev) > 3 and ev[1] == "move" and not ev[2].startswith(role):
                return "move:" + to_id(ev[3])
        return None

    def choose_move(self, battle):
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
        try:
            advice = self.advisor.advise(battle, self.last_opponent_move(battle),
                                         self.logs.get(battle.battle_tag, []),
                                         battle.player_role)
            for label, _ in advice.actions:
                action = label_to_action(label, battle)
                if action is not None and mask[action]:
                    return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)
        except Exception as e:  # noqa: BLE001 - never stall a live battle
            print("advice failed:", repr(e)[:200], flush=True)
        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        log = self.logs.pop(battle.battle_tag, [])
        if battle.player_role == "p2":  # records always have us as p1 (BattleSession, observed, eval_records)
            log = [re.sub(r"\bp([12])(?=[a-z]?[:|]|$)", lambda m: "p" + ("2" if m.group(1) == "1" else "1"), line)
                   for line in log]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        record = {"started": stamp, "result": "win" if battle.won else "loss" if battle.lost else None,
                  "team": [s["species"] for s in self.sets],
                  "preview": {"species": [m.species for m in battle.teampreview_opponent_team]},
                  "lineup": self.lineups.get(battle.battle_tag), "log": log, "advice": [],
                  "source": f"showdown {battle.battle_tag}", "opponent": battle.opponent_username,
                  "rating": getattr(battle, "rating", None)}
        (self.out / f"{stamp}-{battle.battle_tag.split('-')[-1]}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1))
        print(f"{battle.battle_tag}: {record['result']} vs {battle.opponent_username}  "
              f"({self.n_won_battles}-{self.n_lost_battles})", flush=True)


async def run(args) -> None:
    sets = json.loads(Path(args.team).read_text())
    if args.local:
        me = AdvisorPlayer(sets, args.policy, out=Path(args.out), battle_format=FORMAT,
                           server_configuration=LocalhostServerConfiguration, max_concurrent_battles=1)
        foe = RandomPlayer(battle_format=FORMAT, team=TeamPool(load_teams("observed"), 0),
                           server_configuration=LocalhostServerConfiguration)
        await me.battle_against(foe, n_battles=args.battles)
        return
    cred = credentials()
    if cred is None:
        raise SystemExit(f"no Showdown account: put PS_USERNAME / PS_PASSWORD in {CREDENTIALS}")
    me = AdvisorPlayer(sets, args.policy, out=Path(args.out), battle_format=FORMAT,
                       account_configuration=AccountConfiguration(*cred),
                       server_configuration=ShowdownServerConfiguration, max_concurrent_battles=1,
                       start_timer_on_battle_start=True)
    await me.ladder(args.battles)
    print(f"done: {me.n_won_battles}-{me.n_lost_battles}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=30)
    ap.add_argument("--team", default=str(ROOT / "data/my_teams/baton.json"))
    ap.add_argument("--policy", default=str(ROOT / "checkpoints/ppo_baton_any.zip"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--local", action="store_true", help="local server (localhost:8000) against a random player")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
