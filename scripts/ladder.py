"""Play our team on a Showdown server with the advisor (policy + lookahead), saving every battle as a record.

  # public server (needs a registered account): credentials in ~/.config/poke-sim/showdown.env
  #   PS_USERNAME=...   PS_PASSWORD=...
  uv run python scripts/ladder.py --battles 30
  # A/B: battles alternate between the policies (the record says which one played)
  # On the public server it exits between battles when the code or priors change, so a loop restarting it keeps
  # playing the latest version (see /data/poke-sim/ladder_logs/run3h.sh).
  uv run python scripts/ladder.py --battles 30 --policy checkpoints/ppo_baton_v2.zip checkpoints/ppo_baton_items.zip
  # local test against a random player
  uv run python scripts/ladder.py --local --battles 2

Team preview: the lineup ranked best by the matchup model for the opponent's 6. Each turn: the advisor's top
legal action, or with --sample an action drawn from its distribution (the seed and the chosen action's probability
are saved: a mixed strategy played as argmax is no longer mixed). --planner shadow|on runs the common-strategy
planner (pokesim.search.planner) next to / instead of the current advice. Each record says which code, policy and
lineup table played (pokesim.runtime.bundle).
Records go to data/raw/ladder/<time>.json in the format of data/raw/live/logs (opponent preview, lineup, protocol
log, result), so pokesim.live.observed and scripts/eval_records.py can use them.
Showdown is not the Switch game: mechanics follow Showdown's Champions format and opponents are Showdown players.
"""
import argparse
import asyncio
import json
import os
import random
import re
import time
import uuid
from pathlib import Path

import numpy as np
from poke_env import AccountConfiguration, LocalhostServerConfiguration, ShowdownServerConfiguration
from poke_env.environment import SinglesEnv
from poke_env.player import Player, RandomPlayer
from sb3_contrib import MaskablePPO

from pokesim.config import FORMAT, RAW_DIR, ROOT
from pokesim.live.advisor import Advisor
from pokesim.live.selection import MatchupModel, PickModel
from pokesim.rl.bc import CounterPickHeuristic
from pokesim.rl.myteam import FixedTeam, TeamFeatures, counter_preview, to_id
from pokesim.rl.opponents import LeaguePlayer
from pokesim.runtime.bundle import describe, file_hash
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
    """With several policies, battles alternate between them (A/B comparison on the same ladder)."""

    def __init__(self, sets: list[dict], policies: list[str], out: Path = OUT, combine: dict | None = None,
                 planner: dict | None = None, sample: bool = False, seed: int | None = None,
                 opponent_policy: str = "default", belief: bool = False, value_run=None, **kwargs):
        self.sets, self.out = sets, out
        self.belief = belief
        self.advisor = Advisor(sets, policy=policies[0], combine=combine, planner=planner,
                               source="showdown", sample=sample, opponent_policy=opponent_policy,
                               belief=belief, value_run=value_run)
        self.planner, self.sample = self.advisor.planner, sample
        self.seed = seed if seed is not None else random.SystemRandom().randrange(2 ** 31)
        self.rng = random.Random(self.seed)
        roster = [s["species"] for s in sets]
        self.policies = {policies[0]: (self.advisor.model, self.advisor.extra)}
        self.calibrations = {policies[0]: self.advisor.calibration}
        for path in policies[1:]:
            model = MaskablePPO.load(path, device="cpu")
            self.policies[path] = (model, TeamFeatures.for_model(model, roster))
            self.calibrations[path] = self.advisor.calibrations.get("showdown", file_hash(path),
                                                                 self.advisor.calibration_context)
        self.assigned: dict[str, str] = {}  # battle tag -> policy path
        self.advice: dict[str, list[dict]] = {}  # battle tag -> per-decision advice
        self.picks, self.matchups = PickModel(), MatchupModel()
        for path in policies:
            if warning := self.matchups.check(path):
                print(f"lineup table vs {Path(path).name}:", warning, flush=True)
        self.lineups: dict[str, list[str]] = {}
        self.logs: dict[str, list[str]] = {}  # battle tag -> protocol lines
        self.value_logged: set[str] = set()
        super().__init__(team=FixedTeam(sets), **kwargs)

    async def _handle_battle_message(self, split_messages):
        tag = split_messages[0][0].lstrip(">").strip() if split_messages and split_messages[0] else ""
        if tag:
            self.logs.setdefault(tag, []).extend("|".join(m) for m in split_messages[1:]
                                                 if len(m) > 1 and m[1] not in ("request", "t:", "j", "l", "c"))
        await super()._handle_battle_message(split_messages)

    def policy_for(self, battle) -> str:
        if battle.battle_tag not in self.assigned:
            paths = list(self.policies)  # alternate across restarts too: by the number of records saved so far
            done = len(list(self.out.glob("*.json"))) if self.out.exists() else 0  # one battle at a time
            self.assigned[battle.battle_tag] = paths[done % len(paths)]
        return self.assigned[battle.battle_tag]

    def teampreview(self, battle) -> str:
        self.policy_for(battle)
        if self.advisor.engine is not None and self.advisor.engine.value_model is not None:
            print(f'{battle.battle_tag}: matched; value model '
                  f'{self.advisor.engine.value_model.version["dataset_id"][:12]}', flush=True)
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

    def activate_policy(self, policy_path: str) -> None:
        """Switch both the model and its calibration together, including clearing an unsupported calibration."""
        self.advisor.model, self.advisor.extra = self.policies[policy_path]
        self.advisor.calibration = self.calibrations[policy_path]

    async def choose_move(self, battle):
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
        self.activate_policy(self.policy_for(battle))
        entry = {"turn": battle.turn, "force": bool(battle.force_switch)}
        self.advice.setdefault(battle.battle_tag, []).append(entry)
        try:
            advice = await asyncio.to_thread(self.advisor.advise, battle, self.last_opponent_move(battle),
                                              self.logs.get(battle.battle_tag, []), battle.player_role)
            value = advice.status.get('search', {}).get('value', {})
            if value.get('learned', 0) and battle.battle_tag not in self.value_logged:
                self.value_logged.add(battle.battle_tag)
                print(f'{battle.battle_tag}: value active ({value["learned"]} leaves, '
                      f'{advice.elapsed:.2f}s)', flush=True)
            entry.update({"actions": rounded(advice.actions[:5]), "win": advice.win_raw, "win_cal": advice.win_prob,
                          "policy": rounded(sorted(advice.policy.items(), key=lambda kv: -kv[1])[:3]),
                          "lookahead": [(lab, round(p, 3), round(v, 3)) for lab, p, v in advice.lookahead[:3]],
                          "opponent": rounded(advice.opponent[:4]), "secs": round(advice.elapsed, 2),
                          "status": advice.status})
            if advice.shadow:
                entry["shadow"] = [(lab, round(p, 3), round(v, 3)) for lab, p, v in advice.shadow[:5]]
            legal = [(label, p, a) for label, p in advice.actions
                     if (a := label_to_action(label, battle)) is not None and mask[a]]
            if legal:
                label, p, action = legal[0]
                if self.sample and sum(q for _, q, _ in legal) > 0:
                    label, p, action = self.rng.choices(legal, weights=[q for _, q, _ in legal])[0]
                entry.update({"chosen": label, "chosen_p": round(float(p), 4)})
                return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)
        except Exception as e:  # noqa: BLE001 - never stall a live battle
            print("advice failed:", repr(e)[:200], flush=True)
            entry["error"] = repr(e)[:200]
        entry["chosen"] = "random"
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
                  "lineup": self.lineups.get(battle.battle_tag), "log": log,
                  "advice": self.advice.pop(battle.battle_tag, []),
                  "policy": Path(self.assigned.get(battle.battle_tag, "")).stem,
                  "source": f"showdown {battle.battle_tag}", "opponent": battle.opponent_username,
                  "rating": getattr(battle, "rating", None),
                  "versions": describe(self.assigned.get(battle.battle_tag), self.matchups.meta,
                                       planner=self.planner, sample=self.sample, seed=self.seed,
                                       opponent_policy=self.advisor.opponent_policy_version, belief=self.belief,
                                       calibration_context=self.advisor.calibration_context)}
        (self.out / f"{stamp}-{battle.battle_tag.split('-')[-1]}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1))
        print(f"{battle.battle_tag}: {record['result']} vs {battle.opponent_username} [{record['policy']}]  "
              f"({self.n_won_battles}-{self.n_lost_battles})", flush=True)


def code_stamp() -> float:
    """Latest modification time of the code, the sim scripts and the priors the advisor reads."""
    paths = [*(ROOT / "src").rglob("*.py"), *(ROOT / "services" / "sim-tools").glob("*.js"), Path(__file__),
             *(ROOT / "data" / "processed").glob("*_priors*.json")]
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


def rounded(pairs) -> list:
    return [(k, round(float(v), 3)) for k, v in pairs]


async def run(args) -> None:
    sets = json.loads(Path(args.team).read_text())
    combine = json.loads(args.combine) if args.combine else None
    planner = None if args.planner == "off" else {"mode": args.planner}
    if planner and args.search:
        planner["search"] = json.loads(args.search)
    play = {"combine": combine, "planner": planner, "sample": args.sample, "seed": args.seed,
            "opponent_policy": args.opponent_policy, "belief": args.belief, "value_run": args.value_run}
    if args.local:
        tag = uuid.uuid4().hex[:6]
        me = AdvisorPlayer(sets, args.policy, out=Path(args.out), **play, battle_format=FORMAT,
                           server_configuration=LocalhostServerConfiguration, max_concurrent_battles=1,
                           account_configuration=AccountConfiguration(f"adv_{tag}", None))
        pool = TeamPool(load_teams("observed"), args.seed or 0)  # the same seed: the same opponent teams
        if args.local_opponent == "heuristic":
            foe = CounterPickHeuristic(battle_format=FORMAT, team=pool, seed=0, p_counter=0.7,
                                       server_configuration=LocalhostServerConfiguration,
                                       account_configuration=AccountConfiguration(f"opp_{tag}", None))
        elif args.local_opponent:
            foe = LeaguePlayer(battle_format=FORMAT, team=pool, snapshots=[args.local_opponent], p_heuristic=0.0,
                               server_configuration=LocalhostServerConfiguration,
                               account_configuration=AccountConfiguration(f"opp_{tag}", None))
            rng = random.Random(0)
            foe.teampreview = lambda battle: counter_preview(battle, rng, 0.7)
        else:
            foe = RandomPlayer(battle_format=FORMAT, team=pool, server_configuration=LocalhostServerConfiguration,
                               account_configuration=AccountConfiguration(f"opp_{tag}", None))
        await me.battle_against(foe, n_battles=args.battles)
        print(f"done: {me.n_won_battles}-{me.n_lost_battles} combine={combine}", flush=True)
        return
    cred = credentials()
    if cred is None:
        raise SystemExit(f"no Showdown account: put PS_USERNAME / PS_PASSWORD in {CREDENTIALS}")
    me = AdvisorPlayer(sets, args.policy, out=Path(args.out), **play, battle_format=FORMAT,
                       account_configuration=AccountConfiguration(*cred),
                       server_configuration=ShowdownServerConfiguration, max_concurrent_battles=1,
                       start_timer_on_battle_start=True)
    stamp = code_stamp()
    for _ in range(args.battles):
        await asyncio.wait_for(me.ladder(1), timeout=args.battle_timeout)
        if code_stamp() != stamp:  # code or priors changed: exit between battles so the caller restarts us
            print("code changed: exiting to reload", flush=True)
            break
    print(f"done: {me.n_won_battles}-{me.n_lost_battles}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=30)
    ap.add_argument("--team", default=str(ROOT / "data/my_teams/baton.json"))
    ap.add_argument("--opponent-policy", default="default",
                    help="cloned human policy for the opponent prediction (default: checkpoints/bc_human.zip)")
    ap.add_argument("--belief", action="store_true",
                    help="infer the opponent's sets from damage and move order (pokesim.belief) for the lookahead")
    ap.add_argument('--value-run', type=Path, help='validated terminal-value run; requires planner on/shadow')
    ap.add_argument('--battle-timeout', type=float, default=600,
                    help='restart after a stuck public battle/network connection (seconds)')
    ap.add_argument("--policy", nargs="+", default=[str(ROOT / "checkpoints/ppo_baton_v2.zip")],
                    help="one or more policies; with several, battles alternate between them")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--local", action="store_true", help="local server (localhost:8000) against a random player")
    ap.add_argument("--local-opponent", help="with --local: a saved opponent policy instead (counter-picks 70%%), "
                                              "or \"heuristic\" for the counter-picking heuristic bot")
    ap.add_argument("--combine", help='JSON overriding the advisor\'s COMBINE, e.g. \'{"mode": "linear"}\'')
    ap.add_argument("--planner", choices=["off", "shadow", "on"], default="off",
                    help="common-strategy planner: shadow records it next to the current advice, on plays it")
    ap.add_argument("--search", help='JSON overriding the planner\'s worker settings (PLANNER_SEARCH), e.g. '
                                      '\'{"continuations": ["expert/tactical", "expert/disrupt"]}\'')
    ap.add_argument("--sample", action="store_true", help="draw the action from the advice instead of its top")
    ap.add_argument("--seed", type=int, help="seed for --sample (default: random, saved in each record)")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
