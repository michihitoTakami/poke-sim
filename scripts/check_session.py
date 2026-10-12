"""Check that BattleSession rebuilds the same battle state from protocol lines as a real Showdown battle room.

Our player (the heuristic bot with my team) plays on the local server; every message it receives is also fed to a
BattleSession, and at each decision the session's state (HP, status, boosts, available moves/switches, mega, revealed
opponent info, field) is compared with the real poke-env battle.

With --advise, the Advisor (policy + lookahead) also runs on the rebuilt battle at every decision: its time,
errors, and whether the policy's top action matches the one computed on the real battle.

Usage: uv run python scripts/check_session.py --battles 30 [--advise checkpoints/ppo_baton3_seg003.zip]
"""
import argparse
import asyncio
import time
import uuid
from collections import Counter

from poke_env import AccountConfiguration
from poke_env.player import SimpleHeuristicsPlayer

from pokesim.config import FORMAT, ROOT
from pokesim.live.advisor import Advisor
from pokesim.live.session import BattleSession
from pokesim.rl.myteam import FixedTeam, load_my_team
from pokesim.sim.pool import TeamPool, load_teams


def snapshot(b) -> dict:
    def mons(team):
        return sorted((m.species, round(m.current_hp_fraction, 2), m.fainted, m.status.name if m.status else None,
                       tuple(sorted(m.moves)) if team is b.opponent_team else ()) for m in team.values())
    me, opp = b.active_pokemon, b.opponent_active_pokemon
    return {
        "active": (me.species if me else None, opp.species if opp else None),
        "team": mons(b.team),
        "opponent": mons(b.opponent_team),
        "boosts": (dict(me.boosts) if me else None, dict(opp.boosts) if opp else None),
        "moves": sorted(m.id for m in b.available_moves),
        "switches": sorted(m.species for m in b.available_switches),
        "mega": b.can_mega_evolve,
        "field": (sorted(map(str, b.weather)), sorted(map(str, b.fields)), sorted(map(str, b.side_conditions)),
                  sorted(map(str, b.opponent_side_conditions))),
    }


class Mirror(SimpleHeuristicsPlayer):
    def __init__(self, *args, sets: list[dict], advisor: Advisor | None = None, **kwargs):
        self.sets, self.advisor = sets, advisor
        self.times, self.errors, self.policy_agree, self.shown = [], 0, [], 0
        self.buffer: dict[str, list[str]] = {}
        self.sessions: dict[str, BattleSession] = {}
        self.checks, self.mismatch, self.examples = 0, Counter(), []
        super().__init__(*args, **kwargs)

    async def _handle_battle_message(self, split_messages):
        tag = split_messages[0][0].lstrip(">") if split_messages and split_messages[0] else ""
        lines = ["|".join(m) for m in split_messages[1:] if len(m) > 1]
        if tag in self.sessions:
            self.sessions[tag].apply(lines)
        else:
            self.buffer.setdefault(tag, []).extend(lines)
        await super()._handle_battle_message(split_messages)

    def teampreview(self, battle):
        session = BattleSession(self.sets, username=self.username, opponent=battle.opponent_username or "opp",
                                role=battle.player_role)
        session.apply(self.buffer.pop(battle.battle_tag, []))
        session.battle.parse_request(session.request(team_preview=True))
        session.select([0, 1, 2])
        self.sessions[battle.battle_tag] = session
        return "/team 123"

    def choose_move(self, battle):
        session = self.sessions.get(battle.battle_tag)
        if session is not None:
            real, rebuilt = snapshot(battle), snapshot(session.decision(force_switch=battle.force_switch))
            self.checks += 1
            if self.advisor:
                self.run_advisor(battle, session)
            for key in real:
                if real[key] != rebuilt[key]:
                    self.mismatch[key] += 1
                    if len(self.examples) < 8:
                        self.examples.append((battle.turn, key, real[key], rebuilt[key],
                                              "\n      ".join(session.log[-12:])))
        return super().choose_move(battle)


def _advise(self, battle, session):
    start = time.perf_counter()
    try:
        sug = self.advisor.advise(session.battle)
    except Exception as e:  # noqa: BLE001 - count and keep going
        self.errors += 1
        print("advisor failed:", repr(e)[:200])
        return
    self.times.append(time.perf_counter() - start)
    real_policy, _ = self.advisor._policy(battle)
    if real_policy and sug.policy:
        self.policy_agree.append(max(real_policy, key=real_policy.get) == max(sug.policy, key=sug.policy.get))
    if self.shown < 6:
        self.shown += 1
        win = f"{sug.win_prob:.0%}" if sug.win_prob is not None else "-"
        print(f"T{battle.turn} {battle.active_pokemon.species} vs {battle.opponent_active_pokemon.species}: "
              + " / ".join(f"{a} {p:.0%}" for a, p in sug.actions[:3]) + f"  win~{win}  opp: "
              + " / ".join(f"{a} {p:.0%}" for a, p in sug.opponent[:3]) + f"  ({sug.elapsed:.1f}s)")


Mirror.run_advisor = _advise


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battles", type=int, default=30)
    parser.add_argument("--team", default=str(ROOT / "data/my_teams/baton.json"))
    parser.add_argument("--advise", help="policy checkpoint; also run the Advisor on the rebuilt battle")
    args = parser.parse_args()
    sets = load_my_team(args.team, ["Scolipede", "Floette-Eternal", "Gholdengo"])
    teams = load_teams("all")
    tag = uuid.uuid4().hex[:6]
    advisor = Advisor(sets, policy=args.advise) if args.advise else None
    me = Mirror(battle_format=FORMAT, team=FixedTeam(sets), sets=sets, advisor=advisor,
                account_configuration=AccountConfiguration(f"mirror_{tag}", None))
    opp = SimpleHeuristicsPlayer(battle_format=FORMAT, team=TeamPool(teams, 3),
                                 account_configuration=AccountConfiguration(f"mopp_{tag}", None))
    await me.battle_against(opp, n_battles=args.battles)
    print(f"{me.checks} decisions checked in {len(me.battles)} battles; mismatches: {dict(me.mismatch)}")
    if advisor:
        t = sorted(me.times)
        print(f"advisor: {len(t)} runs, {me.errors} errors, mean {sum(t) / max(len(t), 1):.2f}s, "
              f"max {t[-1] if t else 0:.2f}s; policy top action same as on the real battle: "
              f"{sum(me.policy_agree)}/{len(me.policy_agree)}")
        advisor.close()
    for ex in me.examples:
        print("  turn %s %s\n    real:    %s\n    rebuilt: %s\n    log:\n      %s" % ex)


if __name__ == "__main__":
    asyncio.run(main())
