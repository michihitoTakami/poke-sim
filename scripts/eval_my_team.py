"""Play a scripted Baton Pass plan with my team against the team pool and count what disrupted it.

Plan: the passer (Scolipede) uses Protect on turn 1 (a free Speed Boost), Iron Defense to +4, one Substitute,
Iron Defense to +6, then Baton Pass (earlier when below 35% HP); the recipient sets up to +2, heals when low,
otherwise attacks with its best move.
Opponents are the lookahead player (it will phaze/Encore when that scores better) or the heuristic bot.

Usage: uv run python scripts/eval_my_team.py --lineup Scolipede Gholdengo Salamence --battles 60 [--opponent lookahead]
"""
import argparse
import asyncio
import json
import uuid
from collections import Counter

from poke_env import AccountConfiguration
from poke_env.battle import Effect, MoveCategory
from poke_env.player import Player, SimpleHeuristicsPlayer
from poke_env.teambuilder import Teambuilder

from pokesim.config import FORMAT, ROOT
from pokesim.search.player import LookaheadPlayer
from pokesim.sim.pool import TeamPool, load_teams

PASSER = "scolipede"
SETUP = {"dragondance": "atk", "nastyplot": "spa", "calmmind": "spa", "irondefense": "def"}
HEALS = {"roost", "recover", "synthesis", "moonlight", "slackoff"}
DISRUPT_MOVES = {"roar", "whirlwind", "dragontail", "circlethrow", "encore", "taunt", "haze", "clearsmog", "yawn"}


class FixedTeam(Teambuilder):
    def __init__(self, sets: list[dict]):
        self.packed = self.join_team(self.parse_showdown_team(self.to_showdown(sets)))

    @staticmethod
    def to_showdown(sets: list[dict]) -> str:
        blocks = []
        for s in sets:
            evs = " / ".join(f"{v} {k.upper() if k != 'spa' else 'SpA'}" for k, v in s["evs"].items() if v)
            evs = evs.replace("SPD", "SpD").replace("SPE", "Spe").replace("ATK", "Atk").replace("DEF", "Def")
            blocks.append(f"{s['species']} @ {s['item']}\nAbility: {s['ability']}\nLevel: 50\nEVs: {evs}\n"
                          f"{s['nature']} Nature\n" + "\n".join(f"- {m}" for m in s["moves"]))
        return "\n\n".join(blocks)

    def yield_team(self) -> str:
        return self.packed


class BatonScript(Player):
    def __init__(self, *args, lineup_size: int = 3, plan: str = "pass6", pass_to: str | None = None, **kwargs):
        self.plan, self.pass_to = plan, pass_to
        self.lineup_size = lineup_size
        self.logs: dict[str, list[list[str]]] = {}
        self.subbed: dict[str, set[int]] = {}
        super().__init__(*args, **kwargs)

    async def _handle_battle_message(self, split_messages):
        tag = split_messages[0][0].lstrip(">") if split_messages and split_messages[0] else ""
        self.logs.setdefault(tag, []).extend(split_messages[1:])
        await super()._handle_battle_message(split_messages)

    def teampreview(self, battle):
        return "/team " + "".join(str(i + 1) for i in range(self.lineup_size))

    @staticmethod
    def power(move, attacker, defender, boosts) -> float:
        if move.category == MoveCategory.STATUS or not move.base_power:
            return 0.0
        stat = "atk" if move.category == MoveCategory.PHYSICAL else "spa"
        mult = (2 + max(boosts.get(stat, 0), 0)) / (2 - min(boosts.get(stat, 0), 0))
        stab = 1.5 if move.type in attacker.types else 1.0
        return move.base_power * stab * defender.damage_multiplier(move) * mult * (move.accuracy or 1)

    def recipient(self, battle):
        opp = battle.opponent_active_pokemon
        options = battle.available_switches
        if not options:
            return None
        preferred = [m for m in options if self.pass_to and m.species == self.pass_to.lower().replace("-", "")]
        if preferred:
            return preferred[0]
        return max(options, key=lambda m: max((self.power(mv, m, opp, battle.active_pokemon.boosts
                                                          if battle.active_pokemon else {})
                                               for mv in m.moves.values()), default=0) if opp else 0)

    def choose_move(self, battle):
        if battle.force_switch or not battle.active_pokemon:
            target = self.recipient(battle)
            return self.create_order(target) if target else self.choose_random_move(battle)
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        moves = {m.id: m for m in battle.available_moves}
        mega = battle.can_mega_evolve
        if me.species == PASSER:
            if battle.turn == 1 and "protect" in moves:
                return self.create_order(moves["protect"])
            if not moves.keys() & {"irondefense", "substitute", "batonpass"} or Effect.ENCORE in me.effects:
                target = self.recipient(battle)  # Taunted / Encored: get out (boosts are lost)
                return self.create_order(target) if target else self.choose_random_move(battle)
            if me.current_hp_fraction < 0.35 and "batonpass" in moves:
                return self.create_order(moves["batonpass"])
            first_def = 2 if self.plan == "sub" else 4
            if me.boosts["def"] < first_def and "irondefense" in moves:
                return self.create_order(moves["irondefense"])
            subbed = self.subbed.setdefault(battle.battle_tag, set())
            if (Effect.SUBSTITUTE not in me.effects and me.current_hp_fraction > 0.5 and "substitute" in moves
                    and battle.turn not in subbed and not subbed):
                subbed.add(battle.turn)  # one Substitute per game, to block Dragon Tail / Yawn while passing
                return self.create_order(moves["substitute"])
            if self.plan == "sub" and Effect.SUBSTITUTE in me.effects and "batonpass" in moves:
                return self.create_order(moves["batonpass"])  # pass the Substitute along with the boosts
            if me.boosts["def"] < 6 and "irondefense" in moves:
                return self.create_order(moves["irondefense"])
            if "batonpass" in moves:
                return self.create_order(moves["batonpass"])
            return self.choose_random_move(battle)
        heal = next((m for m in moves.values() if m.id in HEALS), None)
        if heal and me.current_hp_fraction < 0.4:
            return self.create_order(heal, mega=mega)
        setup = next((m for m in moves.values() if m.id in SETUP and m.id != "irondefense"), None)
        setup_to = 6 if self.plan == "sub" else 2
        if "substitute" in moves and Effect.SUBSTITUTE not in me.effects and \
                me.current_hp_fraction > (0.55 if self.plan == "sub" else 0.7):
            return self.create_order(moves["substitute"], mega=mega)
        if setup and me.boosts[SETUP[setup.id]] < setup_to and me.current_hp_fraction > 0.5:
            return self.create_order(setup, mega=mega)
        attacks = [m for m in moves.values() if self.power(m, me, opp, me.boosts) > 0] if opp else []
        if attacks:
            return self.create_order(max(attacks, key=lambda m: self.power(m, me, opp, me.boosts)), mega=mega)
        return self.choose_random_move(battle)


def disruptions(log: list[list[str]], my_role: str) -> Counter:
    out = Counter()
    for ev in log:
        if len(ev) >= 2 and ev[1] == "-clearallboost":
            out["haze"] += 1
            continue
        if len(ev) < 3:
            continue
        kind, who = ev[1], ev[2]
        mine = who.startswith(my_role)
        if kind == "move" and not mine:
            mid = ev[3].lower().replace(" ", "").replace("-", "")
            if mid in DISRUPT_MOVES:
                out[f"used:{mid}"] += 1
        elif kind == "drag" and mine:
            out["dragged out"] += 1
        elif kind == "-start" and mine and len(ev) > 3 and ev[3].replace("move: ", "") in ("Encore", "Taunt"):
            out[ev[3].replace("move: ", "").lower() + "ed"] += 1
        elif kind == "move" and mine and ev[3] == "Baton Pass":
            out["baton pass"] += 1
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default=str(ROOT / "data/my_teams/baton.json"))
    parser.add_argument("--lineup", nargs=3, default=["Scolipede", "Gholdengo", "Salamence"])
    parser.add_argument("--battles", type=int, default=60)
    parser.add_argument("--opponent", default="lookahead", choices=["lookahead", "heuristic"])
    parser.add_argument("--pool", default="all")
    parser.add_argument("--plan", default="pass6", choices=["pass6", "sub"],
                        help="pass6: Iron Defense to +6 then pass; sub: Iron Defense once + Substitute, pass both, "
                             "recipient sets up to +6 behind Substitute")
    parser.add_argument("--pass-to", help="preferred Baton Pass recipient (species name)")
    args = parser.parse_args()
    sets = json.load(open(args.team))
    order = [s for name in args.lineup for s in sets if s["species"] == name]
    order += [s for s in sets if s not in order]
    teams = load_teams(args.pool)
    tag = uuid.uuid4().hex[:6]
    me = BatonScript(battle_format=FORMAT, team=FixedTeam(order), plan=args.plan, pass_to=args.pass_to,
                     account_configuration=AccountConfiguration(f"baton_{tag}", None))
    if args.opponent == "lookahead":
        opp = LookaheadPlayer(battle_format=FORMAT, team=TeamPool(teams, 7), teams=teams, seasons=["M-6"],
                              account_configuration=AccountConfiguration(f"opp_{tag}", None))
    else:
        opp = SimpleHeuristicsPlayer(battle_format=FORMAT, team=TeamPool(teams, 7),
                                     account_configuration=AccountConfiguration(f"opp_{tag}", None))
    await me.battle_against(opp, n_battles=args.battles)

    by_case: dict[str, list[bool]] = {}
    total = Counter()
    for battle in me.battles.values():
        role = battle.player_role
        d = disruptions(me.logs.get(battle.battle_tag, []), role)
        total.update(d)
        lost_boosts = d["dragged out"] or d["haze"] or d["encoreed"] or d["taunted"]
        by_case.setdefault("disrupted" if lost_boosts else "clean", []).append(battle.won)
    n = len(me.battles)
    won = me.n_won_battles
    print(f"[{args.plan}] {' / '.join(args.lineup)} vs {args.opponent}: {won}/{n} ({won / n:.0%})")
    for case, results in by_case.items():
        print(f"  {case:9s}: {sum(results)}/{len(results)} won ({sum(results) / len(results):.0%})")
    print("  events:", dict(total.most_common()))
    if args.opponent == "lookahead":
        opp.engine.close()


if __name__ == "__main__":
    asyncio.run(main())
