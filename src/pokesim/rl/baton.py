"""A scripted Baton Pass plan for the fixed team (data/my_teams/baton.json), and counting what disrupted it.

Plans for the passer (Scolipede): Protect on turn 1 (a free Speed Boost), then
- pass6: Iron Defense to +4, one Substitute, Iron Defense to +6, Baton Pass (earlier when below 35% HP);
  the recipient sets up to +2, heals when low, otherwise attacks with its best move;
- sub: Iron Defense once (+2), Substitute, Baton Pass while the Substitute stands; the recipient re-Subs above
  55% HP, sets up to +6 and heals below 40%.
When Taunted or Encored, the passer switches out.
"""
from collections import Counter

from poke_env.battle import Effect, MoveCategory
from poke_env.player import Player

PASSER = "scolipede"
SETUP = {"dragondance": "atk", "nastyplot": "spa", "calmmind": "spa", "irondefense": "def"}
HEALS = {"roost", "recover", "synthesis", "moonlight", "slackoff"}
DISRUPT_MOVES = {"roar", "whirlwind", "dragontail", "circlethrow", "encore", "taunt", "haze", "clearsmog", "yawn"}


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
            effect = ev[3].replace("move: ", "").lower()
            out[effect + "ed"] += 1
            out[f"{effect}ed:{who.split(': ', 1)[-1]}"] += 1
        elif kind == "move" and mine and ev[3] == "Baton Pass":
            out["baton pass"] += 1
    return out
