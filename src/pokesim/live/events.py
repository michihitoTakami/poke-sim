"""Structured turn events (from the web UI, later from screen reading) -> Showdown protocol lines for BattleSession.

Events are dicts with "type" and "side" ("me" / "opp"):
  switch {species, hp?}      move {move}           hp {hp}         faint
  status {status}            cure                  boost {stat, amount}
  mega {species?}            item {item, consumed?} ability {ability}
  weather {weather}          field {field}          side_start / side_end {condition}
  turn                       win {winner: "me"|"opp"}
HP: our side in actual HP points, the opponent in percent (as the game shows them).
Moves, items, abilities and species are Showdown ids or names; display names are looked up for the lines.
"""
from poke_env.data import GenData

from pokesim.live.session import BattleSession
from pokesim.rl.myteam import to_id

DEX = GenData.from_gen(9)
STATUS = {"brn", "par", "slp", "frz", "psn", "tox"}


def move_name(move: str) -> str:
    entry = DEX.moves.get(to_id(move))
    return entry["name"] if entry else move


def species_name(species: str) -> str:
    entry = DEX.pokedex.get(to_id(species))
    return entry["name"] if entry else species


class EventTranslator:
    def __init__(self, session: BattleSession):
        self.s = session

    def ident(self, side: str, species: str | None = None) -> str:
        """'p1a: Floette' for our active (or named) Pokémon, 'p2a: Garchomp' for the opponent's."""
        b = self.s.battle
        if side == "me":
            if species:
                info = next(i for i in self.s.info if to_id(i["species"]) == to_id(species)
                            or to_id(i["name"]) == to_id(species))
                return f"{self.s.role}a: {info['name']}"
            active = b.active_pokemon
            info = next(i for i in self.s.info if to_id(i["species"]) == to_id(active.species)
                        or to_id(active.species).startswith(to_id(i["species"])))
            return f"{self.s.role}a: {info['name']}"
        if species:
            return f"{self.s.opp_role}a: {species_name(species).split('-')[0]}"
        opp = b.opponent_active_pokemon
        return f"{self.s.opp_role}a: {species_name(opp.species).split('-')[0]}"

    def hp(self, side: str, value: float, species: str | None = None) -> str:
        if side == "me":
            name = self.ident("me", species).split(": ", 1)[1]
            info = next(i for i in self.s.info if i["name"] == name)
            return f"{max(0, round(value))}/{info['maxhp']}" if value > 0 else "0 fnt"
        return f"{max(0, round(value))}/100" if value > 0 else "0 fnt"

    def lines(self, ev: dict) -> list[str]:
        kind, side = ev["type"], ev.get("side", "me")
        role = self.s.role if side == "me" else self.s.opp_role
        if kind == "switch":
            if side == "me":
                info = next(i for i in self.s.info if to_id(i["species"]) == to_id(ev["species"]))
                hp = ev.get("hp")
                if hp is None:
                    mon = self.s.battle.team.get(f"{role}: {info['name']}")
                    hp = round(mon.current_hp_fraction * info["maxhp"]) if mon else info["maxhp"]
                return [f"|switch|{role}a: {info['name']}|{info['details']}|{self.hp('me', hp, ev['species'])}"]
            name = species_name(ev["species"])
            hp = ev.get("hp", 100)
            return [f"|switch|{role}a: {name.split('-')[0]}|{name}, L50|{self.hp('opp', hp)}"]
        who = self.ident(side, ev.get("species"))
        target = self.ident("opp" if side == "me" else "me")
        if kind == "move":
            return [f"|move|{who}|{move_name(ev['move'])}|{target}"]
        if kind == "hp":
            return [f"|-damage|{who}|{self.hp(side, ev['hp'], ev.get('species'))}"]
        if kind == "faint":
            return [f"|-damage|{who}|0 fnt", f"|faint|{who}"]
        if kind == "status" and ev["status"] in STATUS:
            return [f"|-status|{who}|{ev['status']}"]
        if kind == "cure":
            return [f"|-curestatus|{who}|{ev.get('status', 'brn')}"]
        if kind == "boost":
            amount = int(ev["amount"])
            verb = "-boost" if amount > 0 else "-unboost"
            return [f"|{verb}|{who}|{ev['stat']}|{abs(amount)}"]
        if kind == "mega":
            mon = self.s.battle.active_pokemon if side == "me" else self.s.battle.opponent_active_pokemon
            mega = species_name(ev.get("species") or f"{species_name(mon.species)}-Mega")
            stone = ev.get("item", "")
            return [f"|detailschange|{who}|{mega}, L50", f"|-mega|{who}|{species_name(mon.species)}|{stone}"]
        if kind == "item":
            verb = "-enditem" if ev.get("consumed") else "-item"
            return [f"|{verb}|{who}|{ev['item']}"]
        if kind == "ability":
            return [f"|-ability|{who}|{ev['ability']}"]
        if kind == "weather":
            return [f"|-weather|{ev['weather']}"]
        if kind == "field":
            return [f"|-fieldstart|move: {ev['field']}"]
        if kind in ("side_start", "side_end"):
            verb = "-sidestart" if kind == "side_start" else "-sideend"
            player = f"{role}: {'me' if side == 'me' else 'opponent'}"
            return [f"|{verb}|{player}|move: {ev['condition']}"]
        if kind == "turn":
            return ["|", "|upkeep", f"|turn|{self.s.battle.turn + 1}"]
        if kind == "win":
            winner = self.s.username if ev.get("winner") == "me" else self.s.battle.opponent_username or "opponent"
            return [f"|win|{winner}"]
        raise ValueError(f"unknown event: {ev}")

    def apply(self, events: list[dict]) -> list[str]:
        out = []
        for ev in events:
            lines = self.lines(ev)
            self.s.apply(lines)
            out += lines
        return out
