"""What each of our Pokémon does for the team, and whether we still have a way to win in a position.

roles(sets): from the team file, per Pokémon: attacker (has a damaging move), passer (Baton Pass), setup (raises its
own stats), recovery, protect. A role is a capability, not a fixed plan: Scolipede is the passer, but who is the
"last attacker" depends on the position.

win_path(battle): our remaining damage sources against what the opponent has left (revealed Pokémon that have not
fainted; members still unseen are unknown). A source is a Pokémon of ours, not fainted, with a damaging move with
PP left that is not immune by type (or a revealed Levitate / Air Balloon) for at least one remaining foe. It flags
  no_source       we cannot damage any remaining foe (the lookahead penalises this; it is not a loss: Struggle,
                  residual damage, Perish Song or the timer can still decide it)
  last_source     one source left: losing it leaves only Pokémon that cannot win on their own
These are warnings for the advice and diagnostics for losses, not rules: sacrificing the last attacker can be right.
"""
from dataclasses import dataclass, field

from poke_env.battle import Battle, Pokemon, PokemonType
from poke_env.data import GenData

from pokesim.rl.myteam import to_id


def move_data(move_id: str) -> dict:
    return GenData.from_gen(9).moves.get(to_id(move_id), {})


def roles(sets: list[dict]) -> dict[str, set[str]]:
    out = {}
    for s in sets:
        tags = set()
        for name in s["moves"]:
            m = move_data(name)
            if m.get("category") in ("Physical", "Special"):
                tags.add("attacker")
            if m.get("selfSwitch") == "copyvolatile":
                tags.add("passer")
            if m.get("target") == "self" and any(v > 0 for v in (m.get("boosts") or {}).values()):
                tags.add("setup")
            if (m.get("heal") or (m.get("flags") or {}).get("heal")) and m.get("target") == "self":
                tags.add("recovery")
            if to_id(name) in ("protect", "detect", "spikyshield", "banefulbunker", "kingsshield", "silktrap"):
                tags.add("protect")
        out[to_id(s["species"])] = tags
    return out


def hits(move_id: str, defender: Pokemon) -> bool:
    """Whether a damaging move can hurt the defender as far as we know (type chart, revealed Levitate / Balloon)."""
    m = move_data(move_id)
    if m.get("category") not in ("Physical", "Special"):
        return False
    mtype = (m.get("type") or "").lower()
    if mtype == "ground" and (defender.ability == "levitate" or defender.item == "airballoon"):
        return False
    try:
        return defender.damage_multiplier(PokemonType.from_name(mtype)) > 0
    except (ValueError, KeyError):
        return True


@dataclass
class WinPath:
    sources: list[str]  # our Pokémon that can damage at least one remaining foe
    blocked: dict[str, list[str]] = field(default_factory=dict)  # remaining foe -> our Pokémon that cannot hurt it
    unseen: int = 0  # opponent members not revealed yet (unknown, not counted as blocked)

    @property
    def no_source(self) -> bool:
        return not self.sources

    @property
    def last_source(self) -> bool:
        return len(self.sources) == 1

    def as_dict(self) -> dict:
        return {"sources": self.sources, "blocked": self.blocked, "unseen": self.unseen,
                "no_source": self.no_source, "last_source": self.last_source}


def win_path(battle: Battle) -> WinPath:
    # battle.team can still list all six from team preview: the brought ones are the active and the switches
    # (a trapped active has no switches listed; its bench is then not counted)
    ours = [m for m in battle.team.values() if not m.fainted]
    if len(battle.team) > 3:
        ours = [m for m in ours if m.active or m in battle.available_switches]
    foes = [m for m in battle.opponent_team.values() if not m.fainted]
    revealed = len(battle.opponent_team)
    unseen = max(0, min(3, len(battle.teampreview_opponent_team) or 3) - revealed)
    sources, blocked = [], {}
    for mon in ours:
        usable = [mid for mid, mv in mon.moves.items() if (mv.current_pp is None or mv.current_pp > 0)]
        if any(hits(mid, foe) for mid in usable for foe in foes) or (unseen and any(
                move_data(mid).get("category") in ("Physical", "Special") for mid in usable)):
            sources.append(mon.species)
    for foe in foes:
        cannot = [mon.species for mon in ours
                  if not any(hits(mid, foe) for mid, mv in mon.moves.items()
                             if mv.current_pp is None or mv.current_pp > 0)]
        if cannot:
            blocked[foe.species] = cannot
    return WinPath(sources, blocked, unseen)
