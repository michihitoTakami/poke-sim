"""What a battle's protocol log shows about the opponent's sets: damage dealt and taken, and who moved first.

The log is replayed through BattleSession from our side. Before each move we take the position (both actives, HP,
boosts, status, field), then read the lines that follow until the next action:

  damage   one damaging hit of a move (no multi-hit, no substitute, no Focus Sash / Sturdy / Endure / Disguise
           survival, no damage "[from]" an item or ability): our Pokémon's HP is exact, the opponent's is a
           percentage. With the critical hit flag from "|-crit|".
  speed    both actives used a move this turn, neither switched before moving, the same priority, no Quick Claw /
           Custap Berry / Quash / Trick Room starting this turn, and the opponent's move is not a status move (Prankster
           would change its priority): who went first.

  no choice two different moves in one stay on the field (Choice items lock the first): not a Choice item.

Each observation carries the position as a lookahead sample (pokesim.search.lookahead SIDE / FIELD) with the
opponent's active first, so a candidate set for it can be put in its place (services/sim-tools/belief.js).
"""
from dataclasses import dataclass

from poke_env.data import GenData

from pokesim.live.session import BattleSession
from pokesim.rl.myteam import to_id
from pokesim.search.lookahead import field_state, mon_state, revealed, side_conditions

SURVIVAL = ("Focus Sash", "Sturdy", "Endure", "Disguise", "Ice Face", "Substitute", "Focus Band")
ORDER_EFFECTS = ("Quick Claw", "Custap Berry", "Quash", "Quick Draw", "After You")
BOUNDARY = ("move", "switch", "drag", "turn", "upkeep", "cant", "replace", "win", "tie")


@dataclass
class Observation:
    id: str  # "<turn>:<index>" unique within a battle
    kind: str  # damage | speed | nochoice
    turn: int
    species: str  # the opponent's active (base id, as the set sampler names it)
    sample: dict  # lookahead sample with the opponent's active first; its set is replaced per candidate
    move: str | None = None  # damage: the move
    attacker: str | None = None  # damage: "p1" (us) or "p2" (the opponent), in the sample's naming
    crit: bool = False
    before: float | None = None  # damage: defender HP before (exact for us, percent for the opponent)
    after: float | None = None
    opponent_first: bool | None = None  # speed
    revealed: tuple = ((), None, None)  # the opponent active's moves, item, ability known at the time

    def job(self) -> dict:
        out = {"sample": self.sample}
        if self.kind == "damage":
            out.update(attacker=self.attacker, move=self.move, crit=self.crit)
        return out


def move_entry(move_id: str) -> dict:
    return GenData.from_gen(9).moves.get(to_id(move_id), {})


def own_sample_side(battle, own_sets: dict, base_id) -> dict:
    """Our active (exact set) padded with two more of ours (the simulator needs a team; they do not act)."""
    active = battle.active_pokemon
    sets = [own_sets[base_id(to_id(active.species))]]
    sets += [s for k, s in own_sets.items() if s is not sets[0]][:2]
    state = mon_state(active, True, own=True)
    for k in ("ppUsed", "pp", "choiceMove"):  # damage and speed do not depend on them; avoid PP mismatches
        state.pop(k, None)
    return {"team": sets, "mons": [state, {"hp": 1.0}, {"hp": 1.0}], "megaUsed": battle.used_mega_evolve,
            "conditions": side_conditions(battle.side_conditions, battle.turn)}


def opponent_sample_side(battle) -> dict:
    """The opponent's active first; its set (team[0]) is filled in per candidate. Padding repeats the slot."""
    state = mon_state(battle.opponent_active_pokemon, True)
    return {"team": [None, None, None], "mons": [state, {"hp": 1.0}, {"hp": 1.0}],
            "megaUsed": battle.opponent_used_mega_evolve,
            "conditions": side_conditions(battle.opponent_side_conditions, battle.turn)}


class Observer:
    """Replays a log incrementally (feed new lines as they arrive) and collects observations."""

    def __init__(self, sets: list[dict], base_id, role: str = "p1"):
        self.sets, self.base_id, self.role = sets, base_id, role
        self.opp = "p2" if role == "p1" else "p1"
        self.own_sets = {base_id(to_id(s["species"])): s for s in sets}
        self.session: BattleSession | None = None
        self.seen = 0  # log lines consumed
        self.observations: list[Observation] = []
        self._pending: dict | None = None  # a move waiting for its result lines
        self._turn: dict | None = None  # this turn's order evidence
        self._stay: set[str] = set()  # moves the opponent's active used since it came in
        self._stay_done = False
        self._n = 0

    # -- feeding ---------------------------------------------------------------------------------------------------
    def feed(self, log: list[str]) -> list[Observation]:
        """Consume log[self.seen:]; returns the observations completed by these lines."""
        new_start = len(self.observations)
        lines = log[self.seen:]
        self.seen = len(log)
        for line in lines:
            if self.session is None:
                if line.startswith("|teampreview"):
                    self._start(log)
                continue
            self._line(line)
        return self.observations[new_start:]

    def _start(self, log: list[str]) -> None:
        self.session = BattleSession(self.sets, role=self.role)
        species = [line.split("|")[3].split(",")[0] for line in log
                   if line.startswith(f"|poke|{self.opp}|")]
        self.session.preview(species)

    def _line(self, line: str) -> None:
        parts = line.split("|")
        kind = parts[1] if len(parts) > 1 else ""
        if kind == "error":  # a server reply to a rejected choice, not a battle event
            return
        if kind in BOUNDARY:
            self._resolve_move()
        if kind == "turn":
            self._resolve_turn()
        if kind in ("switch", "drag", "replace") and len(parts) > 2 and parts[2].startswith(self.opp):
            self._stay, self._stay_done = set(), False
        if kind == "move" and "[from]" not in line and len(parts) > 3:
            if parts[2].startswith(self.opp):
                self._lock_evidence(parts)
            self._begin_move(parts)
        elif self._pending is not None:
            self._pending["lines"].append(line)
        if self._turn is not None:
            self._turn["lines"].append(line)
        self.session.apply([line])
        if kind == "turn":
            self._begin_turn()

    # -- damage ----------------------------------------------------------------------------------------------------
    def _begin_move(self, parts: list[str]) -> None:
        battle = self.session.battle
        side = parts[2][:2]
        if battle.active_pokemon is None or battle.opponent_active_pokemon is None:
            return
        move = to_id(parts[3])
        entry = move_entry(move)
        if entry.get("category") not in ("Physical", "Special") or entry.get("multihit"):
            return
        try:
            sample = self._sample()
        except (KeyError, AttributeError, ValueError):
            return
        defender = battle.opponent_active_pokemon if side == self.role else battle.active_pokemon
        before = defender.current_hp
        self._pending = {"side": side, "move": move, "sample": sample, "before": before,
                         "revealed": revealed(battle.opponent_active_pokemon),
                         "species": self.base_id(to_id(battle.opponent_active_pokemon.species)),
                         "defender": f"{self.opp if side == self.role else self.role}a: ", "lines": [],
                         "turn": battle.turn}
        if self._turn is not None:
            self._turn["moves"].append((side, move))

    def _resolve_move(self) -> None:
        p, self._pending = self._pending, None
        if p is None:
            return
        lines = p["lines"]
        if any(x.startswith("|-hitcount|") for x in lines) or any(s in x for x in lines for s in SURVIVAL):
            return
        hits = [x for x in lines if x.startswith(f"|-damage|{p['defender']}") and "[from]" not in x]
        if len(hits) != 1:
            return
        hp = hits[0].split("|")[3].split()[0]
        after = 0.0 if hp == "0" else float(hp.split("/")[0])
        if p["before"] is None or after > p["before"]:
            return
        self._n += 1
        self.observations.append(Observation(
            f"{p['turn']}:{self._n}", "damage", p["turn"], p["species"], p["sample"], move=p["move"],
            attacker="p1" if p["side"] == self.role else "p2",
            crit=any(x.startswith(f"|-crit|{p['defender']}") for x in lines), before=float(p["before"]),
            after=after, revealed=p["revealed"]))

    # -- Choice items ----------------------------------------------------------------------------------------------
    def _lock_evidence(self, parts: list[str]) -> None:
        battle = self.session.battle
        move = to_id(parts[3])
        if move == "struggle" or battle.opponent_active_pokemon is None:
            return
        self._stay.add(move)
        if len(self._stay) >= 2 and not self._stay_done:
            self._stay_done = True
            opp = battle.opponent_active_pokemon
            self._n += 1
            self.observations.append(Observation(f"{battle.turn}:{self._n}", "nochoice", battle.turn,
                                                 self.base_id(to_id(opp.species)), {}, revealed=revealed(opp)))

    # -- speed -----------------------------------------------------------------------------------------------------
    def _begin_turn(self) -> None:
        battle = self.session.battle
        self._turn = None
        if battle.active_pokemon is None or battle.opponent_active_pokemon is None:
            return
        try:
            sample = self._sample()
        except (KeyError, AttributeError, ValueError):
            return
        self._turn = {"sample": sample, "moves": [], "lines": [], "turn": battle.turn,
                      "revealed": revealed(battle.opponent_active_pokemon),
                      "species": self.base_id(to_id(battle.opponent_active_pokemon.species))}

    def _resolve_turn(self) -> None:
        t, self._turn = self._turn, None
        if t is None:
            return
        lines = t["lines"]
        first_moves: dict[str, tuple[int, str]] = {}
        for i, x in enumerate(lines):
            parts = x.split("|")
            if len(parts) < 4:
                continue
            side = parts[2][:2]
            if parts[1] in ("switch", "drag", "replace") and side not in first_moves:
                return  # someone switched before both moved: not a speed comparison
            if parts[1] == "move" and "[from]" not in x and side not in first_moves:
                first_moves[side] = (i, to_id(parts[3]))
        if set(first_moves) != {self.role, self.opp}:
            return
        if any(s in x for x in lines for s in ORDER_EFFECTS) or any(
                x.startswith("|-fieldstart|") and "Trick Room" in x for x in lines):
            return
        ours, theirs = move_entry(first_moves[self.role][1]), move_entry(first_moves[self.opp][1])
        if not ours or not theirs or theirs.get("category") == "Status" or \
                ours.get("priority", 0) != theirs.get("priority", 0):
            return
        self._n += 1
        self.observations.append(Observation(
            f"{t['turn']}:{self._n}", "speed", t["turn"], t["species"], t["sample"],
            opponent_first=first_moves[self.opp][0] < first_moves[self.role][0], revealed=t["revealed"]))

    def _sample(self) -> dict:
        battle = self.session.battle
        return {"p1": own_sample_side(battle, self.own_sets, self.base_id), "p2": opponent_sample_side(battle),
                "field": field_state(battle)}
