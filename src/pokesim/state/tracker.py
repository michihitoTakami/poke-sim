"""Protocol state shared by live requests, replay reconstruction and opponent beliefs."""
import json
import subprocess
from copy import deepcopy
from functools import cache

from poke_env.battle import Battle, Effect

from pokesim.config import FORMAT, SHOWDOWN_DIR


@cache
def baton_effects() -> frozenset[Effect]:
    """Use the deployed format's copy rules, including effect aliases used by poke-env."""
    ids = {e.name: 'perishsong' if e.name.startswith('PERISH') else move_id(e.name) for e in Effect}
    code = """
const fs = require('fs');
const {Dex} = require(process.argv[1] + '/dist/sim');
const dex = Dex.forFormat(process.argv[2]);
const ids = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify(Object.keys(ids).filter(name => {
  const condition = dex.conditions.get(ids[name]);
  return condition.exists && !condition.noCopy;
})));
"""
    result = subprocess.run(['node', '-e', code, str(SHOWDOWN_DIR), FORMAT], input=json.dumps(ids),
                            capture_output=True, text=True, check=True)
    return frozenset(Effect[name] for name in json.loads(result.stdout))


def move_id(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


class ProtocolTracker:
    def __init__(self):
        self.active = {}
        self.states = {}
        self.total_fainted: dict[str, int] = {}
        self.turn = 0

    def apply(self, parts: list[str]) -> None:
        if parts[1:2] == ['turn']:
            self.turn = int(parts[2])
        if len(parts) < 3 or ": " not in parts[2]:
            return
        kind, ident = parts[1], parts[2]
        role, name = ident[:2], ident.split(": ", 1)[1]
        key = (role, name)
        if kind == "faint":
            self.total_fainted[role] = min(100, self.total_fainted.get(role, 0) + 1)
        if kind in {"switch", "drag"}:
            previous = self.active.get(role)
            if previous:
                self.states.pop((role, previous), None)
            self.states.pop(key, None)
            self.active[role] = name
        elif kind == 'replace':
            previous = self.active.get(role)
            if previous and previous != name:
                self.states[key] = self.states.pop((role, previous), {})
            self.active[role] = name
        state = self.states.setdefault(key, {})
        if kind == "move" and len(parts) > 3 and not any(p.startswith("[from]") for p in parts[4:]):
            state["lastMove"] = move_id(parts[3])
        if kind == "-start" and len(parts) > 3:
            effect = move_id(parts[3].removeprefix('move: '))
            if effect == 'yawn':
                state['yawnTurn'] = self.turn
            if effect.startswith('fallen') and effect[6:].isdigit():
                state['fallen'] = int(effect[6:])
                self.total_fainted[role] = max(self.total_fainted.get(role, 0), state['fallen'])
            if parts[3] == "Disable" and len(parts) > 4:
                state["disabledMove"] = move_id(parts[4])
            elif parts[3] == "Encore" and state.get("lastMove"):
                state["encore"] = state["lastMove"]
        if kind == "-end" and len(parts) > 3:
            if move_id(parts[3].removeprefix('move: ')) == 'yawn':
                state.pop('yawnTurn', None)
            state.pop({"Disable": "disabledMove", "Encore": "encore"}.get(parts[3], ""), None)

    def locks(self, role: str, name: str) -> dict:
        state = self.states.get((role, name), {})
        out = {k: v for k, v in state.items() if k in {"disabledMove", "encore", "fallen", "transform"}}
        if 'yawnTurn' in state:
            out['volatileTurns'] = {'yawn': max(1, 2 - (self.turn - state['yawnTurn']))}
        return out


class TrackedBattle(Battle):
    __slots__ = ("state_tracker", "_pending_baton", "_illusion_entry")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_tracker = ProtocolTracker()
        self._pending_baton: dict[str, str] = {}
        self._illusion_entry: dict[str, dict] = {}

    def parse_message(self, split_message: list[str]) -> None:
        parts = split_message
        kind = parts[1] if len(parts) > 1 else ''
        ident = parts[2] if len(parts) > 2 else ''
        role = ident[:2]
        outgoing = self.active_pokemon if role == self.player_role else self.opponent_active_pokemon
        table = self.team if role == self.player_role else self.opponent_team
        if kind in {'switch', 'drag'} and ': ' in ident:
            key = f'{role}: {ident.split(": ", 1)[1]}'
            # If this entry later proves to be an Illusion, restore what was known about the imitated species
            # before it entered. It may really have appeared earlier; do not erase its genuine history.
            self._illusion_entry[role] = {'key': key, 'before': deepcopy(table.get(key)),
                                          'moves': {}, 'item': False, 'ability': False}
        entry = self._illusion_entry.get(role)
        illusion = None
        if kind == 'replace' and outgoing is not None and entry:
            illusion = {'boosts': dict(outgoing.boosts), 'effects': dict(outgoing.effects),
                        'item': outgoing.item, 'ability': outgoing.ability, 'last_move': outgoing.last_move,
                        'protect_counter': outgoing.protect_counter}
        passed = None
        if kind in {'switch', 'drag'}:
            pending = self._pending_baton.pop(role, None)
            explicit = any(p.lower() == '[from] baton pass' for p in parts[5:])
            if (kind == 'switch' and outgoing is not None and not outgoing.fainted
                    and (explicit or pending == outgoing.name)):
                passed = (dict(outgoing.boosts), {e: n for e, n in outgoing.effects.items() if e in baton_effects()})
        elif kind == 'move':
            if len(parts) > 3 and move_id(parts[3]) == 'batonpass':
                self._pending_baton[role] = ident.split(': ', 1)[-1]
            else:
                self._pending_baton.pop(role, None)
        elif kind in {'-fail', 'cant', 'faint'}:
            self._pending_baton.pop(role, None)
        elif kind == 'turn':
            self._pending_baton.clear()

        self.state_tracker.apply(parts)
        if kind == '-transform' and len(parts) > 3:
            from pokesim.sim.formes import current_forme

            target = self.active_pokemon if parts[3][:2] == self.player_role else self.opponent_active_pokemon
            if target is not None:
                name = ident.split(': ', 1)[1]
                self.state_tracker.states[(role, name)]['transform'] = {
                    'sourceSide': parts[3][:2], 'sourceSpecies': target.species, 'species': current_forme(target),
                    'moves': list(target.moves), 'ability': target.ability,
                    'stats': {k: v for k, v in target.stats.items() if v is not None and k != 'hp'}}
        # Showdown sends |-mega|ident|base species|stone. poke-env reads the fourth field as the stone.
        # Preserve the raw protocol in the caller's log; only adapt the parser input here.
        parsed = parts[:3] + parts[4:] if kind == '-mega' and len(parts) > 4 else parts
        super().parse_message(parsed)
        current = self.active_pokemon if role == self.player_role else self.opponent_active_pokemon
        if kind in {'detailschange', '-formechange', '-mega'} and current is not None:
            # The engine's setSpecies/formeChange resets Soak. poke-env leaves _temporary_types behind.
            current._temporary_types = []
            current.effects.pop(Effect.TYPECHANGE, None)
        if entry and kind == 'move' and len(parts) > 3 and current is not None:
            mid = move_id(parts[3])
            if mid in current.moves:
                entry['moves'][mid] = deepcopy(current.moves[mid])
        elif entry and kind in {'-item', '-enditem'}:
            entry['item'] = True
        elif entry and kind == '-ability':
            entry['ability'] = True
        if illusion is not None and current is not None and current is not outgoing:
            current.moves.update(entry['moves'])
            current.boosts.update(illusion['boosts'])
            current.effects.update(illusion['effects'])
            last = illusion['last_move']
            for mid, move in current.moves.items():
                move._is_last_used = last is not None and mid == last.id
            if entry['item']:
                current.item = illusion['item']
            if entry['ability']:
                current.ability = illusion['ability']
            current._protect_counter = illusion['protect_counter']
            real_key = f'{role}: {current.name}'
            if entry['key'] != real_key:
                if entry['before'] is None:
                    table.pop(entry['key'], None)
                else:
                    table[entry['key']] = entry['before']
            self._illusion_entry.pop(role, None)
        if passed is not None:
            incoming = self.active_pokemon if role == self.player_role else self.opponent_active_pokemon
            incoming.boosts.update(passed[0])
            incoming.effects.update(passed[1])
