"""Protocol state shared by live requests, replay reconstruction and opponent beliefs."""
import json
import subprocess
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

    def apply(self, parts: list[str]) -> None:
        if len(parts) < 3 or ": " not in parts[2]:
            return
        kind, ident = parts[1], parts[2]
        role, name = ident[:2], ident.split(": ", 1)[1]
        key = (role, name)
        if kind in {"switch", "drag"}:
            previous = self.active.get(role)
            if previous:
                self.states.pop((role, previous), None)
            self.states.pop(key, None)
            self.active[role] = name
        state = self.states.setdefault(key, {})
        if kind == "move" and len(parts) > 3 and not any(p.startswith("[from]") for p in parts[4:]):
            state["lastMove"] = move_id(parts[3])
        if kind == "-start" and len(parts) > 3:
            if parts[3] == "Disable" and len(parts) > 4:
                state["disabledMove"] = move_id(parts[4])
            elif parts[3] == "Encore" and state.get("lastMove"):
                state["encore"] = state["lastMove"]
        if kind == "-end" and len(parts) > 3:
            state.pop({"Disable": "disabledMove", "Encore": "encore"}.get(parts[3], ""), None)

    def locks(self, role: str, name: str) -> dict:
        return {k: v for k, v in self.states.get((role, name), {}).items() if k in {"disabledMove", "encore"}}


class TrackedBattle(Battle):
    __slots__ = ("state_tracker", "_pending_baton")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_tracker = ProtocolTracker()
        self._pending_baton: dict[str, str] = {}

    def parse_message(self, split_message: list[str]) -> None:
        parts = split_message
        kind = parts[1] if len(parts) > 1 else ''
        ident = parts[2] if len(parts) > 2 else ''
        role = ident[:2]
        outgoing = self.active_pokemon if role == self.player_role else self.opponent_active_pokemon
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
        super().parse_message(parts)
        if passed is not None:
            incoming = self.active_pokemon if role == self.player_role else self.opponent_active_pokemon
            incoming.boosts.update(passed[0])
            incoming.effects.update(passed[1])
