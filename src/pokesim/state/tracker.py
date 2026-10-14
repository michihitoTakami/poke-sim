"""Move locks with public protocol evidence. No hidden timers or opponent set fields are stored."""
from poke_env.battle import Battle


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
    __slots__ = ("state_tracker",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_tracker = ProtocolTracker()
