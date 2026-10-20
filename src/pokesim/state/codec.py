"""Known counters only: never invent exact sleep duration or substitute HP from a public effect flag."""
from poke_env.battle import Pokemon

VERSION = 'observed-state-v6'


def known_counters(mon: Pokemon, active: bool, own: bool = False) -> dict:
    out = {}
    if active and any(e.name == 'TYPECHANGE' for e in mon.effects):
        # Keep the engine's typeless slot after Burn Up / Double Shock. It is not an unknown observation
        # or Normal, and preserving it also matters to Burn Up + Roost's grounding rule.
        out['types'] = ['???' if t.name == 'THREE_QUESTION_MARKS' else t.name.title() for t in mon.types]
    # None is an observed empty slot (consumed/removed); only unknown_item means unknown.
    # Keep current equipment separate from the original team sheet on BOTH sides.
    if mon.item != "unknown_item":
        out["item"] = mon.item or ""
    if own:
        # poke-env's standard-generation max PP differs from Champions. Transfer consumption on the bench,
        # and let the sampled simulator set supply its own maximum. The active request overrides this below.
        out["ppUsed"] = {mid: max(0, move.max_pp - move.current_pp) for mid, move in mon.moves.items()}
    if mon.status and mon.status.name == "TOX":
        out["toxicStage"] = mon.status_counter
    if active and mon.last_move is not None:
        out["lastMove"] = mon.last_move.id
        if own and mon.item in {"choiceband", "choicespecs", "choicescarf"}:
            out["choiceMove"] = mon.last_move.id
    return out


def fainted_count(battle, role: str, mons) -> int:
    """Public cumulative faints, including revivals; plain Battle callers have the visible roster fallback."""
    visible = sum(m.fainted for m in mons)
    tracker = getattr(battle, "state_tracker", None)
    return max(visible, tracker.total_fainted.get(role, 0)) if tracker else visible
