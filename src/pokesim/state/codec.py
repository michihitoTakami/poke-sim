"""Known counters only: never invent exact sleep duration or substitute HP from a public effect flag."""
from poke_env.battle import Pokemon


def known_counters(mon: Pokemon, active: bool, own: bool = False) -> dict:
    out = {}
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
