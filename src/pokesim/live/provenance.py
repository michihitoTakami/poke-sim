"""Keep recommendations, submissions and protocol execution distinct, including historical unknown actors."""
from poke_env.data import GenData

from pokesim.rl.myteam import to_id


def base(name: str) -> str:
    sid = to_id(name.split(',')[0])
    entry = GenData.from_gen(9).pokedex.get(sid, {})
    return to_id(entry.get('baseSpecies', sid))


def request_lineup(request: dict) -> list[str] | None:
    mons = request.get('side', {}).get('pokemon', [])
    if request.get('teamPreview') or len(mons) != 3:
        return None
    names = [m['details'].split(',')[0] for m in mons]
    return names if len(set(map(base, names))) == 3 else None


def actual_lineup(rec: dict, *, through_turn: int | None = None) -> dict:
    """Recover own membership only. Future opponent events/sets never contribute to reconstruction.

    A private request proves membership, not original lead order (requests reorder the active first).
    Without one, all three public reveals prove membership; partial reveals remain explicitly partial.
    """
    saved = rec.get('actual_lineup')
    if saved and saved.get('complete'):
        return saved
    for entry in rec.get('advice') or []:
        lineup = request_lineup(entry.get('request') or {})
        if lineup:
            return {'species': lineup, 'complete': True, 'evidence': 'private_request', 'source': 'unknown'}
    seen = []
    for line in rec.get('log', []):
        p = line.split('|')
        if through_turn is not None and p[1:2] == ['turn'] and int(p[2]) > through_turn:
            break
        if len(p) > 3 and p[1] in {'switch', 'drag', 'replace'} and p[2].startswith('p1'):
            name = p[3].split(',')[0]
            if base(name) not in set(map(base, seen)):
                seen.append(name)
    # Illusion may leave four apparent identities; don't assert which three without private evidence.
    return {'species': seen if len(seen) <= 3 else [], 'complete': len(seen) == 3,
            'evidence': 'public_reveals' if len(seen) == 3 else 'partial_public_reveals', 'source': 'unknown'}


def reconstruction_lineup(rec: dict, turn: int | None = None) -> tuple[list[str], dict]:
    recovered = actual_lineup(rec)
    if recovered['complete']:
        return recovered['species'], recovered
    # Legacy suggestions are only a declared hypothesis while consistent with everything observed.
    suggested = rec.get('lineup') or []
    if len(suggested) == 3 and set(map(base, recovered['species'])) <= set(map(base, suggested)):
        return suggested, dict(recovered, evidence='unverified_suggestion', complete=False)
    partial = actual_lineup(rec, through_turn=turn)
    return partial['species'], partial


def execution_rows(rec: dict) -> list[dict]:
    from pokesim.learn.records import turn_actions

    out = []
    for i, line in enumerate(rec.get('log', [])):
        if not line.startswith('|turn|'):
            continue
        turn = int(line.split('|')[2])
        action = turn_actions(rec['log'], i)['p1']
        advice = next((a for a in reversed(rec.get('advice') or [])
                       if a.get('turn') == turn and not a.get('force', a.get('force_switch', False))), {})
        declaration = next((e for e in reversed(rec.get('execution_sources', []))
                            if e['turn'] == turn and e['action'] == action), {})
        submitted = advice.get('submission') or {}
        source = declaration.get('source', 'unknown')
        if (not declaration and source == 'unknown' and action and rec.get('control_mode') == 'autonomous_ai' and
                submitted.get('source') == 'ai' and submitted.get('action') == action):
            source = 'ai'
        out.append({'turn': turn, 'suggested': advice.get('suggested', advice.get('chosen')),
                    'submitted': submitted or None, 'executed': action, 'source': source,
                    'evidence': 'protocol' if action else 'no_visible_action'})
    return out
