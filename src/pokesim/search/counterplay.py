"""Reject a sampled immediate sacrifice dominated by another legal root action.

Surviving preparation is deliberately incomparable with an attack's immediate damage.
Boosts, shields and safe relays keep their long-term search value.
"""
import time

VERSION = 'paired-root-counterplay-v1'


def dominates(report, alternative, action):
    if not report.get('complete') or alternative == action or not report.get('cells'):
        return False
    improved = False
    for cell in report['cells'].values():
        old, new = cell.get(action), cell.get(alternative)
        if old is None or new is None:
            return False
        # Never exchange an extra teammate for extra damage. A real KO trade can offset a faint.
        old_loss, new_loss = old['lost'] - old['ko'], new['lost'] - new['ko']
        if new_loss > old_loss + 1e-9 or new['ko'] + 1e-9 < old['ko']:
            return False
        if old['lost'] >= 2 / 3 and old_loss > 0:
            if new_loss < old_loss - .5:
                improved = True
            elif new['lost'] >= 2 / 3 and new_loss >= old_loss - 1e-9:
                # If both actions sacrifice the active, do not replace a hit with a fatal boost,
                # or a stronger hit with a resisted one, merely because a rollout later wins.
                if new['damage'] < old['damage'] - .05:
                    return False
                improved |= new['damage'] > old['damage'] + .1
    return improved


def preferred(report, action, ranked):
    """Among non-dominated alternatives, retain the main search's ordering."""
    candidates = [a for a in ranked if dominates(report, a, action)]
    frontier = [a for a in candidates if not any(dominates(report, b, a) for b in candidates)]
    return frontier[0] if frontier else action


def inspect(engine, request, actions, deadline):
    remaining = min(1.5, deadline - time.monotonic() - .2)
    if remaining < .15:
        return {'version': VERSION, 'complete': False, 'reason': 'no-counterplay-budget'}
    query = {k: request[k] for k in ('format', 'samples', 'observationLog')}
    query['rootTactics'] = {'actions': actions, 'deadlineMs': int(remaining * 1000) - 50}
    previous = getattr(engine, 'query_timeout', 30.)
    started = time.monotonic()
    try:
        engine.query_timeout = remaining
        return dict(engine._query(query), seconds=time.monotonic() - started)
    except (RuntimeError, TimeoutError, ValueError) as error:
        return {'version': VERSION, 'complete': False, 'reason': str(error)[:200]}
    finally:
        engine.query_timeout = previous
