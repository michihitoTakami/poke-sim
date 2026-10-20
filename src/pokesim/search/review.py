"""Spend surplus deliberation on paired public-policy games, rather than a deeper fitted-value search."""
import itertools
import time
from collections import defaultdict

VERSION = 'paired-terminal-review-v3'
CONTINUATIONS = ['planned/tactical', 'planned/disrupt', 'planned/expert', 'planned/counterplay']


def search_with_review(engine, request, *, started=None, predictor=None, max_budget=20., normal_move=True):
    from pokesim.search.terminal import search

    started = time.monotonic() if started is None else started
    total_ms = request.get('decisionMs', 8500)
    if not 100 <= total_ms <= max_budget * 1000:
        raise ValueError(f'decisionMs must be between 100 and {int(max_budget * 1000)}')
    # Long offline oracles retain their requested tree budget; surplus review is a live-budget strategy.
    enabled = request.get('terminalReview') and 8500 < total_ms <= 20000 and normal_move
    tactical = normal_move and request.get('rootCounterplay', True) and 1500 <= total_ms <= 20000
    if enabled or tactical:
        request['decisionMs'] = min(8500 if enabled else total_ms, total_ms - (1000 if tactical else 0))
    matrices = search(engine, request, started=started, predictor=predictor, max_budget=max_budget)
    if tactical and not matrices[0]['terminal_search'].get('immediate_wins'):
        from pokesim.search.counterplay import inspect

        result = inspect(engine, request, matrices[0]['p1'], started + total_ms / 1000)
        for row in matrices:
            row['root_counterplay'] = result
    if enabled:
        result = review(engine, request, matrices, started + total_ms / 1000)
        for row in matrices:
            row['terminal_review'] = result
    return matrices


def summarize(reply, actions, worlds):
    if reply.get('informationScope') != 'public_observation':
        raise ValueError('terminal review must use actor observations')
    expected = set(itertools.product(range(worlds), [0, 1], CONTINUATIONS))
    by_action = {}
    for action in actions:
        trials = [t for t in reply['trials'] if t['action'] == action]
        keys = [(t['world'], t['seed'], t['continuation']) for t in trials]
        if set(keys) != expected or len(keys) != len(expected):
            raise ValueError('missing or duplicate paired terminal comparisons')
        if any(t['outcome'] not in ['win', 'loss', 'draw'] or t['target'] is None for t in trials):
            return {'version': VERSION, 'complete': False, 'reason': 'unfinished-or-failed-comparison'}
        cells = defaultdict(list)
        for t in trials:
            cells[f'{t["world"]}:{t["continuation"]}'].append(t['target'])
        values = {k: sum(v) / len(v) for k, v in cells.items()}
        by_action[action] = {'mean': sum(values.values()) / len(values), 'cells': values,
                             'wins': sum(t['outcome'] == 'win' for t in trials), 'trials': len(trials)}
    return {'version': VERSION, 'complete': True, 'actions': by_action,
            'scope': 'same sampled worlds/chance seeds; four public opponent policies incl. counterplay; not a proof'}


def preferred(report, incumbent):
    """Change only for a completed gain >= .1 with no worse world/policy mean, not for a lone lucky game."""
    if not report.get('complete') or incumbent not in report.get('actions', {}):
        return incumbent
    old = report['actions'][incumbent]
    candidates = [(action, row) for action, row in report['actions'].items()
                  if row['mean'] >= old['mean'] + .1 and
                  all(row['cells'].get(k, -1) >= value for k, value in old['cells'].items())]
    return max(candidates, key=lambda x: x[1]['mean'])[0] if candidates else incumbent


def review(engine, request, matrices, deadline):
    remaining = deadline - time.monotonic() - .25
    actions = matrices[0]['p1']
    if remaining < .5 or len(actions) < 2 or matrices[0]['terminal_search'].get('immediate_wins'):
        return {'version': VERSION, 'complete': False, 'reason': 'no-surplus-or-already-decided'}
    trial = {k: request[k] for k in ['format', 'samples', 'observationLog']}
    trial['branches'] = {'informationScope': 'public_observation', 'observationLog': request['observationLog'],
        'seeds': [0, 1], 'maxTurns': 120, 'actions': actions, 'recordTrajectories': False, 'livePolicies': True,
        'continuations': CONTINUATIONS}
    previous = getattr(engine, 'query_timeout', 30.)
    started = time.monotonic()
    try:
        engine.query_timeout = remaining
        result = summarize(engine._query(trial), actions, len(request['samples']))
    except (TimeoutError, RuntimeError, ValueError) as error:
        result = {'version': VERSION, 'complete': False, 'reason': str(error)[:200]}
    finally:
        engine.query_timeout = previous
    result['seconds'] = time.monotonic() - started
    return result
