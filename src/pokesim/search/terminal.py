"""Budgeted actor-observation tree search with exact sampled-world one-step win checks."""
import time

import numpy as np


def search(engine, request, *, started=None):
    started = time.monotonic() if started is None else started
    budget = float(request.pop('decisionMs', 8500)) / 1000
    if not .1 <= budget <= 9:
        raise ValueError('decisionMs must be between 100 and 9000')
    deadline = started + budget
    request.pop('valueLeaves', None)
    request['terminalSearch'] = {'op': 'start', 'sliceMs': 400, 'certifyMs': min(1200, budget * 200)}
    old_timeout = getattr(engine, 'query_timeout', 30.)
    feedback, mix = [], None
    inference_ms = 0.
    try:
        while True:
            engine.query_timeout = max(.1, deadline - time.monotonic() + .2)
            reply = engine._query(request)
            start_inference = time.monotonic()
            probabilities = engine.value_model.probabilities([p['observation'] for p in reply['pending']])
            feedback = [{'id': row['id'], 'p': float(p)} for row, p in
                        zip(reply['pending'], probabilities, strict=True)]
            inference_ms += 1000 * (time.monotonic() - start_inference)
            summary = reply['summary']
            remaining = deadline - time.monotonic()
            if summary['immediate_wins'] or remaining < .8:
                break
            if summary['ready']:
                # Cheap allocation only; the final decision still uses the common-world robust matrix solver.
                matrices = [np.asarray(r['payoff'], dtype=float) for r in reply['results']]
                if all(np.isfinite(m).all() for m in matrices):
                    scores = np.mean([.7 * m.mean(axis=1) + .3 * m.min(axis=1) for m in matrices], axis=0)
                    weights = np.exp((scores - scores.max()) / .15)
                    mix = (weights / weights.sum()).tolist()
            request = {'terminalSearch': {'op': 'advance', 'session': reply['session'],
                                         'feedback': feedback, 'mix': mix,
                                         'sliceMs': min(500, max(1, int((remaining - .7) * 1000)))}}
        engine.query_timeout = max(.1, deadline - time.monotonic() + .2)
        final = engine._query({'terminalSearch': {'op': 'finish', 'session': reply['session'],
                                                 'feedback': feedback}})
        if not final['summary']['ready']:
            raise TimeoutError('terminal search could not cover all root replies within budget')
        final['summary'].update(budget_ms=budget * 1000, elapsed_ms=1000 * (time.monotonic() - started),
                                hidden_sets_exhaustive=False)
        for res in final['results']:
            res['terminal_search'] = final['summary']
            res['value_status'] = {'model': engine.value_model.version,
                                   'terminal': final['summary']['terminal'],
                                   'learned': 0, 'failed': 0, 'inference_ms': inference_ms}
        # Counts are global, not repeated once per world in Advisor's aggregation.
        final['results'][0]['value_status'].update(learned=final['summary'].get('learned', 0),
                                                 failed=final['summary']['failed'])
        for res in final['results'][1:]:
            res['value_status']['terminal'] = 0
        return final['results']
    finally:
        engine.query_timeout = old_timeout
