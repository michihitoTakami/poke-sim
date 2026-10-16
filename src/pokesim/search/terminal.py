"""Budgeted actor-observation tree search with exact sampled-world one-step win checks."""
import time

import numpy as np

from pokesim.search.evaluation import leaf_value


def column_weights(results, predictor):
    """Per world, the predicted likelihood of each of the opponent's root replies in the worker's order.
    predictor(payoff, opponent_labels) -> mix over the labels kept by clean(); dropped (failed) replies get 0."""
    from pokesim.search.lookahead import clean

    out = []
    for res in results:
        try:
            payoff, labels, _ = clean(res)
        except ValueError:
            out.append([1.0] * len(res['p2']))
            continue
        q = predictor(payoff, labels)
        weights = dict(zip(labels, (float(x) for x in q), strict=True))
        out.append([weights.get(lab, 0.0) for lab in res['p2']])
    return out


def search(engine, request, *, started=None, max_budget=9., predictor=None):
    """max_budget: the live limit is 9 s (one decision); offline oracles (pokesim.evaluation.positions) raise it.
    predictor: the opponent prediction the planner will use (pokesim.search.planner.predict with its models);
    once every root cell has its first trial, the worker allocates the opponent's replies by it (column_weights).
    """
    started = time.monotonic() if started is None else started
    if request.get('valueUtility', 'reference-material-v1') != 'reference-material-v1':
        raise ValueError('unsupported search value utility')
    budget = float(request.pop('decisionMs', 8500)) / 1000
    if not .1 <= budget <= max_budget:
        raise ValueError(f'decisionMs must be between 100 and {int(max_budget * 1000)}')
    deadline = started + budget
    request.pop('valueLeaves', None)
    request['terminalSearch'] = {'op': 'start', 'sliceMs': 400, 'certifyMs': min(1200, budget * 200)}
    old_timeout = getattr(engine, 'query_timeout', 30.)
    feedback, mix, columns = [], None, None
    inference_ms = 0.
    try:
        while True:
            engine.query_timeout = max(.1, deadline - time.monotonic() + .2)
            reply = engine._query(request)
            start_inference = time.monotonic()
            probabilities = engine.value_model.probabilities([p['observation'] for p in reply['pending']])
            # p on this wire is a normalized bounded utility, not a calibrated human-game win probability.
            feedback = [{'id': row['id'], 'p': (leaf_value(row['observation'], float(p)) + 1) / 2} for row, p in
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
                if predictor is not None and columns is None:
                    columns = column_weights(reply['results'], predictor)
            request = {'terminalSearch': {'op': 'advance', 'session': reply['session'],
                                         'feedback': feedback, 'mix': mix, 'columns': columns,
                                         'sliceMs': min(500, max(1, int((remaining - .7) * 1000)))}}
            columns = None  # sent once; the worker keeps it
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
