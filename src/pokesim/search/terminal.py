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
        prior = dict(zip(res['p2'], res['p2Prior'], strict=True)) if res.get('p2Prior') else None
        q = predictor(payoff, labels, prior) if prior else predictor(payoff, labels)
        weights = dict(zip(labels, (float(x) for x in q), strict=True))
        out.append([weights.get(lab, 0.0) for lab in res['p2']])
    return out


SUMMED = ('simulations', 'terminal', 'learned', 'censored', 'failed', 'unfinished_at_deadline', 'chance_paths',
          'tree_nodes')
MAXED = ('max_tree_decisions', 'max_rollout_turns')


def merge_replies(replies: list[dict], prior_visits: float) -> dict:
    """One reply from several workers' replies over the same worlds: per cell the rollouts are pooled, the value
    bootstraps averaged, and the payoff recomputed as the worker does (the bootstrap weighing prior_visits).
    Certificates come from the first worker (the only one that certifies); the summary adds the counts."""
    first = replies[0]
    summary = dict(first['summary'])
    for key in SUMMED:
        summary[key] = sum(r['summary'].get(key, 0) for r in replies)
    for key in MAXED:
        summary[key] = max(r['summary'].get(key, 0) for r in replies)
    summary['errors'] = [e for r in replies for e in r['summary'].get('errors', [])][:5]
    summary['ready'] = all(r['summary']['ready'] for r in replies)
    summary['workers'] = len(replies)
    summary['worker_simulations'] = [r['summary'].get('simulations', 0) for r in replies]
    certified = set(first['summary'].get('immediate_wins') or [])
    results = []
    for w, world in enumerate(first['results']):
        merged = {k: v for k, v in world.items() if k not in ('payoff', 'winSpeed', 'visits', 'invalid', 'stats')}
        priors = [r['results'][w].get('p2Prior') for r in replies if r['results'][w].get('p2Prior')]
        if priors:
            merged['p2Prior'] = np.mean(priors, axis=0).tolist()
        payoff, speed, visits, invalid, stats = [], [], [], [], []
        for i, label in enumerate(world['p1']):
            row_p, row_s, row_v, row_i, row_st = [], [], [], [], []
            for j in range(len(world['p2'])):
                cells = [r['results'][w]['stats'][i][j] for r in replies]
                n = sum(c['n'] for c in cells)
                total = sum(c['sum'] for c in cells)
                total_speed = sum(c['speed'] for c in cells)
                priors = [c['prior'] for c in cells if c['prior'] is not None]
                prior = float(np.mean(priors)) if priors else None
                prior_speed = float(np.mean([c['priorSpeed'] for c in cells if c['prior'] is not None] or [0.]))
                n0 = prior_visits if prior is not None else 0.
                if label in certified:
                    value = 1.
                elif n:
                    value = (n0 * prior + total) / (n0 + n) if prior is not None else total / n
                else:
                    value = prior
                speed_value = ((n0 * prior_speed + total_speed) / (n0 + n)) if n else prior_speed
                row_p.append(value)
                row_s.append(speed_value)
                row_v.append(n)
                row_i.append(sum(c['failed'] for c in cells))
                row_st.append({'n': n, 'sum': total, 'prior': prior, 'speed': total_speed,
                               'priorSpeed': prior_speed, 'ends': sum(c['ends'] for c in cells),
                               'failed': row_i[-1]})
            payoff.append(row_p)
            speed.append(row_s)
            visits.append(row_v)
            invalid.append(row_i)
            stats.append(row_st)
        merged.update(payoff=payoff, winSpeed=speed, visits=visits, invalid=invalid, stats=stats,
                      terminal_search=summary)
        results.append(merged)
    return {'session': first['session'], 'pending': [], 'summary': summary, 'results': results}


def search(engine, request, *, started=None, max_budget=9., predictor=None):
    """max_budget: the live limit is 9 s (one decision); offline oracles (pokesim.evaluation.positions) raise it.
    predictor: the opponent prediction the planner will use (pokesim.search.planner.predict with its models);
    once every root cell has its first trial, the worker allocates the opponent's replies by it (column_weights).
    request['workers']: Node worker processes searching the same worlds on independent random streams; their
    cells are pooled (merge_replies). A worker that fails mid-decision is dropped; the rest decide.
    """
    started = time.monotonic() if started is None else started
    if request.get('valueUtility', 'reference-material-v1') != 'reference-material-v1':
        raise ValueError('unsupported search value utility')
    budget = float(request.pop('decisionMs', 8500)) / 1000
    if not .1 <= budget <= max_budget:
        raise ValueError(f'decisionMs must be between 100 and {int(max_budget * 1000)}')
    count = int(request.pop('workers', 1) or 1)
    if not 1 <= count <= 16:
        raise ValueError('workers must be between 1 and 16')
    prior_visits = float(request.get('priorVisits', 0) or 0)
    deadline = started + budget
    request.pop('valueLeaves', None)
    workers = engine.pool(count) if count > 1 else None
    inference_ms = 0.
    failures: list[str] = []
    old_timeout = getattr(engine, 'query_timeout', 30.)

    def exchange(requests: dict) -> dict:
        """requests: worker index -> request. Replies by worker index; a failing worker is dropped and noted."""
        timeout = max(.1, deadline - time.monotonic() + .2)
        if workers is None:
            engine.query_timeout = timeout
            return {0: engine._query(requests[0])}
        replies = {}
        for k, req in requests.items():
            try:
                workers[k].send(req)
            except OSError as e:
                failures.append(f'worker {k}: {e!r}'[:120])
        for k in list(requests):
            if any(f.startswith(f'worker {k}:') for f in failures):
                continue
            try:
                replies[k] = workers[k].recv(max(.1, deadline - time.monotonic() + .2))
            except (TimeoutError, RuntimeError) as e:
                failures.append(f'worker {k}: {e!r}'[:120])
                workers[k].kill()
        if not replies:
            raise RuntimeError('every search worker failed: ' + '; '.join(failures))
        return replies

    try:
        start = {'op': 'start', 'sliceMs': 400, 'certifyMs': min(1200, budget * 200)}
        requests = {0: dict(request, terminalSearch=start)}
        for k in range(1, count):
            requests[k] = dict(request, stream=request.get('stream', 0) * 16 + k,
                               terminalSearch=dict(start, certifyMs=0))
        feedback: dict[int, list] = {}
        mix, columns, sent_columns = None, None, False
        while True:
            replies = exchange(requests)
            start_inference = time.monotonic()
            for k, reply in replies.items():
                probabilities = engine.value_model.probabilities([p['observation'] for p in reply['pending']])
                # p on this wire is a normalized bounded utility, not a calibrated human-game win probability.
                feedback[k] = [{'id': row['id'], 'p': (leaf_value(row['observation'], float(p)) + 1) / 2}
                               for row, p in zip(reply['pending'], probabilities, strict=True)]
            inference_ms += 1000 * (time.monotonic() - start_inference)
            merged = merge_replies(list(replies.values()), prior_visits) if len(replies) > 1 else \
                next(iter(replies.values()))
            summary = merged['summary']
            remaining = deadline - time.monotonic()
            if summary['immediate_wins'] or remaining < .8:
                break
            if summary['ready']:
                # Cheap allocation only; the final decision still uses the common-world robust matrix solver.
                matrices = [np.asarray(r['payoff'], dtype=float) for r in merged['results']]
                if all(np.isfinite(m).all() for m in matrices):
                    scores = np.mean([.7 * m.mean(axis=1) + .3 * m.min(axis=1) for m in matrices], axis=0)
                    weights = np.exp((scores - scores.max()) / .15)
                    mix = (weights / weights.sum()).tolist()
                if predictor is not None and not sent_columns:
                    columns, sent_columns = column_weights(merged['results'], predictor), True
            slice_ms = min(500, max(1, int((remaining - .7) * 1000)))
            requests = {k: {'terminalSearch': {'op': 'advance', 'session': reply['session'],
                                               'feedback': feedback[k], 'mix': mix, 'columns': columns,
                                               'sliceMs': slice_ms}} for k, reply in replies.items()}
            columns = None  # sent once; the workers keep it
        finals = exchange({k: {'terminalSearch': {'op': 'finish', 'session': reply['session'],
                                                  'feedback': feedback[k]}} for k, reply in replies.items()})
        final = merge_replies(list(finals.values()), prior_visits) if len(finals) > 1 else \
            next(iter(finals.values()))
        if not final['summary']['ready']:
            raise TimeoutError('terminal search could not cover all root replies within budget')
        final['summary'].update(budget_ms=budget * 1000, elapsed_ms=1000 * (time.monotonic() - started),
                                hidden_sets_exhaustive=False, workers=len(finals), workers_requested=count,
                                workers_failed=failures)
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
