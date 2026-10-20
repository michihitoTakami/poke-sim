import itertools
import time
from types import SimpleNamespace

import pytest

from pokesim.search.planner import plan
from pokesim.search.review import CONTINUATIONS, preferred, review, summarize


def evidence():
    return {'complete': True, 'actions': {
        'move:calmmind': {'mean': .5, 'cells': {'0:a': .5, '1:a': .5}},
        'move:moonblast': {'mean': .75, 'cells': {'0:a': 1., '1:a': .5}}}}


def test_review_requires_a_completed_gain_without_a_worse_world_or_opponent_policy():
    result = evidence()
    assert preferred(result, 'move:calmmind') == 'move:moonblast'
    result['actions']['move:moonblast']['cells']['1:a'] = 0
    assert preferred(result, 'move:calmmind') == 'move:calmmind'
    result['complete'] = False
    assert preferred(result, 'move:calmmind') == 'move:calmmind'


def test_unfinished_games_do_not_become_losses_and_missing_pairs_are_rejected():
    trials = [{'action': 'move:a', 'world': 0, 'seed': s, 'continuation': c,
               'outcome': 'win', 'target': 1}
              for s, c in itertools.product([0, 1], CONTINUATIONS)]
    reply = {'informationScope': 'public_observation', 'trials': trials}
    assert summarize(reply, ['move:a'], 1)['complete']
    trials[-1].update(outcome='unfinished', target=None)
    assert not summarize(reply, ['move:a'], 1)['complete']
    trials.pop()
    with pytest.raises(ValueError, match='missing or duplicate'):
        summarize(reply, ['move:a'], 1)


def matrix():
    return {'p1': ['move:calmmind', 'move:moonblast'], 'p2': ['move:tackle'],
            'payoff': [[.6], [.2]], 'winSpeed': [[0], [0]], 'visits': [[4], [4]],
            'terminal_search': {'immediate_wins': [], 'value_kind': 'test'}, 'terminal_review': evidence()}


def test_review_can_correct_fitted_value_optimism_but_never_override_an_immediate_win():
    row = matrix()
    result = plan([row], pure=True)
    assert result.ranked()[0][0] == 'move:moonblast'
    assert result.status['terminal_review']['incumbent'] == 'move:calmmind'
    row['terminal_search']['immediate_wins'] = ['move:calmmind']
    assert plan([row], pure=True).ranked()[0][0] == 'move:calmmind'


def test_mixed_strategy_and_disagreeing_world_reviews_keep_their_original_semantics():
    row = matrix()
    assert plan([row], pure=False).ranked()[0][0] == 'move:calmmind'
    other = dict(row, terminal_review={'complete': False})
    assert plan([row, other], pure=True).ranked()[0][0] == 'move:calmmind'


def test_timeout_preserves_the_primary_search_and_restores_the_query_timeout():
    def fail(_):
        raise TimeoutError('surplus expired')
    engine = SimpleNamespace(query_timeout=30., _query=fail)
    query = {'format': 'test', 'samples': [{}], 'observationLog': []}
    result = review(engine, query, [{'p1': ['move:a', 'move:b'], 'terminal_search': {}}], time.monotonic() + 2)
    assert not result['complete'] and result['reason'] == 'surplus expired'
    assert engine.query_timeout == 30.


def test_surplus_search_keeps_the_original_tree_allowance_and_global_deadline(monkeypatch):
    from pokesim.search import review as module
    calls = []
    def search(engine, request, **kwargs):
        calls.append(request['decisionMs'])
        return [{'p1': ['move:a', 'move:b'], 'terminal_search': {}}]
    def compare(engine, request, matrices, deadline):
        calls.append(deadline)
        return {'complete': False, 'reason': 'fixture'}
    monkeypatch.setattr('pokesim.search.terminal.search', search)
    monkeypatch.setattr(module, 'review', compare)
    request = {'decisionMs': 15000, 'terminalReview': True, 'rootCounterplay': False}
    result = module.search_with_review(object(), request, started=100.)
    assert calls == [8500, 115.] and result[0]['terminal_review']['reason'] == 'fixture'
    with pytest.raises(ValueError, match='decisionMs'):
        module.search_with_review(object(), {'decisionMs': 99000, 'terminalReview': True})


def test_disabling_trajectories_preserves_live_policy_terminal_outcomes():
    import copy
    import json

    from pokesim.config import ROOT
    from pokesim.search.lookahead import Lookahead

    source = json.loads((ROOT / 'tests/fixtures/terminal-moonblast-turn14.json').read_text())
    request = {k: source[k] for k in ['format', 'samples', 'observationLog']}
    request['samples'] = request['samples'][:1]
    request['branches'] = {'informationScope': 'public_observation', 'observationLog': source['observationLog'],
        'seeds': [0, 1], 'maxTurns': 80, 'actions': ['move:moonblast'], 'livePolicies': True,
        'continuations': ['planned/expert']}
    engine = Lookahead(None)
    try:
        before = engine._query(copy.deepcopy(request))
        request['branches']['recordTrajectories'] = False
        after = engine._query(request)
    finally:
        engine.close()
    assert all(t['trajectory'] for t in before['trials'])
    assert all(not t['trajectory'] for t in after['trials'])
    assert [(t['outcome'], t['target']) for t in before['trials']] == [
           (t['outcome'], t['target']) for t in after['trials']]


def test_long_offline_oracles_retain_the_requested_tree_budget(monkeypatch):
    from pokesim.search.review import search_with_review
    budgets = []
    def search(engine, request, **kwargs):
        budgets.append(request['decisionMs'])
        return [{}]
    monkeypatch.setattr('pokesim.search.terminal.search', search)
    assert search_with_review(object(), {'decisionMs': 60000, 'terminalReview': True}, max_budget=60.) == [{}]
    assert budgets == [60000]
