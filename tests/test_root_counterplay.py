import json

import pytest
from test_expert_continuation import CHOMP, DIRGE, GHOLDENGO, PASSER, side

from pokesim.search.counterplay import dominates, preferred
from pokesim.search.planner import plan


def probe(sample, actions):
    from test_worker_transitions import worker

    return worker("""
const {State}=require(process.argv[2]+'/dist/sim/state');
const {extractChannelMessages}=require(process.argv[2]+'/dist/sim/battle');
const {rootTactics}=require('./services/sim-tools/root_tactics');
const b=setUp(sample,format,1), log=[];
for(const side of b.sides) {
 const m=side.active[0];log.push(`|switch|${side.id}a: ${m.name}|${m.species.name}, L50|${m.hp}/${m.maxhp}`);
}
log.push('|turn|2');b.destroy();
console.log(JSON.stringify(rootTactics({format,samples:[sample],observationLog:log,
 rootTactics:{actions:ACTIONS,deadlineMs:5000}}, {setUp,step,choices,State,extractChannelMessages})));
""".replace('ACTIONS', json.dumps(actions)), sample)


def test_fatal_boost_is_dominated_by_a_hit_but_safe_boosts_are_not_penalized_for_no_damage():
    foe = dict(CHOMP, species='Glimmora', ability='Toxic Debris', item='', nature='Modest',
               moves=['Earth Power'], evs={'hp': 2, 'spa': 32, 'spe': 32})
    ours = dict(GHOLDENGO, item='')
    sample = {'p1': side([ours], [{'hp': .25, 'boosts': {'spe': 2}}]), 'p2': side([foe]), 'field': {}}
    result = probe(sample, ['move:nastyplot', 'move:flashcannon', 'move:shadowball'])
    assert result['complete']
    assert dominates(result, 'move:flashcannon', 'move:nastyplot')
    assert preferred(result, 'move:nastyplot', result['actions']) == 'move:flashcannon'
    sample['p1']['mons'][0].update(hp=1, boosts={'spe': 2, 'spd': 6})
    safe = probe(sample, result['actions'])
    assert not dominates(safe, 'move:flashcannon', 'move:nastyplot')


def test_a_real_ko_trade_is_not_vetoed_to_preserve_a_teammate():
    report = {'complete': True, 'cells': {'0:move:hit': {
        'trade': {'lost': 1, 'ko': 1, 'damage': 1}, 'wait': {'lost': 0, 'ko': 0, 'damage': 0}}}}
    assert not dominates(report, 'wait', 'trade')
    assert preferred(report, 'trade', ['wait', 'trade']) == 'trade'


def test_no_override_if_an_alternative_worsens_another_legal_reply_or_evidence_is_incomplete():
    report = {'complete': True, 'cells': {
        '0:move:a': {'setup': {'lost': 1, 'ko': 0, 'damage': 0},
                     'switch': {'lost': 0, 'ko': 0, 'damage': 0}},
        '0:move:b': {'setup': {'lost': 0, 'ko': 0, 'damage': 0},
                     'switch': {'lost': 1, 'ko': 0, 'damage': 0}}}}
    assert not dominates(report, 'switch', 'setup')
    del report['cells']['0:move:b']
    assert dominates(report, 'switch', 'setup')
    report['complete'] = False
    assert preferred(report, 'setup', ['switch']) == 'setup'


def test_tree_and_terminal_review_cannot_restore_a_dominated_fatal_setup():
    report = {'complete': True, 'cells': {'0:move:a': {
        'move:shadowball': {'lost': 1, 'ko': 0, 'damage': .5},
        'move:nastyplot': {'lost': 1, 'ko': 0, 'damage': 0}}}}
    row = {'p1': ['move:shadowball', 'move:nastyplot'], 'p2': ['move:a'],
           'payoff': [[.1], [.2]], 'winSpeed': [[0], [0]],
           'visits': [[3], [3]],
           'terminal_search': {'immediate_wins': []}, 'root_counterplay': report}
    advice = plan([row], pure=True)
    assert advice.ranked()[0][0] == 'move:shadowball'
    assert advice.status['root_counterplay']['selected'] == 'move:shadowball'
    row['payoff'] = [[.2], [.1]]
    row['terminal_review'] = {'complete': True, 'actions': {
        'move:shadowball': {'mean': .5, 'cells': {'a': .5}},
        'move:nastyplot': {'mean': 1, 'cells': {'a': 1}}}}
    advice = plan([row], pure=True)
    assert advice.ranked()[0][0] == 'move:shadowball'
    assert advice.status['terminal_review']['veto'] == 'dominated-immediate-sacrifice'
    row['terminal_search']['immediate_wins'] = ['move:nastyplot']
    assert plan([row], pure=True).ranked()[0][0] == 'move:nastyplot'


def test_passing_finishes_the_pending_recipient_choice_without_losing_the_passer():
    sample = {'p1': side([PASSER, DIRGE], [{'hp': .8, 'boosts': {'spe': 4, 'def': 4}}, {'hp': 1}]),
              'p2': side([dict(CHOMP, moves=['Earthquake'])]), 'field': {}}
    result = probe(sample, ['move:batonpass', 'move:irondefense'])
    assert result['complete']
    assert result['cells']['0:move:earthquake']['move:batonpass']['lost'] == 0


def test_counterplay_keeps_global_deadline_and_skips_forced_switches(monkeypatch):
    from pokesim.search import review as module

    calls = []
    def search(engine, request, **kwargs):
        calls.append(request['decisionMs'])
        return [{'p1': ['move:a', 'move:b'], 'terminal_search': {}}]
    def inspect(engine, request, actions, deadline):
        calls.append(deadline)
        return {'complete': True}
    monkeypatch.setattr('pokesim.search.terminal.search', search)
    monkeypatch.setattr('pokesim.search.counterplay.inspect', inspect)
    result = module.search_with_review(object(), {'decisionMs': 4375}, started=100)
    assert calls == [3375, 104.375] and result[0]['root_counterplay']['complete']
    calls.clear()
    module.search_with_review(object(), {'decisionMs': 4375}, started=100, normal_move=False)
    assert calls == [4375]


def test_counterplay_timeout_restores_timeout_and_never_changes_the_action():
    import time
    from types import SimpleNamespace

    from pokesim.search.counterplay import inspect

    def fail(_):
        raise TimeoutError('fixture deadline')
    engine = SimpleNamespace(query_timeout=30., _query=fail)
    result = inspect(engine, {'format': 'test', 'samples': [{}], 'observationLog': []}, ['move:a'],
                     time.monotonic() + 2)
    assert not result['complete'] and result['reason'] == 'fixture deadline'
    assert engine.query_timeout == 30.
    assert preferred(result, 'move:a', ['move:b']) == 'move:a'


@pytest.mark.parametrize('fixture', ['counterplay-blaziken-turn5', 'counterplay-baton-turn9'])
def test_real_public_prefixes_reject_the_resisted_sacrifice_and_preserve_the_ready_relay(fixture):
    from pokesim.config import ROOT
    from pokesim.search.lookahead import Lookahead

    source = json.loads((ROOT / f'tests/fixtures/{fixture}.json').read_text())
    actions = ['move:shadowball', 'move:flashcannon'] if 'blaziken' in fixture else [
        'move:batonpass', 'move:protect', 'move:substitute']
    request = {k: source[k] for k in ('format', 'samples', 'observationLog')}
    request['rootTactics'] = {'actions': actions, 'deadlineMs': 4000}
    engine = Lookahead(None)
    try:
        result = engine._query(request)
    finally:
        engine.close()
    assert result['complete']
    if 'blaziken' in fixture:
        assert preferred(result, 'move:flashcannon', actions) == 'move:shadowball'
    else:
        assert preferred(result, 'move:batonpass', actions) == 'move:batonpass'
