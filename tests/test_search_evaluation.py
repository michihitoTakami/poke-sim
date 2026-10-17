import copy
import json
import subprocess

import pytest

from pokesim.config import ROOT
from pokesim.live.advisor import action_evaluations
from pokesim.search.evaluation import health, leaf_value, position_value


def observation(ours=None, foes=None):
    own = ours or [{'condition': '100/100', 'active': True} for _ in range(3)]
    known = foes or {'foe': {'condition': '100/100', 'boosts': {}}}
    return {'side': 'p1', 'request': {'side': {'pokemon': own}},
            'public': {'mons': {'p1': {'me': {'condition': own[0]['condition'], 'boosts': {}}}, 'p2': known},
                       'active': {'p1': 'me', 'p2': next(iter(known))}}}


def test_unseen_foes_are_not_assumed_dead_and_a_high_reference_prediction_is_not_a_certain_win():
    obs = observation()
    assert position_value(obs) == pytest.approx(0)
    assert leaf_value(obs, .99) == pytest.approx(.49)
    behind = observation([{'condition': '10/100', 'active': True}, {'condition': '0 fnt'}, {'condition': '0 fnt'}])
    assert leaf_value(behind, .99) < .1


def test_material_damage_and_recovery_create_real_score_differences_at_a_flat_model_value():
    even = observation()
    damage = copy.deepcopy(even)
    damage['public']['mons']['p2']['foe']['condition'] = '20/100y'
    faint = copy.deepcopy(even)
    faint['public']['mons']['p2']['foe']['condition'] = '0 fnt'
    assert leaf_value(even, .8) < leaf_value(damage, .8) < leaf_value(faint, .8)
    hurt = observation([{'condition': '30/100', 'active': True}, {'condition': '100/100'}, {'condition': '100/100'}])
    assert leaf_value(hurt, .8) < leaf_value(even, .8)
    assert health('20/100y') == (.2, '')


def test_setup_has_diminishing_returns_and_open_values_stay_below_terminal_values():
    obs = observation()
    scores = []
    for stages in [0, 2, 4, 6, 8]:
        obs['public']['mons']['p1']['me']['boosts'] = {'spa': stages}
        scores.append(leaf_value(obs, 1))
    assert scores[0] < scores[1] < scores[2] < scores[3] == scores[4]
    assert scores[1] - scores[0] > scores[3] - scores[2]
    assert max(scores) <= .85


def test_action_scores_and_exclusions_are_not_confused_with_one_hot_selection_probabilities():
    status = {'search': {'root_evidence': [
        {'label': 'move:a', 'utility': .2, 'worst': -.1, 'expected': .3, 'trials': 20},
        {'label': 'move:b', 'utility': .5, 'worst': .3, 'expected': .6, 'trials': 30}],
        'terminal_search': {'action_filter': {'excluded': [
            {'label': 'move:recover', 'reasons': ['full-hp-heal-before-any-damage']}]}}}}
    rows = action_evaluations(status)
    assert [row['label'] for row in rows] == ['move:b', 'move:a', 'move:recover']
    assert rows[1]['gap'] == pytest.approx(.3)
    assert rows[2]['utility'] is None and rows[2]['excluded']
    selected = action_evaluations(status, 'move:a')
    assert selected[0]['label'] == 'move:a' and selected[0]['selected']
    assert selected[0]['gap'] == pytest.approx(.3)


def test_estimated_tiny_score_differences_prefer_finishing_but_meaningful_gains_are_preserved():
    from pokesim.search.planner import plan

    row = {'p1': ['slow', 'fast'], 'p2': ['reply'], 'payoff': [[.95], [.94]],
           'winSpeed': [[.1], [.5]], 'visits': [[30], [30]], 'terminal_search': {'immediate_wins': []}}
    assert plan([row], pure=True).ranked()[0][0] == 'fast'
    row['payoff'][1][0] = .7
    assert plan([row], pure=True).ranked()[0][0] == 'slow'
    # Exact/non-tree matrices still use the strict win-first comparison.
    row.pop('terminal_search')
    row['payoff'][1][0] = .94
    assert plan([row], pure=True).ranked()[0][0] == 'slow'


def test_near_ties_use_immediate_progress_before_deep_rollout_win_speed():
    from pokesim.search.planner import plan

    row = {'p1': ['move:nastyplot', 'move:shadowball'], 'p2': ['reply'],
           'payoff': [[.55], [.54]], 'winSpeed': [[.5], [.1]], 'rootProgress': [[0.], [.4]],
           'visits': [[30], [30]], 'terminal_search': {'immediate_wins': []}}
    assert plan([row], pure=True).ranked()[0][0] == 'move:shadowball'
    # A meaningful estimated advantage still overrides this secondary preference.
    row['payoff'][0][0] = .9
    assert plan([row], pure=True).ranked()[0][0] == 'move:nastyplot'
    # A beneficial recovery is progress too; the rule is not a blanket ban on status moves.
    row['p1'] = ['move:recover', 'move:shadowball']
    row['payoff'] = [[.55], [.54]]
    row['rootProgress'] = [[.3], [-.7]]
    assert plan([row], pure=True).ranked()[0][0] == 'move:recover'


def test_overlay_shows_move_utilities_instead_of_one_hot_percentages():
    script = (ROOT / 'web/overlay.html').read_text().split('<script>')[1].split('</script>')[0]
    code = """
const fs=require('fs'),vm=require('vm');const nodes={};
const context={document:{getElementById:id=>nodes[id]??={}},WebSocket:class{},location:{host:'test'}};
vm.createContext(context);vm.runInContext(fs.readFileSync(0,'utf8'),context);
context.render({advice:{actions:[{label:'Attack',p:1}],opponent:[],win_prob:null,
 evaluations:[{label:'Attack',utility:.4},{label:'Heal',utility:.2}]}});
console.log(JSON.stringify(nodes));
"""
    result = subprocess.run(['node', '-e', code], input=script, text=True, capture_output=True, check=True)
    nodes = json.loads(result.stdout)
    assert '0.400' in nodes['actions']['innerHTML'] and '0.200' in nodes['actions']['innerHTML']
    assert '100%' not in nodes['actions']['innerHTML']
    assert '勝率ではありません' in nodes['me-title']['textContent']


def test_nonterminal_rollouts_stop_at_the_tactical_horizon_and_do_not_claim_99_percent():
    from test_expert_continuation import CHOMP, side
    from test_terminal_search import FlatValue

    from pokesim.search.lookahead import Lookahead
    from pokesim.search.terminal import search

    mon = dict(CHOMP, item='', ability='', moves=['Splash'])
    log = ['|switch|p1a: Me|Garchomp, L50|100/100', '|switch|p2a: Foe|Garchomp, L50|100/100', '|turn|1']
    request = {'format': 'gen9championsbssregmc', 'decisionMs': 1200, 'observationLog': log,
               'samples': [{'p1': side([mon]), 'p2': side([mon]), 'field': {}}]}
    engine = Lookahead(None, value_model=FlatValue())
    try:
        result = search(engine, request)[0]
    finally:
        engine.close()
    summary = result['terminal_search']
    assert summary['horizon_turns'] == 8 and summary['max_rollout_turns'] == 8
    assert summary['censored'] > 0 and summary['terminal'] == 0 and summary['failed'] == 0
    assert result['payoff'][0][0] == pytest.approx(.49)
    assert not summary['calibrated_win_probability']


def test_api_preserves_selection_probability_and_separate_named_move_evaluations():
    from pokesim.api.server import Live
    from pokesim.live.advisor import Suggestion

    live = object.__new__(Live)
    live.reset()
    live.ja, live.obs = {'shadowball': 'シャドーボール'}, None
    live.suggestion = Suggestion([('move:shadowball', 1)], evaluations=[
        {'label': 'move:shadowball', 'utility': .4, 'worst': .1, 'gap': 0., 'selected': True}])
    advice = live.state()['advice']
    assert advice['actions'][0]['p'] == 1
    assert advice['evaluations'][0]['utility'] == .4
    assert advice['evaluations'][0]['label'] == 'シャドーボール'
    assert advice['evaluations'][0]['id'] == 'move:shadowball'


def test_fixed_depth_advice_also_exposes_scores_without_turning_them_into_win_probabilities():
    from types import SimpleNamespace

    from test_expert_continuation import GHOLDENGO

    from pokesim.live.advisor import Advisor
    from pokesim.live.session import BattleSession

    session = BattleSession([GHOLDENGO])
    session.preview(['Blissey'])
    session.select([0])
    session.apply(['|switch|p1a: Gholdengo|Gholdengo, L50|175/175',
                   '|switch|p2a: Blissey|Blissey, L50|100/100', '|turn|1'])
    row = {'p1': ['move:shadowball', 'move:flashcannon'], 'p2': ['move:splash'], 'payoff': [[-.5], [.6]]}
    advisor = Advisor([GHOLDENGO], policy=None, lookahead=False, opponent_policy=None, belief=False)
    advisor.engine = SimpleNamespace(query=lambda *args, **kwargs: [row], opponent_model={})
    result = advisor.advise(session.decision())
    assert result.win_prob is None
    assert {r['label']: r['expected'] for r in result.evaluations} == {'move:shadowball': -.5, 'move:flashcannon': .6}
