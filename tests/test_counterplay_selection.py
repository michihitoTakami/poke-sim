import itertools
import random
from types import SimpleNamespace

import pytest

from pokesim.live.lineup_search import LineupSearch, counterplay_worlds
from pokesim.live.selection import OpponentPicks


def selection_fixture():
    names = ['passer', 'special', 'fairy', 'wall', 'physical', 'sixth']
    sets = [{'species': name, 'moves': ['Baton Pass'] if name == 'passer' else ['Tackle']} for name in names]
    lineups = [[lead, *sorted(set(trio) - {lead})] for trio in itertools.combinations(names, 3) for lead in trio]
    lineups.sort(key=lambda trio: trio[0] != 'passer')
    ranking = [(trio, 1 - i / 100, {}) for i, trio in enumerate(lineups)]
    rng = random.Random(9)
    def sample(name):
        return {'species': name, 'moves': ['Taunt'] if name == 'x' else
                ['Swords Dance', 'Earthquake'] if name == 'y' else ['Tackle']}
    calls = []
    def query(request):
        cfg = request['lineupMatches']
        calls.append(cfg)
        rows = []
        for i, team in enumerate(cfg['ours']):
            safe = any(m['species'] == 'wall' for m in team)
            for w, policy in itertools.product(range(len(cfg['worlds'])), cfg['policies']):
                target = .5 if safe else 0. if policy == 'counterplay' else 1.
                rows.append({'candidate': i, 'world': w, 'opponent': policy,
                             'outcome': {0.: 'loss', .5: 'draw', 1.: 'win'}[target], 'target': target})
        return {'version': 'fixture', 'informationScope': 'public_observation', 'trials': rows}
    engine = SimpleNamespace(sampler=SimpleNamespace(rng=rng, sample=sample), _query=query)
    picks = OpponentPicks(['x', 'y', 'z', 'u', 'v', 'w'], [1.] * 6, [1/6] * 6,
                          [(['z', 'u', 'v'], .9), (['x', 'y', 'w'], .1)])
    return LineupSearch(sets, engine), ranking, picks, calls


def test_selection_covers_every_trio_and_ranks_the_weakest_response_without_a_species_bonus():
    selector, ranking, picks, calls = selection_fixture()
    original = selector.engine.sampler.rng.getstate()
    ranked, info = selector.rerank(ranking, picks)
    cfg = calls[0]
    assert len({tuple(sorted(m['species'] for m in team)) for team in cfg['ours']}) == 20
    assert 'wall' in ranked[0][0]
    assert info['nominal_worlds'] == 4 and info['worlds'] == 6
    assert info['stress_worlds'] == ['disruption', 'physical_setup']
    assert cfg['worlds'][4][0]['species'] == 'x' and cfg['worlds'][5][0]['species'] == 'y'
    assert all(set(m['species'] for m in team) <= set(picks.species) for team in cfg['worlds'])
    assert selector.engine.sampler.rng.getstate() == original
    assert selector.rerank(ranking, picks) == (ranked, info) and len(calls) == 1


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'failed'])
def test_incomplete_comparisons_cannot_silently_rerank_lineups(corruption):
    selector, ranking, picks, _ = selection_fixture()
    query = selector.engine._query
    def broken(request):
        reply = query(request)
        if corruption == 'missing':
            reply['trials'].pop()
        elif corruption == 'duplicate':
            reply['trials'].append(reply['trials'][0])
        else:
            reply['trials'][0].update(target=None, outcome='failed')
        return reply
    selector.engine._query = broken
    if corruption == 'failed':
        ranked, info = selector.rerank(ranking, picks)
        assert ranked == ranking and info['fallback'] == 'failed-continuation'
    else:
        with pytest.raises(ValueError, match='missing or duplicate'):
            selector.rerank(ranking, picks)


def test_stress_policy_uses_visible_disruption_and_an_exposed_attack():
    from test_expert_continuation import CHOMP, PASSER, side
    from test_proposals37 import planned_sample

    taunter = dict(CHOMP, species='Sableye', ability='Prankster', item='', moves=['Taunt', 'Foul Play'])
    sample = {'p1': side([taunter]), 'p2': side([PASSER]), 'field': {}}
    action = planned_sample(sample, """
const {chooseCounterplay}=require('./services/sim-tools/counterplay_policy');
console.log(JSON.stringify(chooseCounterplay(obs.view(b.p1),b.dex)[1]));b.destroy();
""")
    assert action == 'move:taunt'


def test_final_ranking_retains_real_ladder_evidence_without_overriding_a_clear_simulated_loss():
    selector, ranking, picks, _ = selection_fixture()
    first, second = ranking[:2]
    # Both candidates have the same simulated matchup; a reliable real-game difference breaks that tie.
    ranking[0] = (first[0], first[1], {'history': {'mean': .1, 'ordered_games': 80, 'trio_games': 100}})
    ranking[1] = (second[0], second[1], {'history': {'mean': .8, 'ordered_games': 80, 'trio_games': 100}})
    ranked, info = selector.rerank(ranking, picks)
    by_names = {tuple(r[0]): r[1] for r in ranked}
    assert by_names[tuple(second[0])] > by_names[tuple(first[0])]
    assert any(e['history_weight'] == .35 for e in info['candidates'])
    assert all(e['history_weight'] == 0 for e in info['candidates'] if e['lineup'] not in [first[0], second[0]])


def test_preview_stresses_special_setup_and_soak_toxic_using_complete_prior_sets():
    selector, _, picks, _ = selection_fixture()
    def sample(name):
        return {'species': name, 'moves': ['Calm Mind', 'Moonblast'] if name == 'x' else
                ['Soak', 'Toxic', 'Thunderbolt'] if name == 'y' else ['Tackle']}
    selector.engine.sampler.sample = sample
    worlds, tags = counterplay_worlds(selector.engine.sampler, picks)
    assert tags == ['special_setup', 'status_conversion']
    assert worlds[0][0]['moves'] == ['Calm Mind', 'Moonblast']
    assert worlds[1][0]['moves'] == ['Soak', 'Toxic', 'Thunderbolt']
    assert all(len(world) == 3 for world in worlds)
