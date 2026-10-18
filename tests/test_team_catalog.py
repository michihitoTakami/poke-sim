import copy
import random

import numpy as np
import pytest

from pokesim.belief.team_catalog import TeamCatalog
from pokesim.config import FORMAT
from pokesim.data.team_catalog import digest
from pokesim.search.budget import decision_budget
from scripts.ladder import search_quality_issues


def catalog():
    teams = []
    for move, item, nature in [('Surf', 'Choice Scarf', 'modest'), ('Earthquake', 'Choice Band', 'adamant')]:
        teams.append({'id': move, 'sets': [dict(species=s, moves=[move, 'Protect', 'Toxic', 'Rest'],
            item=item if s == 'a' else '', nature=nature, ability='blaze', evs={'spe': 32})
            for s in 'abcdef']})
    return TeamCatalog({'schema_version': 1, 'format': FORMAT, 'battle_mode': 'single',
                        'teams': teams, 'teams_sha256': digest(teams)})


def test_team_members_come_from_one_sheet_instead_of_independent_frankenstein_sets():
    prior = catalog()
    rng = random.Random(2)
    samples = [prior.draw(list('abcdef'), {}, rng) for _ in range(100)]
    assert any(samples) and any(not s for s in samples)  # broad fallback remains supported
    for sample in filter(bool, samples):
        assert len({m['nature'] for m in sample.values()}) == 1
        assert len({m['moves'][0] for m in sample.values()}) == 1


def test_incompatible_reveals_reject_a_sheet_instead_of_patching_its_moves():
    prior = catalog()
    rng = random.Random(1)
    for _ in range(30):
        sample = prior.draw(list('abcdef'), {'a': {'moves': ['Surf'], 'item': 'choicescarf'}}, rng)
        assert not sample or sample['a']['moves'][0] == 'Surf'
    assert not prior.draw(list('abcdef'), {'a': {'moves': ['Surf', 'Earthquake']}}, rng)


def test_similar_teams_do_not_add_species_outside_the_actual_preview_and_need_four_matches():
    prior = catalog()
    assert prior.info(list('abcxyz'))['matched'] == 0
    assert prior.info(list('abcdxy'))['best_overlap'] == 4
    samples = [prior.draw(list('abcdxy'), {}, random.Random(i)) for i in range(12)]
    assert any(samples)
    assert all(set(s) <= set('abcdxy') for s in samples)


def test_consumed_items_and_mega_abilities_do_not_falsify_original_team_sheets():
    prior = catalog()
    mon = copy.deepcopy(prior.teams[0]['sets'][0])
    mon['item'] = 'Charizardite Y'
    prior.mega_abilities['charizarditey'] = ['drought']
    assert prior.compatible(mon, {'ability': 'drought', 'item': ''})
    assert not prior.compatible(mon, {'ability': 'levitate'})
    result = next(s for i in range(20) if (s := prior.draw_set('a', ['Surf'], '', None, random.Random(i))))
    assert result['item'] == '' and result['moves'][0] == 'Surf'
    assert prior.teams[0]['sets'][0]['item'] == 'Choice Scarf'


def test_reliable_speed_evidence_can_reject_all_catalog_sheets():
    from types import SimpleNamespace
    prior = catalog()
    belief = SimpleNamespace(weights=lambda sid, sets: np.ones(len(sets)) / len(sets),
                             relevant=lambda sid: [],
                             speed_compatible=lambda obs, sets: np.zeros(len(sets), bool))
    assert not prior.draw(list('abcdef'), {'a': {}}, random.Random(1), belief)


def test_content_or_format_mismatches_are_rejected():
    with pytest.raises(ValueError, match='mismatch'):
        TeamCatalog({'schema_version': 1, 'format': FORMAT, 'battle_mode': 'double',
                     'teams': [], 'teams_sha256': digest([])})


@pytest.mark.parametrize('clock,expected', [('', 15000),
    ('|inactive|Time left: 55 sec this turn | 420 sec total', 15000),
    ('|inactive|Time left: 55 sec this turn | 55 sec total', 5000),
    ('|inactive|Time left: 8 sec this turn | 420 sec total', 3000),
    ('|inactive|Time left: 3 sec this turn | 8 sec total', 1000)])
def test_longer_search_respects_the_observed_clock(clock, expected):
    assert decision_budget(15000, [clock]) == expected


def test_quality_gate_uses_the_declared_allowance_without_accepting_unbounded_delays():
    healthy = {'decision_budget_ms': 15000, 'search': {'terminal_search': {'failed': 0}}}
    assert search_quality_issues(healthy, 15.2) == []
    assert search_quality_issues(healthy, 17.) == ['decision exceeded 16.5 seconds']
    assert search_quality_issues({'decision_budget_ms': 999999}, 22.) == ['decision exceeded 21.5 seconds']


def test_importer_resolves_internal_ids_through_masters_and_rejects_incomplete_sheets():
    from types import SimpleNamespace

    from pokesim.data.team_catalog import STATS, decode
    from pokesim.lookup.names import NameIndex

    species, master, pokemon = {}, [], []
    for i in range(6):
        sid = f'mon{i}'
        stats = [80 + i, 80, 80, 80, 80, 80]
        species[sid] = SimpleNamespace(num=i + 1, name=sid, id=sid, is_mega=False,
                                      battle_only=None, base_stats=dict(zip(STATS, stats)), abilities=['pressure'])
        # Internal master ids deliberately differ from national numbers.
        master.append({'id': str(i + 500), 'n': sid, 'i': str(i + 1), 'bs': stats})
        pokemon.append({'id': str(i + 500), 'i': 0, 'a': 1, 'mo': [1, 2, 3, 4],
                        'e': [2, 0, 0, 32, 0, 32], 'na': 14})
    index = NameIndex([('move', str(i), 'en', f'move{i}') for i in range(1, 5)] +
                      [('ability', 'pressure', 'en', 'Pressure')])
    masters = {'pokemon': master, 'move': [{'id': str(i), 'n': f'move{i}'} for i in range(1, 5)],
               'ability': [{'id': '1', 'n': 'Pressure'}], 'item': [{'id': '0', 'n': 'なし'}]}
    source = {'po': pokemon, 'r': '2', 's': 'M-6'}
    teams, rejected = decode([source], masters, index, species, {})
    assert not rejected and len(teams) == 1
    assert teams[0]['sets'][0]['species'] == 'mon0'
    assert teams[0]['sets'][0]['nature'] == 'modest'
    bad = copy.deepcopy(source)
    bad['po'][0]['e'][0] = 33
    teams, rejected = decode([bad], masters, index, species, {})
    assert not teams and 'spread' in rejected[0]['reason']


def test_unmatched_previews_still_allow_complete_species_sets_without_a_team_identity_claim():
    prior = catalog()
    samples = [prior.draw_set('a', ['Surf'], None, None, random.Random(i), preview=list('axyzuv'))
               for i in range(20)]
    assert any(samples) and any(s is None for s in samples)
    assert all(s is None or (s['nature'] == 'modest' and s['item'] == 'Choice Scarf') for s in samples)
    assert prior.info(list('axyzuv'))['matched'] == 0
