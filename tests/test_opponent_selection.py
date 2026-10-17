import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from pokesim.live.opponent_selection import OpponentLineups, draw_unseen, pick_rows
from pokesim.live.selection import PickModel


def uniform(tmp_path):
    return PickModel(tmp_path / 'missing')


def row(i, shown, lead='a'):
    return {'record': str(i), 'preview': list('abcdef'), 'lead': [lead], 'shown': [(s,) for s in shown]}


def test_joint_lineups_learn_pairs_and_keep_consistent_marginals(tmp_path):
    base = uniform(tmp_path)
    rows = [row(i, 'abc') for i in range(30)] + [row(i + 30, 'ade') for i in range(30)]
    picks = OpponentLineups(rows, base).predict(base.predict(list('abcdef')))
    assert len(picks.lineups) == 60
    assert sum(p for _, p in picks.lineups) == pytest.approx(1)
    assert picks.brought.sum() == pytest.approx(3)
    assert picks.lead.sum() == pytest.approx(1)
    assert np.all(picks.lead <= picks.brought)
    together = sum(p for names, p in picks.lineups if set('bc') <= set(names))
    cross = sum(p for names, p in picks.lineups if set('be') <= set(names))
    assert together > cross * 1.5
    assert picks.lead[0] > .5


def test_unrevealed_third_member_is_censored_instead_of_four_negative_labels(tmp_path):
    base = uniform(tmp_path)
    rows = [row(i, 'ab') for i in range(30)]
    picks = OpponentLineups(rows, base).predict(base.predict(list('abcdef')))
    assert picks.brought[1] > picks.brought[2]
    assert picks.brought[2] == pytest.approx(picks.brought[3])
    assert picks.brought[3] == pytest.approx(picks.brought[4])
    assert picks.brought[4] == pytest.approx(picks.brought[5])
    assert min(picks.brought[2:]) > .15


def test_sampling_conditions_on_all_seen_members_and_keeps_a_coherent_trio(tmp_path):
    picks = uniform(tmp_path).predict(list('abcdef'))
    picks.lineups = [(['a', 'b', 'c'], .99), (['a', 'd', 'e'], .01)]
    for seed in range(10):
        assert set(draw_unseen(picks, ['d'], random.Random(seed))) == {'a', 'e'}
    assert draw_unseen(picks, ['c', 'd'], random.Random(0)) is None


def test_only_this_visible_rosters_completed_rated_records_enter_and_illusion_corrects_lead(tmp_path):
    roster = list('uvwxyz')
    rec = {'source': 'showdown battle-1', 'team': roster, 'result': 'loss',
           'preview': {'species': ['Starmie', 'Zoroark-Hisui', 'Garchomp', 'Gengar', 'Scizor', 'Tyranitar']},
           'log': ['|rated|', '|switch|p2a: Starmie|Starmie, L50|100/100',
                   '|replace|p2a: Zoroark|Zoroark-Hisui, L50',
                   '|switch|p2a: Garchomp|Garchomp, L50|100/100']}
    (tmp_path / '001.json').write_text(json.dumps(rec))
    (tmp_path / '002.json').write_text(json.dumps(rec))  # duplicate battle, even with a different file name
    (tmp_path / '003.json').write_text(json.dumps({**rec, 'source': 'other', 'team': list('abcdef')}))
    (tmp_path / '004.json').write_text(json.dumps({**rec, 'source': 'local', 'log': []}))
    (tmp_path / '005.json').write_text(json.dumps({**rec, 'source': 'unfinished', 'result': None}))
    rows = pick_rows(tmp_path, roster)
    assert len(rows) == 1 and rows[0]['lead'] == ['zoroarkhisui']
    assert ('starmie',) not in rows[0]['shown']
    base = uniform(tmp_path)
    picks = base.predict(rec['preview']['species'], roster=roster, records=tmp_path)
    assert picks.context['records'] == 1 and picks.context['censored_labels']


def test_live_search_uses_conditional_lineups_to_fill_the_hidden_bench():
    from test_worker_transitions import CHOMP

    from pokesim.live.session import BattleSession
    from pokesim.search.lookahead import Lookahead

    preview = ['Garchomp', 'Gengar', 'Scizor', 'Tyranitar', 'Rotom-Wash', 'Dragonite']
    session = BattleSession([CHOMP])
    session.preview(preview)
    session.select([0])
    session.apply(['|switch|p1a: Garchomp|Garchomp, L50|200/200',
                   '|switch|p2a: Rotom|Rotom-Wash, L50|100/100', '|turn|1'])
    engine = object.__new__(Lookahead)
    engine.sampler = SimpleNamespace(base_id=lambda sid: sid, rng=random.Random(1),
                                     sample=lambda sid, *a: dict(CHOMP, species=sid))
    picks = SimpleNamespace(lineups=[(['Garchomp', 'Gengar', 'Scizor'], .99),
                                     (['Rotom-Wash', 'Tyranitar', 'Dragonite'], .01)])
    engine.set_opponent_picks(session.battle.battle_tag, picks)
    side = engine.opponent_side(session.decision())
    assert {s['species'] for s in side['team']} == {'rotomwash', 'tyranitar', 'dragonite'}


def test_preview_risk_uses_joint_trios_even_when_each_species_marginal_is_identical(tmp_path):
    from pokesim.live.preview_matchups import CoverageProfile

    profile = CoverageProfile(['x', 'y', 'z'], list('abcdef'), [
        dict(ours=i, foe=s, mega=False, outgoing=.1 if s in 'ab' else .8,
             incoming=1.1 if s in 'ab' else .3, faster=0) for i in range(3) for s in 'abcdef'])
    together = uniform(tmp_path).predict(list('abcdef'))
    split = uniform(tmp_path).predict(list('abcdef'))
    together.brought[:] = split.brought[:] = .5
    together.lead[:] = split.lead[:] = 1 / 6
    together.lineups = [([*'abc'], .5), ([*'def'], .5)]
    split.lineups = [([*'ade'], .5), ([*'bcf'], .5)]
    assert profile.score(['x', 'y', 'z'], together)[0] > profile.score(['x', 'y', 'z'], split)[0]
