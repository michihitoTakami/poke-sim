import random
from types import SimpleNamespace

import numpy as np
import pytest

from pokesim.live.preview_matchups import CoverageProfile, PreviewCoverage
from pokesim.live.selection import LadderLineupModel, PickModel, PreviewLogistic, ladder_rows


def test_a_counter_on_the_bench_changes_selection_even_with_an_unseen_lead():
    roster = ['a', 'b', 'c', 'counter']
    picks = PickModel().predict(['foe'])
    rows = [{'ours': i, 'foe': 'foe', 'mega': False, 'outgoing': .8 if i == 3 else .2,
             'incoming': .2 if i == 3 else 1.2, 'faster': 0} for i in range(4)]
    profile = CoverageProfile(roster, ['foe'], rows)
    model = LadderLineupModel([])
    old, _ = model.choose(roster, random.Random(0), picks=picks, explore=False)
    new, info = model.choose(roster, random.Random(0), picks=picks, explore=False, coverage=profile)
    assert old == ['a', 'b', 'c']
    assert 'counter' in new and not info['coverage']['uncovered']
    assert model.rank(roster, picks, profile)[0][0] == new


def test_both_mega_stones_cannot_cover_different_threats_at_once():
    roster, species = ['mence', 'floette', 'passer'], ['physical', 'special']
    rows = []
    for i in range(3):
        for j, s in enumerate(species):
            for mega in (False, True) if i < 2 else (False,):
                good = mega and i == j
                rows.append(dict(ours=i, foe=s, mega=mega, outgoing=1 if good else .1,
                                 incoming=.1 if good else 1.5, faster=0))
    profile = CoverageProfile(roster, species, rows)
    score, info = profile.score(roster, PickModel().predict(species))
    assert info['mega'] in {'mence', 'floette'}
    assert len(info['uncovered']) == 1 and score < 0


def test_a_passive_passer_is_not_an_offensive_answer_just_because_it_resists():
    profile = CoverageProfile(['passer', 'a', 'b'], ['foe'], [
        dict(ours=i, foe='foe', mega=False, outgoing=0 if i == 0 else .1,
             incoming=0 if i == 0 else 1.5, faster=0) for i in range(3)])
    _, info = profile.score(['passer', 'a', 'b'], PickModel().predict(['foe']))
    assert info['uncovered'] == ['foe']


def test_preview_interactions_learn_the_other_two_members_as_well_as_the_lead():
    rows = [dict(ordered=('lead', 'safe', 'x'), trio=('lead', 'safe', 'x'), preview=['water'], win=True)] * 20
    rows += [dict(ordered=('lead', 'weak', 'x'), trio=('lead', 'weak', 'x'), preview=['water'], win=False)] * 20
    model = PreviewLogistic(['water'], ['lead', 'safe', 'weak', 'x'])
    model.fit(rows, [0.] * len(rows))
    good = model.adjust('lead', ['water'], ['lead', 'safe', 'x'])
    bad = model.adjust('lead', ['water'], ['lead', 'weak', 'x'])
    assert good > bad + .05


def test_roster_mismatch_is_ignored_by_selection_training(tmp_path):
    import json

    from test_ladder_selection import ROSTER, ladder_record

    rec = ladder_record(1, ROSTER[:3], 'win')
    rec['log'].append('|switch|p1a: Surprise|Salamence, L50|100/100')
    (tmp_path / 'bad.json').write_text(json.dumps(rec))
    assert ladder_rows(tmp_path, ROSTER) == []


def test_profiles_use_only_preview_and_restore_the_shared_sampler_rng():
    rng = random.Random(9)
    calls = []
    sampler = SimpleNamespace(rng=rng, sample=lambda s: dict(species=s, ability='Drizzle' if s == 'pelipper' else ''))

    def query(req):
        calls.append(req)
        return {'matchups': [dict(ours=0, foe=s, mega=False, outgoing=.5, incoming=.5, faster=0)
                              for s in ['archaludon', 'pelipper']]}

    engine = SimpleNamespace(sampler=sampler, _query=query)
    coverage = PreviewCoverage([{'species': 'ours'}], engine)
    before = rng.getstate()
    a = coverage.profile(PickModel().predict(['pelipper', 'archaludon']))
    b = coverage.profile(PickModel().predict(['archaludon', 'pelipper']))
    assert a is b and len(calls) == 1 and sampler.rng is rng and rng.getstate() == before
    assert calls[0]['previewMatchups']['weather'] == ['', 'raindance']
    assert 'log' not in calls[0] and 'result' not in calls[0]


def test_the_simulator_estimate_applies_aerilate_before_checking_ghost_immunity():
    from test_expert_continuation import CHOMP, DIRGE

    from pokesim.search.lookahead import Lookahead

    ours = dict(CHOMP, species='Salamence', ability='Intimidate', item='Salamencite', moves=['Double-Edge'])
    foe = dict(DIRGE, moves=['Splash'])
    engine = Lookahead(None)
    try:
        rows = engine._query({'format': 'gen9championsbssregmc', 'previewMatchups': {
            'ours': [ours], 'opponents': [{'species': 'skeledirge', 'sets': [foe]}]}})['matchups']
    finally:
        engine.close()
    assert rows[0]['outgoing'] == 0 and rows[1]['outgoing'] > 0


def test_preview_does_not_score_damage_into_an_absorption_ability():
    from test_expert_continuation import DIRGE, GHOLDENGO

    from pokesim.search.lookahead import Lookahead

    ours = dict(GHOLDENGO, species='Primarina', ability='Torrent', moves=['Surf'], item='')
    foe = dict(DIRGE, species='Vaporeon', ability='Water Absorb', moves=['Splash'])
    engine = Lookahead(None)
    try:
        rows = engine._query({'format': 'gen9championsbssregmc', 'previewMatchups': {
            'ours': [ours], 'opponents': [{'species': 'vaporeon', 'sets': [foe]}]}})['matchups']
    finally:
        engine.close()
    assert rows[0]['outgoing'] == 0


def test_incomplete_or_nonfinite_estimates_cannot_silently_enter_ranking():
    with pytest.raises(ValueError, match='incomplete'):
        CoverageProfile(['a'], ['foe'], [])
    with pytest.raises(ValueError, match='nonfinite'):
        CoverageProfile(['a'], ['foe'], [dict(ours=0, foe='foe', mega=False,
                                             outgoing=np.nan, incoming=0., faster=0)])


def test_api_preview_uses_the_same_trio_ranking_when_the_old_table_is_empty(monkeypatch):
    import asyncio

    from pokesim.api import server

    roster = ['a', 'b', 'c', 'counter']
    model = LadderLineupModel([])
    profile = CoverageProfile(roster, ['foe'], [
        dict(ours=i, foe='foe', mega=False, outgoing=.8 if i == 3 else .2,
             incoming=.2 if i == 3 else 1.2, faster=0) for i in range(4)])
    live = object.__new__(server.Live)
    live.roster, live.sets, live.ja = roster, [], {}
    live.recognized = None
    live.reset = lambda: None
    live.name = lambda sid: sid
    live.picks_model = PickModel()
    live.matchups = None
    live.preview_coverage = SimpleNamespace(profile=lambda _: profile)

    async def broadcast():
        return None

    live.broadcast = broadcast
    monkeypatch.setattr(server.LadderLineupModel, 'from_records', lambda *_: model)
    monkeypatch.setattr(server, 'BattleSession', lambda _: SimpleNamespace(preview=lambda _: None, info=[]))
    monkeypatch.setattr(server, 'EventTranslator', lambda _: None)
    monkeypatch.setattr(server.Vocab, 'build', lambda *_: None)
    monkeypatch.setattr(server, 'ScreenTracker', lambda *_, **kw: None)
    result = asyncio.run(live.start_preview(['foe']))
    assert 'counter' in result['lineups'][0]['lineup']
    assert result['lineups'][0]['win'] is None and 'score' in result['lineups'][0]
