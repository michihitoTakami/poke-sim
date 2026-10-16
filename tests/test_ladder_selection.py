import json
from types import SimpleNamespace

import numpy as np

from pokesim.live.selection import LADDER_PRIOR, MatchupModel

ROSTER = ['Scolipede', 'Umbreon', 'Gholdengo', 'Salamence']


def test_live_evidence_demotes_failing_trios_and_preserves_matchup_information(tmp_path):
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'lineups': {
        'scolipede,umbreon,gholdengo': {'bias': 1.386294, 'effects': {}},  # 80% simulated
        'scolipede,gholdengo,salamence': {'bias': .619039, 'effects': {}},  # 65%
    }}))
    model = MatchupModel(path)
    picks = SimpleNamespace(species=[], brought=np.array([]))
    assert model.rank(ROSTER, picks)[0][0] == ROSTER[:3]
    for i in range(15):
        rec = dict(source=f'showdown battle-{i}', result='win' if i < 2 else 'loss', team=ROSTER,
                   lineup=ROSTER[:3], log=['|rated|'])
        (tmp_path / f'{i:03}.json').write_text(json.dumps(rec))
    # A local or unrated win must not undo the rated evidence.
    (tmp_path / '999.json').write_text(json.dumps(dict(rec, result='win', source='local', log=[])))
    # A duplicate record for the same battle counts once.
    (tmp_path / '998.json').write_text(json.dumps(rec))
    (tmp_path / 'unfinished.json').write_text('{')
    model.observe_ladder(tmp_path, ROSTER)
    ranked = model.rank(ROSTER, picks)
    assert ranked[0][0] == [ROSTER[0], ROSTER[2], ROSTER[3]]
    assert model.meta['ladder_evidence']['games'] == 15
    assert abs(ranked[1][1] - (LADDER_PRIOR * .8 + 2) / (LADDER_PRIOR + 15)) < 1e-6
    assert model.rank(ROSTER, picks) == ranked


def test_evidence_for_a_different_roster_is_ignored(tmp_path):
    model = MatchupModel(tmp_path / 'missing')
    rec = dict(source='showdown battle-1', result='win', team=['Other'] * 6,
               lineup=ROSTER[:3], log=['|rated|'])
    (tmp_path / 'record.json').write_text(json.dumps(rec))
    model.observe_ladder(tmp_path, ROSTER)
    assert not model.evidence


def ladder_record(i, lineup, result, preview=('garchomp',) * 6):
    return dict(source=f'showdown battle-{i}', started=f'20261016-{i:06d}', result=result, team=ROSTER,
                lineup=lineup, log=['|rated|'], preview={'species': list(preview)})


def test_ladder_posterior_separates_leads_decays_and_falls_back_to_the_trio(tmp_path):
    from pokesim.live.selection import LINEUP_KAPPA, LadderLineupModel, ladder_rows

    lead_a, lead_b = [ROSTER[0], ROSTER[1], ROSTER[2]], [ROSTER[1], ROSTER[0], ROSTER[2]]
    records = [ladder_record(i, lead_a, 'win') for i in range(10)] + \
              [ladder_record(10 + i, lead_b, 'loss') for i in range(10)]
    for i, rec in enumerate(records):
        (tmp_path / f'{i:03}.json').write_text(json.dumps(rec))
    (tmp_path / 'local.json').write_text(json.dumps(dict(records[0], source='local', log=[])))
    rows = ladder_rows(tmp_path, ROSTER)
    assert len(rows) == 20 and rows[0]['name'] == '019.json'  # most recent first, local games ignored
    model = LadderLineupModel(rows, half_life=1e9)
    assert model.mean(lead_a) > .8 > .2 > model.mean(lead_b)
    # Same trio, a lead never played: only the trio's pooled evidence (10-10) and the roster rate speak.
    other = [ROSTER[2], ROSTER[0], ROSTER[1]]
    assert abs(model.mean(other) - .5) < .05
    assert model.evidence(other) == {'ordered_games': 0, 'trio_games': 20}
    # An unseen trio sits at the roster's overall rate with kappa pseudo-games.
    a, b = model.posterior([ROSTER[0], ROSTER[1], ROSTER[3]])
    assert abs(a + b - LINEUP_KAPPA) < 1e-9 and abs(a / (a + b) - model.global_rate) < 1e-9
    # Recent losses with lead_b outweigh older wins with lead_a when evidence decays quickly.
    quick = LadderLineupModel(rows, half_life=2)
    assert quick.global_rate < .35


def test_thompson_choice_is_reproducible_and_stays_near_the_best_posterior(tmp_path):
    import random

    from pokesim.live.selection import LINEUP_MARGIN, LadderLineupModel

    good = [ROSTER[0], ROSTER[1], ROSTER[2]]
    rows = [dict(name=str(i), started='', win=i % 4 != 0,
                 ordered=(good[0].lower(), *sorted(x.lower() for x in good[1:])),
                 trio=tuple(sorted(x.lower() for x in good)), preview=[]) for i in range(40)]
    model = LadderLineupModel(rows, half_life=1e9)
    picks = [model.choose(ROSTER, random.Random(7))[0] for _ in range(3)]
    assert picks[0] == picks[1] == picks[2]
    lineup, info = model.choose(ROSTER, random.Random(7))
    assert info['method'] == 'ladder-thompson' and info['pool'] >= 1 and info['records'] == 40
    assert info['posterior_mean'] >= info['best_mean'] - LINEUP_MARGIN
    best, greedy = model.choose(ROSTER, random.Random(0), explore=False)
    assert best == good and greedy['method'] == 'ladder-mean'
    empty = LadderLineupModel([])
    lineup, info = empty.choose(ROSTER, random.Random(1), explore=False)
    assert len(lineup) == 3 and info['games'] == 0
