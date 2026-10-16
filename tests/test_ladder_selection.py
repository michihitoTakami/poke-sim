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
