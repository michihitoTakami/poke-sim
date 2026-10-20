"""Search hypothetical opponent selections against several of our possible selections, before we pick.

All 60 opponent lead/trio choices are compared on one coherent sampled six at a time. The opponent actor
receives only its private request and public observations. No actual hidden opponent team or our final
private selection is an input. These stress worlds are kept out of the nominal prior expectation.
"""
import hashlib
import itertools
import json

from pokesim.config import FORMAT
from pokesim.rl.myteam import to_id

VERSION = 'preview-best-response-v1'


def responses(sets, pool, picks, sampler, query, worlds=2):
    by_id = {to_id(s['species']): s for s in sets}
    passers = {sid for sid, s in by_id.items() if 'batonpass' in map(to_id, s.get('moves', []))}
    # The adversary faces a mixture of plausible own selections, including a relay. It never gets our pick.
    refs, seen, trios = [], set(), set()
    relay = next((r for r in pool if to_id(r[0][0]) in passers), None)
    for row in [pool[0], *([relay] if relay is not None else []), *pool]:
        key = tuple(map(to_id, row[0]))
        trio = tuple(sorted(key))
        if key not in seen and (len(refs) < 2 or trio not in trios):
            refs.append([by_id[sid] for sid in key])
            seen.add(key)
            trios.add(trio)
        if len(refs) == 3:
            break
    selected, evidence = [], []
    for draw in range(worlds):
        catalog = getattr(sampler, 'catalog', None)
        joint = catalog.draw(picks.species, {}, sampler.rng) if catalog is not None else {}
        six = [joint.get(to_id(s)) or sampler.sample(to_id(s)) for s in picks.species]
        options = [[six[lead], *(six[i] for i in bench)] for lead in range(len(six))
                   for bench in itertools.combinations([i for i in range(len(six)) if i != lead], 2)]
        reply = query({'format': FORMAT, 'lineupMatches': {
            'ours': options, 'worlds': refs, 'oursPolicy': 'counterplay', 'policies': ['planned'],
            'preview': {'p1': picks.species, 'p2': [s['species'] for s in sets]}, 'maxTurns': 120}})
        if reply.get('informationScope') != 'public_observation':
            raise ValueError('opponent selection search must use public actors')
        scores = []
        for i, lineup in enumerate(options):
            trials = [t for t in reply['trials'] if t['candidate'] == i]
            keys = [(t['world'], t['opponent']) for t in trials]
            if set(keys) != {(w, 'planned') for w in range(len(refs))} or len(keys) != len(refs):
                raise ValueError('missing or duplicate reverse selection comparison')
            if any(t['outcome'] == 'failed' for t in trials):
                raise ValueError('failed reverse selection comparison')
            known = sum(t['target'] for t in trials if t['target'] is not None)
            unknown = sum(t['target'] is None for t in trials)
            scores.append({'lineup': [s['species'] for s in lineup], 'candidate': i,
                           'bounds': [known / len(refs), (known + unknown) / len(refs)],
                           'completed': len(refs) - unknown})
        best = max(scores, key=lambda r: (r['bounds'][0], r['completed']))
        selected.append(options[best['candidate']])
        evidence.append({'version': VERSION, 'draw': draw, 'candidates': len(options),
                         'reference_lineups': [[s['species'] for s in team] for team in refs],
                         'sample_sha': hashlib.sha256(json.dumps(six, sort_keys=True).encode()).hexdigest()[:16],
                         'selected': best, 'trials': len(reply['trials']), 'policy': reply['version'],
                         'scope': 'hypothetical best response to a mixture, not knowledge of our chosen trio'})
    return selected, evidence
