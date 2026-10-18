"""Preview-only matchup coverage for the entire trio, with one Mega allocation at a time.

The damage estimates are static comparisons of sampled sets, not simulated win probabilities. They complement
the roster's sparse historical results; a missing profile leaves the historical selector usable.
"""
import hashlib
import json
import random
from collections import OrderedDict

import numpy as np

from pokesim.rl.myteam import to_id

VERSION = 'preview-coverage-v2'


class PreviewCoverage:
    def __init__(self, sets, engine, samples=3):
        self.sets, self.engine, self.samples = sets, engine, samples
        self.roster = [to_id(s['species']) for s in sets]
        self.cache = OrderedDict()

    def profile(self, picks):
        species = sorted(set(map(to_id, picks.species)))
        signature = json.dumps([self.sets, species, self.samples,
                                getattr(getattr(self.engine.sampler, 'catalog', None), 'sha', None)], sort_keys=True)
        key = hashlib.sha256(signature.encode()).hexdigest()[:16]
        if key not in self.cache:
            sampler = self.engine.sampler
            old_rng = sampler.rng
            try:
                sampler.rng = random.Random(int(key, 16))
                opponents = [{'species': sid, 'sets': [sampler.sample(sid, preview=species)
                              if getattr(sampler, 'catalog', None) is not None else sampler.sample(sid)
                              for _ in range(self.samples)]}
                             for sid in species]
            finally:
                sampler.rng = old_rng
            # A weather setter shown in preview makes that weather a possible joint context for its teammates.
            weather = ['']
            for opponent in opponents:
                for s in opponent['sets']:
                    w = {'drizzle': 'raindance', 'drought': 'sunnyday', 'sandstream': 'sandstorm',
                         'snowwarning': 'snowscape'}.get(to_id(s.get('ability', '')))
                    if w and w not in weather:
                        weather.append(w)
            reply = self.engine._query({'format': 'gen9championsbssregmc', 'previewMatchups': {
                'ours': self.sets, 'opponents': opponents, 'weather': weather}})
            self.cache[key] = CoverageProfile(self.roster, species, reply['matchups'], key)
            while len(self.cache) > 64:
                self.cache.popitem(last=False)
        return self.cache[key]


class CoverageProfile:
    def __init__(self, roster, species, rows, signature=''):
        self.roster, self.species, self.signature = roster, species, signature
        self.cells = {(r['ours'], r['foe'], r['mega']): r for r in rows}
        if any((i, s, False) not in self.cells for i in range(len(roster)) for s in species):
            raise ValueError('incomplete preview matchup estimates')
        if any(not np.isfinite([r['outgoing'], r['incoming'], r['faster']]).all() for r in rows):
            raise ValueError('nonfinite preview matchup estimates')

    def score(self, lineup, picks):
        ids = [self.roster.index(to_id(s)) for s in lineup]
        pick_ids = list(map(to_id, picks.species))
        brought = np.array([picks.brought[pick_ids.index(s)] for s in self.species], float)
        lead = np.array([picks.lead[pick_ids.index(s)] for s in self.species], float)
        brought /= brought.sum()
        lead /= lead.sum()
        allocations = [None] + [i for i in ids if all((i, s, True) in self.cells for s in self.species)]
        scores = []
        for mega in allocations:
            cells = [[self.cells[i, s, i == mega] for s in self.species] for i in ids]
            outgoing = np.array([[r['outgoing'] for r in row] for row in cells])
            incoming = np.array([[r['incoming'] for r in row] for row in cells])
            faster = np.array([[r['faster'] for r in row] for row in cells])
            duel = np.minimum(outgoing, 1.25) - np.minimum(incoming, 1.5) + .15 * faster * (outgoing >= .2)
            # A passive passer alone cannot answer a threat merely by resisting its first hit.
            duel = np.where(outgoing > .05, duel, np.minimum(duel, -.75))
            cover = duel.max(axis=0)
            worst = float(cover.min())
            # Opponents bring three together. Use their joint selection's weakest matchup, while retaining
            # a 30% guard for an unexpected selection. Without joint predictions retain the old all-six guard.
            if getattr(picks, 'lineups', None):
                index = {s: j for j, s in enumerate(self.species)}
                expected_worst = sum(p * float(min(cover[index[to_id(s)]] for s in names))
                                     for names, p in picks.lineups)
                worst = .7 * expected_worst + .3 * worst
            score = .6 * float(cover @ brought) + .4 * worst + .25 * float(duel[0] @ lead)
            scores.append((score, mega, cover, incoming, worst))
        score, mega, cover, incoming, worst = max(scores, key=lambda x: x[0])
        return score, {'version': VERSION, 'profile': self.signature, 'logit_adjustment': round(score, 4),
                       'mega': self.roster[mega] if mega is not None else None,
                       'selection_worst': round(worst, 4),
                       'uncovered': [s for s, value in zip(self.species, cover) if value < -.25],
                       'minimum_incoming': {s: round(float(v), 3) for s, v in zip(self.species, incoming.min(axis=0))}}
