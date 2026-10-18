"""Shortlist static preview candidates, then rank real public-policy three-on-three continuations."""
import hashlib
import itertools
import json
import random
from collections import OrderedDict

import numpy as np

from pokesim.config import FORMAT
from pokesim.rl.myteam import to_id

VERSION = 'preview-continuations-v2'


class LineupSearch:
    def __init__(self, sets, engine, *, shortlist=4, worlds=4, max_turns=160):
        self.sets, self.engine = sets, engine
        self.shortlist, self.worlds, self.max_turns = shortlist, worlds, max_turns
        self.cache = OrderedDict()

    def rerank(self, ranking, picks):
        """Paired hidden worlds across candidates. Only the preview/prior and static shortlist are inputs."""
        pool, trios = [], set()
        # Preserve the best two leads; also compare different actual trios, rather than only changing the lead.
        for row in ranking:
            key = tuple(sorted(map(to_id, row[0])))
            if len(pool) < 2 or key not in trios:
                pool.append(row)
                trios.add(key)
            if len(pool) == self.shortlist:
                break
        # Static one-on-one coverage undervalues a passive passer before any payload exists.
        # Give every passer-led trio an actual full-game comparison, including two-hop relays.
        # Only known own moves are consulted; no speculative boost bonus enters the final score.
        passers = {to_id(s['species']) for s in self.sets if 'batonpass' in map(to_id, s.get('moves', []))}
        included = {tuple(map(to_id, row[0])) for row in pool}
        relay_candidates = []
        for row in ranking:
            names = tuple(map(to_id, row[0]))
            if names[0] in passers and names not in included:
                pool.append(row)
                included.add(names)
                relay_candidates.append(row[0])
        if len(pool) < 2:
            return ranking, {'version': VERSION, 'skipped': 'less-than-two-candidates'}
        signature = json.dumps([self.sets, [r[0] for r in pool], picks.species, picks.lineups,
                                picks.context, self.worlds, self.max_turns], sort_keys=True)
        key = hashlib.sha256(signature.encode()).hexdigest()
        if key not in self.cache:
            sampler, saved = self.engine.sampler, self.engine.sampler.rng
            try:
                sampler.rng = random.Random(int(key[:16], 16))
                options = picks.lineups or [(list(t), 1.) for t in itertools.permutations(picks.species, 3)]
                names, weights = zip(*options)
                cumulative = np.cumsum(weights, dtype=float) / sum(weights)
                offset = sampler.rng.random()
                selected = [names[min(len(names) - 1, int(np.searchsorted(cumulative, (i + offset) / self.worlds)))]
                            for i in range(self.worlds)]
                worlds = [[sampler.sample(to_id(s)) for s in trio] for trio in selected]
            finally:
                sampler.rng = saved
            by_id = {to_id(s['species']): s for s in self.sets}
            reply = self.engine._query({'format': FORMAT, 'lineupMatches': {
                'ours': [[by_id[to_id(s)] for s in r[0]] for r in pool], 'worlds': worlds,
                'preview': {'p1': [s['species'] for s in self.sets], 'p2': picks.species},
                'maxTurns': self.max_turns}})
            if reply.get('informationScope') != 'public_observation':
                raise ValueError('preview continuation must use public actor observations')
            reply['world_lineups'] = [[s['species'] for s in world] for world in worlds]
            self.cache[key] = reply
            while len(self.cache) > 64:
                self.cache.popitem(last=False)
        reply = self.cache[key]
        evidence, scored = [], []
        for i, row in enumerate(pool):
            trials = [t for t in reply['trials'] if t['candidate'] == i]
            if not trials:
                raise ValueError('missing candidate continuations')
            known = sum(t['target'] for t in trials if t['target'] is not None)
            unknown = sum(t['target'] is None for t in trials)
            low, high = known / len(trials), (known + unknown) / len(trials)
            by_world = [sum(t['target'] or 0 for t in trials if t['world'] == w) /
                        sum(t['world'] == w for t in trials) for w in range(self.worlds)]
            utility = .7 * low + .3 * min(by_world)
            info = {'lineup': row[0], 'trials': len(trials), 'terminal': len(trials) - unknown,
                    'failed': sum(t['outcome'] == 'failed' for t in trials),
                    'target_bounds': [low, high], 'utility': utility,
                    'relay': [t.get('relay', {}) for t in trials]}
            evidence.append(info)
            scored.append((row[0], utility, dict(row[2] or {}, continuation=info)))
        info = {'version': VERSION, 'profile': key[:16], 'policy': reply['version'],
                'score_kind': 'sampled-public-continuation-utility', 'calibrated_win_probability': False,
                'candidates': evidence, 'worlds': self.worlds, 'opponent_hypotheses': reply['world_lineups'],
                'additional_relay_candidates': relay_candidates,
                'unknown_outcomes': 'bounds-only; never training loss labels'}
        # A partial simulator failure must not silently change the selection. Keep the static ranking usable.
        if any(e['failed'] for e in evidence):
            return ranking, dict(info, fallback='failed-continuation')
        scored.sort(key=lambda r: r[1], reverse=True)
        return scored, info
