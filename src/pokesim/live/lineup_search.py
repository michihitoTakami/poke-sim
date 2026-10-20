"""Shortlist static preview candidates, then rank real public-policy three-on-three continuations."""
import hashlib
import itertools
import json
import random
from collections import OrderedDict

import numpy as np

from pokesim.config import FORMAT
from pokesim.rl.myteam import to_id

VERSION = 'preview-adversarial-v7'
POLICIES = ['counterplay', 'disrupt', 'expert']


def counterplay_worlds(sampler, picks, limit=4):
    """Stress hypotheses use complete sampled sets, never the opponent's later actual moves.

    Keep their prior weight out of the expected score; they contribute only to downside robustness.
    """
    catalog = getattr(sampler, 'catalog', None)
    draws = []
    for _ in range(4):
        joint = catalog.draw(picks.species, {}, sampler.rng) if catalog is not None else {}
        draws.append([joint.get(to_id(s)) or sampler.sample(to_id(s)) for s in picks.species])
    worlds, tags = [], []
    for kind in ('disruption', 'physical_setup', 'special_setup', 'status_conversion'):
        def strength(mon):
            moves = set(map(to_id, mon.get('moves', [])))
            if kind == 'disruption':
                return len(moves & {'taunt', 'haze', 'roar', 'whirlwind', 'encore', 'clearsmog'})
            if kind == 'physical_setup':
                return len(moves & {'swordsdance', 'dragondance', 'tidyup', 'bulkup', 'shellsmash'})
            if kind == 'special_setup':
                return len(moves & {'calmmind', 'nastyplot', 'quiverdance', 'shellsmash'})
            return int('soak' in moves) + int('toxic' in moves)
        choices = [(strength(mon), team, mon) for team in draws for mon in team if strength(mon)]
        if not choices:
            continue
        _, team, lead = max(choices, key=lambda x: x[0])
        # The remaining members follow the preview-conditioned lineup prior, with this stress lead fixed.
        options = [(names, weight) for names, weight in picks.lineups or []
                   if to_id(lead['species']) in set(map(to_id, names))]
        selected = max(options, key=lambda x: x[1])[0] if options else [m['species'] for m in team[:3]]
        bench = [m for m in team if to_id(m['species']) != to_id(lead['species']) and
                 to_id(m['species']) in set(map(to_id, selected))][:2]
        if len(bench) < 2:
            bench = [m for m in team if to_id(m['species']) != to_id(lead['species'])][:2]
        if len(bench) == 2:
            worlds.append([lead, *bench])
            tags.append(kind)
        if len(worlds) == limit:
            break
    return worlds, tags


class LineupSearch:
    def __init__(self, sets, engine, *, shortlist=4, worlds=4, max_turns=160, reverse_worlds=2):
        self.sets, self.engine = sets, engine
        self.shortlist, self.worlds, self.max_turns = shortlist, worlds, max_turns
        self.reverse_worlds = reverse_worlds
        self.cache = OrderedDict()

    def rerank(self, ranking, picks):
        """Paired hidden worlds across candidates. Only the preview/prior and static shortlist are inputs."""
        pool, trios = [], set()
        # Keep the static shortlist's leads, then cover EVERY available trio at its best static lead.
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
        all_trios = {tuple(sorted(map(to_id, row[0]))) for row in pool}
        additional_trios = []
        for row in ranking:
            trio = tuple(sorted(map(to_id, row[0])))
            if trio not in all_trios:
                pool.append(row)
                included.add(tuple(map(to_id, row[0])))
                all_trios.add(trio)
                additional_trios.append(row[0])
        relay_candidates = []
        for row in ranking:
            names = tuple(map(to_id, row[0]))
            if names[0] in passers and names not in included:
                pool.append(row)
                included.add(names)
                relay_candidates.append(row[0])
        if len(pool) < 2:
            return ranking, {'version': VERSION, 'skipped': 'less-than-two-candidates'}
        signature = json.dumps([VERSION, POLICIES, getattr(getattr(self.engine.sampler, 'catalog', None), 'sha', None),
                                self.sets, [r[0] for r in pool], picks.species, picks.lineups,
                                picks.context, self.worlds, self.max_turns, self.reverse_worlds], sort_keys=True)
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
                worlds = []
                for trio in selected:
                    catalog = getattr(sampler, 'catalog', None)
                    joint = catalog.draw(picks.species, {}, sampler.rng) if catalog is not None else {}
                    worlds.append([joint.get(to_id(s)) or sampler.sample(to_id(s)) for s in trio])
                stress, tags = counterplay_worlds(sampler, picks)
                worlds.extend(stress)
                from pokesim.live.best_response import responses

                reverse, reverse_evidence = responses(self.sets, pool, picks, sampler, self.engine._query,
                                                      worlds=self.reverse_worlds)
                worlds.extend(reverse)
                tags.extend(['best_response'] * len(reverse))
            finally:
                sampler.rng = saved
            by_id = {to_id(s['species']): s for s in self.sets}
            reply = self.engine._query({'format': FORMAT, 'lineupMatches': {
                'ours': [[by_id[to_id(s)] for s in r[0]] for r in pool], 'worlds': worlds,
                'preview': {'p1': [s['species'] for s in self.sets], 'p2': picks.species},
                'maxTurns': self.max_turns, 'policies': POLICIES}})
            if reply.get('informationScope') != 'public_observation':
                raise ValueError('preview continuation must use public actor observations')
            reply['world_lineups'] = [[s['species'] for s in world] for world in worlds]
            reply['stress_tags'] = tags
            reply['reverse_selection'] = reverse_evidence
            self.cache[key] = reply
            while len(self.cache) > 64:
                self.cache.popitem(last=False)
        reply = self.cache[key]
        evidence, scored = [], []
        for i, row in enumerate(pool):
            trials = [t for t in reply['trials'] if t['candidate'] == i]
            pairs = [(t['world'], t['opponent']) for t in trials]
            expected = set(itertools.product(range(len(reply['world_lineups'])), POLICIES))
            if set(pairs) != expected or len(pairs) != len(expected):
                raise ValueError('missing or duplicate candidate/world/opponent continuations')
            nominal = [t for t in trials if t['world'] < self.worlds]
            known = sum(t['target'] for t in nominal if t['target'] is not None)
            unknown = sum(t['target'] is None for t in nominal)
            low, high = known / len(nominal), (known + unknown) / len(nominal)
            groups = {}
            for t in trials:
                groups.setdefault((t['world'], t.get('opponent')), []).append(t['target'])
            by_world = {}
            for (world, _), values in groups.items():
                by_world.setdefault(world, []).append(sum(v or 0 for v in values) / len(values))
            # A weak opponent policy must not average away another policy's successful counterplay.
            # Average the worst response per world, rather than the binary minimum of all games.
            worst_responses = {world: min(values) for world, values in by_world.items()}
            downside = sum(worst_responses.values()) / len(worst_responses)
            simulation = .7 * low + .3 * downside
            history = (row[2] or {}).get('history', {})
            # Empirical posterior is already shrunk/decayed. Missing history cannot become fake evidence.
            # Bound its influence: observational lineup results are not randomized counterfactual wins.
            ordered = max(0., float(history.get('ordered_games', 0)))
            trio = max(ordered, float(history.get('trio_games', 0)))
            evidence_games = ordered + .25 * (trio - ordered)
            history_weight = min(.35, evidence_games / (evidence_games + 40))
            empirical = float(history.get('mean', .5))
            if not np.isfinite(empirical) or not 0 <= empirical <= 1:
                raise ValueError('invalid empirical lineup posterior')
            utility = (1 - history_weight) * simulation + history_weight * empirical
            info = {'lineup': row[0], 'trials': len(trials),
                    'terminal': sum(t['target'] is not None for t in trials), 'downside': downside,
                    'worst_response_by_world': worst_responses,
                    'failed': sum(t['outcome'] == 'failed' for t in trials),
                    'target_bounds': [low, high], 'utility': utility,
                    'simulation_utility': simulation, 'history_mean': empirical,
                    'history_weight': history_weight, 'history_effective_games': evidence_games,
                    'relay': [t.get('relay', {}) for t in trials]}
            evidence.append(info)
            scored.append((row[0], utility, dict(row[2] or {}, continuation=info)))
        info = {'version': VERSION, 'profile': key[:16], 'policy': reply['version'],
                'score_kind': 'public-continuation-with-ladder-evidence', 'calibrated_win_probability': False,
                'candidates': evidence, 'worlds': len(reply['world_lineups']), 'nominal_worlds': self.worlds,
                'stress_worlds': reply['stress_tags'], 'opponent_hypotheses': reply['world_lineups'],
                'reverse_selection': reply['reverse_selection'],
                'additional_relay_candidates': relay_candidates,
                'additional_trio_candidates': additional_trios, 'policies': POLICIES,
                'unknown_outcomes': 'bounds-only; never training loss labels'}
        catalog = getattr(self.engine.sampler, 'catalog', None)
        if catalog is not None:
            info['team_catalog'] = catalog.info(picks.species)
        # A partial simulator failure must not silently change the selection. Keep the static ranking usable.
        if any(e['failed'] for e in evidence):
            return ranking, dict(info, fallback='failed-continuation')
        scored.sort(key=lambda r: r[1], reverse=True)
        return scored, info
