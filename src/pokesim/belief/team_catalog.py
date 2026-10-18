"""Joint public team-sheet hypotheses. Similar previews suggest sets; they never identify a hidden team."""
import copy
import json
import math
from functools import lru_cache

import numpy as np

from pokesim.config import FORMAT, PROCESSED_DIR
from pokesim.data.team_catalog import digest

PATH = PROCESSED_DIR / 'top_teams.json'
VERSION = 'public-team-catalog-v1'
MIX = .65  # Keep broad independent hypotheses, including entirely unlisted sets.


def ident(value):
    return ''.join(c for c in value.lower() if c.isalnum())


class TeamCatalog:
    def __init__(self, data, base_id=ident):
        if (data.get('schema_version') != 1 or data.get('format') != FORMAT or
                data.get('battle_mode') != 'single' or data.get('teams_sha256') != digest(data['teams'])):
            raise ValueError('team catalog format/content mismatch')
        self.sha, self.base_id = data['teams_sha256'], base_id
        self.mega_abilities = data.get('mega_abilities', {})
        self.teams = data['teams']
        self.members = [{base_id(ident(s['species'])): s for s in t['sets']} for t in self.teams]

    @classmethod
    def load(cls, base_id=ident, path=PATH):
        return cls(json.loads(path.read_text()), base_id) if path.exists() else None

    @lru_cache(maxsize=128)
    def _neighbors(self, preview):
        visible = set(preview)
        found = [(i, len(visible & set(team))) for i, team in enumerate(self.members)]
        return [(i, overlap) for i, overlap in found if overlap >= 4]

    def neighbors(self, preview):
        return self._neighbors(tuple(sorted({self.base_id(ident(s)) for s in preview})))

    def info(self, preview):
        rows = self.neighbors(preview)
        best = max((n for _, n in rows), default=0)
        return {'version': VERSION, 'sha': self.sha[:16], 'matched': len(rows), 'best_overlap': best,
                'catalog_mixture': MIX, 'scope': 'public preview; similar published teams are hypotheses',
                'nearest': [{'id': self.teams[i]['id'], 'overlap': n,
                             'article': self.teams[i].get('article')}
                            for i, n in sorted(rows, key=lambda x: -x[1])[:5]]}

    def compatible(self, pset, known):
        if not set(map(ident, known.get('moves', []))) <= set(map(ident, pset['moves'])):
            return False
        # A consumed/removed item is current state, not evidence that the original team held no item.
        item = known.get('item')
        if item and ident(pset.get('item', '')) != ident(item):
            return False
        ability = ident(known.get('ability') or '')
        return not ability or ability == ident(pset['ability']) or ability in self.mega_abilities.get(
            ident(pset.get('item', '')), [])

    def candidates(self, preview, known):
        rows = []
        for i, overlap in self.neighbors(preview):
            if all(sid not in self.members[i] or self.compatible(self.members[i][sid], info)
                   for sid, info in known.items()):
                rows.append((i, overlap))
        return rows

    def draw(self, preview, known, rng, belief=None):
        """One coherent article per world, conditioned on revealed facts and optional damage/order evidence."""
        rows = self.candidates(preview, known)
        if not rows or rng.random() >= MIX:
            return {}
        weights = np.array([math.exp(2 * n) for _, n in rows], dtype=float)
        if belief is not None:
            for sid in known:
                indices = [j for j, (i, _) in enumerate(rows) if sid in self.members[i]]
                if indices:
                    sets = [self.members[rows[j][0]][sid] for j in indices]
                    posterior = belief.weights(sid, sets)
                    compatible = belief.speed_compatible(belief.relevant(sid), sets)
                    weights[indices] *= posterior * len(indices) * compatible
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            return {}
        index = rng.choices(range(len(rows)), weights=weights)[0]
        selected = self.members[rows[index][0]]
        visible = {self.base_id(ident(s)) for s in preview}
        result = {sid: copy.deepcopy(s) for sid, s in selected.items() if sid in visible}
        for sid, info in known.items():
            if sid in result and info.get('item') == '':
                result[sid]['item'] = ''
        return result

    def draw_set(self, species, moves, item, ability, rng, preview=None):
        sid = self.base_id(ident(species))
        known = {'moves': moves, 'item': item, 'ability': ability}
        # A context-free belief still benefits from complete sets, without pretending to match a whole team.
        if preview and self.neighbors(preview):
            matches = self.draw(preview, {sid: known}, rng)
            return matches.get(sid)
        choices = [team[sid] for team in self.members if sid in team and self.compatible(team[sid], known)]
        if not choices or rng.random() >= MIX:
            return None
        result = copy.deepcopy(rng.choice(choices))
        if item == '':
            result['item'] = ''
        return result
