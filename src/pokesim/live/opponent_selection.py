"""Roster-specific, censored likelihood of the opponent's three picks and lead.

Enumerate all 60 legal lead/trio choices. A game showing two members constrains a set of lineups; it does not
label the four unobserved members as absent. Only completed rated games with the same visible own roster enter.
"""
import itertools
import json
from collections import Counter

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import logsumexp, softmax

from pokesim.rl.myteam import to_id

VERSION = 'roster-conditional-lineups-v1'


def options(species):
    return [(lead, *bench) for lead in range(len(species))
            for bench in itertools.combinations([i for i in range(len(species)) if i != lead], 2)]


def pick_rows(root, roster):
    from pokesim.live.selection import _battle_formes

    aliases = _battle_formes()
    own = sorted(map(to_id, roster))
    rows, seen = [], set()
    for path in sorted(root.glob('*.json'), reverse=True):
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if (not isinstance(rec, dict) or rec.get('result') not in ('win', 'loss') or
                '|rated|' not in rec.get('log', []) or sorted(map(to_id, rec.get('team', []))) != own or
                not rec.get('source') or rec['source'] in seen):
            continue
        preview = list(map(to_id, (rec.get('preview') or {}).get('species', [])))
        if len(preview) != 6 or len(set(preview)) != 6:
            continue
        episodes = []
        for line in rec.get('log', []):
            p = line.split('|')
            if len(p) > 3 and p[2].startswith('p2') and p[1] in ('switch', 'drag', 'replace'):
                sid = to_id(p[3].split(',')[0])
                sid = aliases.get(sid, sid)
                if p[1] == 'replace' and episodes:
                    episodes[-1] = [sid]
                else:
                    # An unbroken Illusion is ambiguous. Never manufacture an exact lead/trio label from it.
                    disguised = [s for s in preview if s.startswith('zoroark') and s != sid]
                    episodes.append(sorted(set([sid, *disguised])))
        if not episodes or any(not set(e) <= set(preview) for e in episodes):
            continue
        shown = sorted({tuple(e) for e in episodes})
        # No outcome feature or chosen own trio: opponents saw the six, not our private selection.
        rows.append({'record': path.name, 'preview': preview, 'lead': episodes[0], 'shown': shown})
        seen.add(rec['source'])
    return rows


def prior_logits(picks):
    p = np.clip(picks.brought, .02, .98)
    odds = np.log(p / (1 - p))
    lead = np.maximum(picks.lead, 1e-6)
    return np.array([sum(odds[list(o)]) + np.log(lead[o[0]] / sum(lead[list(o)]))
                     for o in options(picks.species)])


class OpponentLineups:
    def __init__(self, rows, base, signature='', l2=6.):
        self.base, self.signature, self.rows = base, signature, rows
        counts = Counter(tuple(sorted(pair)) for r in rows for pair in itertools.combinations(r['preview'], 2))
        species = sorted({s for r in rows for s in r['preview']})
        features = [('member', s) for s in species] + [('lead', s) for s in species]
        features += [('pair', *p) for p, n in sorted(counts.items()) if n >= 4]
        self.features = {f: i for i, f in enumerate(features)}
        self.theta = np.zeros(len(features))
        self.fit_status = 'prior-only'
        if not rows or not features:
            return
        designs, priors, masks = [], [], []
        for row in rows:
            species = row['preview']
            opts = options(species)
            mask = np.array([species[o[0]] in row['lead'] and all(
                set(shown) & {species[i] for i in o} for shown in row['shown']) for o in opts])
            if not mask.any():
                continue
            designs.append(self.design(species))
            priors.append(prior_logits(base.predict(species)))
            masks.append(mask)
        if not designs:
            return
        from scipy.sparse import vstack

        x = vstack(designs, format='csr')
        prior, mask = np.array(priors), np.array(masks)
        weights = .5 ** (np.arange(len(priors)) / 150.)

        def objective(theta):
            z = prior + (x @ theta).reshape(prior.shape)
            compatible = np.where(mask, z, -np.inf)
            loss = weights @ (logsumexp(z, axis=1) - logsumexp(compatible, axis=1)) + .5 * l2 * (theta @ theta)
            delta = weights[:, None] * (softmax(z, axis=1) - softmax(compatible, axis=1))
            return float(loss), np.asarray(x.T @ delta.ravel()) + l2 * theta

        result = minimize(objective, self.theta, jac=True, method='L-BFGS-B', options={'maxiter': 100})
        if np.isfinite(result.fun) and np.isfinite(result.x).all() and result.fun <= objective(self.theta)[0]:
            self.theta = result.x
            self.fit_status = 'fitted' if result.success else 'iteration-limit'

    def design(self, species):
        indices, indptr = [], [0]
        for o in options(species):
            names = [species[i] for i in o]
            fs = [('member', s) for s in names] + [('lead', names[0])]
            fs += [('pair', *sorted(pair)) for pair in itertools.combinations(names, 2)]
            indices.extend(self.features[f] for f in fs if f in self.features)
            indptr.append(len(indices))
        return csr_matrix((np.ones(len(indices)), indices, indptr), shape=(len(indptr) - 1, len(self.theta)))

    def predict(self, picks):
        from pokesim.live.selection import OpponentPicks

        ids = list(map(to_id, picks.species))
        opts = options(ids)
        prob = softmax(prior_logits(picks) + self.design(ids) @ self.theta)
        brought, lead = np.zeros(len(ids)), np.zeros(len(ids))
        for o, p in zip(opts, prob, strict=True):
            brought[list(o)] += p
            lead[o[0]] += p
        lineups = [([picks.species[i] for i in o], float(p)) for o, p in zip(opts, prob, strict=True)]
        return OpponentPicks(picks.species, brought, lead, sorted(lineups, key=lambda x: -x[1]),
                             {'version': VERSION, 'sha': self.signature[:16], 'records': len(self.rows),
                              'prior_sha': self.base.prior_sha,
                              'fit': self.fit_status, 'censored_labels': True})


def draw_unseen(picks, seen, rng):
    """One coherent selected trio, conditional on every reliably observed opponent; no new hidden information."""
    seen = set(map(to_id, seen))
    compatible = [(names, p) for names, p in picks.lineups if seen <= set(map(to_id, names))]
    if not compatible:
        return None
    names, _ = rng.choices(compatible, weights=[p for _, p in compatible])[0]
    return [to_id(s) for s in names if to_id(s) not in seen]
