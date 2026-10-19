"""Nested sets of Pokemon and move slots, with public history timers and explicit PP masks.

The v1 encoder stays frozen so incumbent checkpoints remain usable for paired comparisons.
Only the actor's observation is read; missing opponent PP/stats are never reconstructed.
"""
import numpy as np

from pokesim.value.features import Encoder as LegacyEncoder
from pokesim.value.features import effect, to_id

VERSION = 'observed-value-sets-v2'
PROTECTS = {'protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker', 'obstruct',
            'silktrap', 'burningbulwark'}


def weather_clock(observation):
    """Public duration bounds, not a sampled hidden weather timer. Upkeep never restarts it."""
    current = effect(observation['public'].get('weather'))
    start, elapsed, entries, source_species = None, 0, {}, ''
    for line in observation.get('history', []):
        p = line.split('|')
        if len(p) > 3 and p[1] in {'switch', 'drag', 'replace', 'detailschange', '-formechange'}:
            entries[p[2]] = to_id(p[3].split(',')[0])
        if len(p) < 3 or p[1] != '-weather':
            continue
        if '[upkeep]' in p:
            if start is not None:
                elapsed += 1
        else:
            start, elapsed = p, 0
            origin = next((s.removeprefix('[of] ') for s in p if s.startswith('[of] ')), '')
            source_species = entries.get(origin, '')
    if not current or start is None or effect(start[2]) != current:
        return dict(known=0., elapsed=0., remaining_min=0., remaining_max=0.)
    # Five or eight turns for ordinary weather, unless the source's public/private item rules out extension.
    lo, hi = 5, 8
    if source_species == 'charizardmegay':
        lo = hi = 5
    if current in {'desolateland', 'primordialsea', 'deltastream'}:
        return dict(known=0., elapsed=0., remaining_min=0., remaining_max=0.)
    return dict(known=1., elapsed=min(elapsed, 8) / 8,
                remaining_min=max(0, lo - elapsed) / 8, remaining_max=max(0, hi - elapsed) / 8)


class Encoder(LegacyEncoder):
    def __init__(self, dex):
        super().__init__(dex)
        self.species_ids = {k: i + 1 for i, k in enumerate(sorted(self.species))}
        self.move_ids = {k: i + 1 for i, k in enumerate(sorted(self.moves))}
        self._schema = None

    def move(self, move='', slot=None):
        mid = to_id(move)
        out = dict(present=float(bool(mid)), identity=float(self.move_ids.get(mid, 0)))
        out.update(self.moves_info([mid] if mid else []))
        known = slot is not None and 'pp' in slot and 'maxpp' in slot
        out.update(pp_known=float(known), pp=slot.get('pp', 0) / max(1, slot.get('maxpp', 1)) if known else 0.,
                   max_pp=slot.get('maxpp', 0) / 64 if known else 0.,
                   disabled_known=float(slot is not None), disabled=float(bool((slot or {}).get('disabled'))))
        return out

    def token(self, details='', observed=None, *, private=None, active=False, request=None):
        present = observed is not None or private is not None
        name = (private or {}).get('details') or (observed or {}).get('species') or details
        out = dict(present=float(present), active=float(active),
                   identity=float(self.species_ids.get(to_id(name.split(',')[0]), 0)))
        # Private stats complement the PUBLIC boosts/volatiles, rather than replacing them with empty request fields.
        data = self.private_mon(private, request) if private is not None else self.private_mon()
        public = self.mon(details, observed)
        data.update(public)
        if private is not None:
            exact = self.private_mon(private, request)
            for key, value in exact.items():
                if not key.startswith(('boost_', 'volatile_')):
                    data[key] = value
        out.update(data)
        moves = sorted(set(map(to_id, (private or observed or {}).get('moves', []))))
        slots = {to_id(s.get('id') or s.get('move')): s for s in (request or {}).get('moves', [])}
        # Request-only Struggle must be represented even when it is absent from the learned moveset.
        moves = sorted(set(moves) | set(slots))
        if len(moves) > 4:
            # Public Transform/Mimic history can include old moves. Keep the currently legal request first.
            moves = sorted(slots) + [m for m in moves if m not in slots]
        for i in range(4):
            mid = moves[i] if i < len(moves) else ''
            out.update({f'move{i}.{k}': v for k, v in self.move(mid, slots.get(mid)).items()})
        return out

    def encode(self, observation):
        p, q, baseline, flags, old_schema = super().encode(observation)
        public = dict(zip(old_schema['public'], p, strict=True))
        private = dict(zip(old_schema['private'], q, strict=True))
        side = observation['side']
        roles = (side, 'p2' if side == 'p1' else 'p1')
        state = observation['public']
        blocks = []

        def append_block(target, label, rows, count, template, branch):
            if len(rows) > count:
                raise ValueError('more Pokemon than the supported BSS roster')
            start = len(target)
            rows = rows + [dict.fromkeys(template, 0.) for _ in range(count - len(rows))]
            for i, row in enumerate(rows):
                target.update({f'{label}{i}.{k}': v for k, v in row.items()})
            blocks.append(dict(branch=branch, start=start, count=count, width=len(template), label=label))

        for label, role in zip(('own', 'foe'), roles, strict=True):
            mons = state['mons'][role]
            rows = [self.token(observed=m, active=key == state['active'][role]) for key, m in sorted(mons.items())]
            append_block(public, label + '.mon', rows, 3, self.token(), 'public')
            preview = [dict(present=1., identity=float(self.species_ids.get(to_id(name.split(',')[0]), 0)),
                            **self.static(name)) for name in sorted(state['preview'][role])]
            template = dict(present=0., identity=0., **self.static(''))
            append_block(public, label + '.preview', preview, 6, template, 'public')
        public.update({f'weather_clock.{k}': v for k, v in weather_clock(observation).items()})
        req = observation.get('request') or {}
        rows = []
        for m in sorted(req.get('side', {}).get('pokemon', []), key=lambda m: m.get('ident', m['details'])):
            key = self.key(m['details'])
            observed = state['mons'][side].get(key)
            rows.append(self.token(observed=observed, private=m, active=m.get('active', False),
                                   request=(req.get('active') or [{}])[0] if m.get('active') else None))
        append_block(private, 'own.mon', rows, 3, self.token(), 'private')
        arrays = [np.asarray(list(d.values()), np.float32) for d in (public, private)]
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError('nonfinite structured features')
        schema = dict(old_schema, public=list(public), private=list(private), version=VERSION,
                      sets=dict(blocks=blocks, mon_width=len(self.token()), move_width=len(self.move()),
                                move_start=list(self.token()).index('move0.present'),
                                species_count=len(self.species_ids) + 1, move_count=len(self.move_ids) + 1,
                                public_context=len(p), private_context=len(q)))
        return *arrays, baseline, flags, schema
