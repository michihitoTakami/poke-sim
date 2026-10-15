"""Versioned numeric features from an observation only, with explicit missing-information masks.

Public and private branches are separate: public replay pretraining never reconstructs a player's unseen set.
Preview and bench pooling are invariant to ordering; opponent requests, outcome, root action and source are not inputs.
"""
import json
import re
from pathlib import Path

import numpy as np

VERSION = 'observed-value-v1'
STATS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
BOOSTS = (*STATS[1:], 'accuracy', 'evasion')
TYPES = tuple(sorted(('Normal Fighting Flying Poison Ground Rock Bug Ghost Steel Fire Water Grass Electric Psychic '
                      'Ice Dragon Dark Fairy').split()))
STATUSES = ('brn', 'par', 'psn', 'tox', 'slp', 'frz', 'fnt')
VOLATILES = ('substitute', 'yawn', 'taunt', 'encore', 'disable', 'leechseed', 'confusion', 'saltcure',
             'perishsong', 'curse', 'ingrain', 'aquaring')
ABILITIES = ('unaware', 'goodasgold', 'speedboost', 'levitate', 'regenerator', 'illusion', 'moldbreaker',
             'waterabsorb', 'stormdrain', 'flashfire', 'sturdy', 'magicbounce', 'intimidate', 'poisonheal')
ITEMS = ('leftovers', 'sitrusberry', 'mentalherb', 'airballoon', 'focussash', 'choicescarf', 'choiceband',
         'choicespecs', 'eviolite', 'lumberry', 'rockyhelmet', 'heavydutyboots')
HEALS = {'recover', 'synthesis', 'slackoff', 'moonlight', 'morningsun', 'roost', 'softboiled', 'rest',
         'wish', 'shoreup', 'strengthsap', 'milkdrink', 'healorder'}
CONDITIONS = ('stealthrock', 'spikes', 'toxicspikes', 'stickyweb', 'reflect', 'lightscreen', 'auroraveil', 'tailwind')
FLAGS = ('early', 'middle', 'late', 'poison_yawn', 'setup', 'recovery_available', 'baton_available', 'substitute')


def to_id(value):
    return re.sub('[^a-z0-9]', '', str(value or '').lower())


def effect(value):
    return to_id(re.sub(r'^(move|ability|item):\s*', '', str(value or ''), flags=re.I))


def health(value):
    if value is None:
        return 0., '', False, 0.
    parts = str(value).split()
    match = re.match(r'^(\d+)(?:/(\d+))?', parts[0]) if parts else None
    if not match:
        raise ValueError('invalid observed health')
    hp, maximum = int(match[1]), int(match[2] or 100)
    if maximum <= 0 or hp > maximum:
        raise ValueError('invalid observed health range')
    return hp / maximum, parts[1] if len(parts) > 1 else '', True, float(maximum)


class Encoder:
    def __init__(self, dex: dict):
        self.dex = dex
        self.species = {s['id']: s for s in dex['species']}
        self.moves = {m['id']: m for m in dex['moves']}
        self.items = {i['id']: i for i in dex.get('items', [])}
        self.chart = dex.get('typechart', {})

    @classmethod
    def load(cls, path: Path):
        return cls(json.loads(Path(path).read_text()))

    def species_info(self, details):
        return self.species.get(to_id(str(details or '').split(',')[0]), {})

    def key(self, details):
        spec = self.species_info(details)
        return to_id(spec.get('baseSpecies') or spec.get('name') or str(details or '').split(',')[0])

    def static(self, details):
        spec = self.species_info(details)
        return {'species_known': float(bool(spec)),
                **{f'base_{s}': spec.get('baseStats', {}).get(s, 0) / 255 for s in STATS},
                **{f'type_{t}': float(t in spec.get('types', [])) for t in TYPES}}

    def moves_info(self, moves):
        ids = sorted(set(map(to_id, moves)))
        known = [self.moves[m] for m in ids if m in self.moves]
        out = {'moves_seen': len(ids) / 4, 'move_data_known': len(known) / 4}
        for cat in ('Physical', 'Special', 'Status'):
            out[f'moves_{cat}'] = sum(m['category'] == cat for m in known) / 4
        out['max_power'] = max([m.get('basePower', 0) for m in known] or [0]) / 250
        out['priority'] = max([m.get('priority', 0) for m in known] or [0]) / 5
        for name, group in {'heal': HEALS, 'protect': {'protect', 'detect', 'spikyshield', 'kingsshield'},
                            'substitute': {'substitute'}, 'baton': {'batonpass'},
                            'phaze': {'roar', 'whirlwind', 'dragontail', 'circlethrow', 'haze', 'clearsmog'},
                            'encore': {'encore'}, 'taunt': {'taunt'}, 'toxic': {'toxic'},
                            'yawn': {'yawn', 'spore', 'sleeppowder', 'hypnosis'}, 'burn': {'willowisp'},
                            'pivot': {'uturn', 'voltswitch', 'flipturn', 'partingshot', 'shedtail'},
                            'bodypress': {'bodypress'}, 'hazard_clear': {'defog', 'rapidspin', 'mortalspin'}}.items():
            out[f'move_{name}'] = float(bool(set(ids) & group))
        for stat in BOOSTS[:5]:
            gains = [max(0, (m.get('boosts') or {}).get(stat, 0)) for m in known if m.get('target') == 'self']
            out[f'setup_{stat}'] = max(gains or [0]) / 6
        for t in TYPES:
            out[f'attack_{t}'] = float(any(m['type'] == t and m['category'] != 'Status' for m in known))
        return out

    def mon(self, details='', observed=None):
        m = observed or {}
        hp, status, hp_known, _ = health(m.get('condition'))
        out = self.static(m.get('species') or details)
        out.update(revealed=float(observed is not None), hp_known=float(hp_known), hp=hp,
                   **{f'status_{s}': float(s == status) for s in STATUSES})
        out.update({f'boost_{s}': m.get('boosts', {}).get(s, 0) / 6 for s in BOOSTS})
        out.update({f'volatile_{v}': float(v in m.get('volatiles', [])) for v in VOLATILES})
        out.update(item_known=float('item' in m), ability_known=float('ability' in m))
        out.update({f'item_{i}': float(to_id(m.get('item')) == i) for i in ITEMS})
        out['mega_item'] = float(bool(self.items.get(to_id(m.get('item')), {}).get('megaStone')))
        out.update({f'ability_{a}': float(to_id(m.get('ability')) == a) for a in ABILITIES})
        out.update(self.moves_info(m.get('moves', [])))
        return out

    def pooled(self, rows, template, scale):
        return {k: sum(row[k] for row in rows) / scale for k in template}

    def public_side(self, state, side):
        mons = state['mons'][side]
        active = mons.get(state['active'][side])
        out = {f'active.{k}': v for k, v in self.mon(observed=active).items()}
        bench = [self.mon(observed=m) for key, m in mons.items() if key != state['active'][side]]
        out.update({f'bench.{k}': v for k, v in self.pooled(bench, self.mon(), 3).items()})
        preview = [self.static(name) for name in state['preview'][side]]
        out.update({f'preview.{k}': v for k, v in self.pooled(preview, self.static(''), 6).items()})
        hp = [health(m.get('condition')) for m in mons.values()]
        out.update(seen=len(mons) / 3, known_hp=sum(h[0] for h in hp) / 3,
                   fainted=sum(h[1] == 'fnt' for h in hp) / 3, unseen_selected=max(0, 3 - len(mons)) / 3)
        conditions = {effect(k): v for k, v in state.get('conditions', {}).get(side, {}).items()}
        out.update({f'condition_{c}': conditions.get(c, 0) / (3 if c == 'spikes' else 2 if c == 'toxicspikes' else 1)
                    for c in CONDITIONS})
        return out

    def private_mon(self, m=None, request=None):
        m = m or {}
        public = dict(m, species=m.get('details', '').split(',')[0]) if m else None
        out = self.mon(observed=public)
        out.update({f'exact_{s}': m.get('stats', {}).get(s, 0) / 500 for s in STATS[1:]})
        _, _, known, maximum = health(m.get('condition'))
        out.update(exact_hp_known=float(known), max_hp=maximum / 500 if known else 0.)
        slots = (request or {}).get('moves', [])
        out['pp_known'] = float(bool(slots))
        out['pp_remaining'] = sum(s.get('pp', 0) / max(1, s.get('maxpp', 1)) for s in slots) / 4
        out['pp_empty'] = sum(s.get('pp') == 0 for s in slots) / 4
        out['moves_disabled'] = sum(bool(s.get('disabled')) for s in slots) / 4
        return out

    def encode(self, observation):
        if observation.get('schema') != 1 or observation.get('phase') != 'move':
            raise ValueError('unsupported observation schema or phase')
        side = observation['side']
        if side not in ('p1', 'p2'):
            raise ValueError('invalid observation side')
        foe = 'p2' if side == 'p1' else 'p1'
        state = observation['public']
        public = {}
        for label, role in (('own', side), ('foe', foe)):
            public.update({f'{label}.{k}': v for k, v in self.public_side(state, role).items()})
        public['turn'] = min(observation['turn'], 100) / 100
        weather, terrain = effect(state.get('weather')), effect(state.get('terrain'))
        for w in ('sunnyday', 'raindance', 'sandstorm', 'snow', 'hail'):
            public[f'weather_{w}'] = float(weather == w)
        for t in ('electricterrain', 'grassyterrain', 'psychicterrain', 'mistyterrain'):
            public[f'terrain_{t}'] = float(terrain == t)
        req = observation.get('request')
        if req is not None and req.get('side', {}).get('id') != side:
            raise ValueError('private request perspective mismatch')
        own = req.get('side', {}).get('pokemon', []) if req else []
        active = next((m for m in own if m.get('active')), None)
        active_req = (req.get('active') or [{}])[0] if req else {}
        private = {'available': float(req is not None), 'can_mega': float(bool(active_req.get('canMegaEvo'))),
                   'trapped': float(bool(active_req.get('trapped')))}
        private.update({f'active.{k}': v for k, v in self.private_mon(active, active_req).items()})
        private.update({f'bench.{k}': v for k, v in self.pooled(
            [self.private_mon(m) for m in own if not m.get('active')], self.private_mon(), 3).items()})
        if req is None:
            private = dict.fromkeys(private, 0.)
        baseline = [public['own.known_hp'] + public['own.unseen_selected'] - public['foe.known_hp'] -
                    public['foe.unseen_selected'], public['foe.fainted'] - public['own.fainted'],
                    public['own.seen'] - public['foe.seen'], public['turn']]
        all_mons = [m for role in (side, foe) for m in state['mons'][role].values()]
        own_moves = set(m for p in own for m in p.get('moves', []))
        own_moves.update(m for p in state['mons'][side].values() for m in p.get('moves', []))
        turn = observation['turn']
        flags = [turn <= 3, 3 < turn <= 10, turn > 10,
                 any(health(m.get('condition'))[1] in ('psn', 'tox') or 'yawn' in m.get('volatiles', [])
                     for m in all_mons),
                 any(any(b != 0 for b in m.get('boosts', {}).values()) for m in all_mons),
                 bool(own_moves & HEALS), 'batonpass' in own_moves,
                 any('substitute' in m.get('volatiles', []) for m in all_mons)]
        arrays = [np.asarray(list(d.values()), np.float32) for d in (public, private)]
        arrays += [np.asarray(baseline, np.float32), np.asarray(flags, bool)]
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError('nonfinite observation features')
        return (*arrays, {'public': list(public), 'private': list(private),
                         'baseline': ['hp', 'fainted', 'seen', 'turn'],
                         'slices': list(FLAGS), 'version': VERSION})
