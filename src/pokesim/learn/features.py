"""Policy features of an actor's view and of its legal choices: the Python twin of policy_features.js.

Both read the same view JSON and the same dex data (the exported training dex) and must produce identical
vectors in the order of services/sim-tools/policy_features.schema.json (tests/test_policy_feature_parity.py).
"""
import json
import math
import re
from pathlib import Path

import numpy as np

from pokesim.config import ROOT

VERSION = 'policy-features-v1'
SCHEMA = ROOT / 'services/sim-tools/policy_features.schema.json'
DEX = ROOT / 'data/processed/public_value/training-dex-v1.json'
STATS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
BOOSTS = ('atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion')
TYPES = ('Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying', 'Ghost', 'Grass', 'Ground',
         'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water')
STATUSES = ('brn', 'par', 'psn', 'tox', 'slp', 'frz', 'fnt')
VOLATILES = ('substitute', 'yawn', 'taunt', 'encore', 'disable', 'leechseed', 'confusion', 'saltcure',
             'perishsong', 'curse', 'ingrain', 'aquaring')
ITEMS = ('leftovers', 'sitrusberry', 'mentalherb', 'airballoon', 'focussash', 'choicescarf', 'choiceband',
         'choicespecs', 'eviolite', 'lumberry', 'rockyhelmet', 'heavydutyboots')
ABILITIES = ('unaware', 'goodasgold', 'speedboost', 'levitate', 'regenerator', 'illusion', 'moldbreaker',
             'waterabsorb', 'stormdrain', 'flashfire', 'sturdy', 'magicbounce', 'intimidate', 'poisonheal')
CONDITIONS = ('stealthrock', 'spikes', 'toxicspikes', 'stickyweb', 'reflect', 'lightscreen', 'auroraveil', 'tailwind')
WEATHERS = ('sunnyday', 'raindance', 'sandstorm', 'snow', 'hail')
TERRAINS = ('electricterrain', 'grassyterrain', 'psychicterrain', 'mistyterrain')
PROTECTS = ('protect', 'detect', 'kingsshield', 'spikyshield', 'banefulbunker', 'obstruct', 'silktrap',
            'burningbulwark')
HAZES = ('haze', 'clearsmog')
CATEGORIES = ('Physical', 'Special', 'Status')


def to_id(value) -> str:
    return re.sub('[^a-z0-9]', '', str(value or '').lower())


def effect_id(value) -> str:
    return to_id(re.sub(r'^(?:move|ability|item): ', '', str(value or '')))


def stage(s: float) -> float:
    return (2 + s) / 2 if s >= 0 else 2 / (2 - s)


def clip(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def parse_float(text: str) -> float:
    """JavaScript parseFloat: the leading number of the text (NaN when there is none)."""
    m = re.match(r'^\s*[+-]?(\d+\.?\d*|\.\d+)', str(text))
    return float(m.group(0)) if m else float('nan')


def health(text) -> dict:
    """As public_observation.js health(): hp, maxhp, fraction, status ("49/100y" public colours included)."""
    parts = str(text or '').split(' ')
    hp_part = parts[0].split('/')
    n = parse_float(hp_part[0])
    d = parse_float(hp_part[1]) if len(hp_part) > 1 else 100.
    n = 0. if math.isnan(n) else n
    return {'hp': n, 'maxhp': d, 'fraction': n / d, 'status': parts[1] if len(parts) > 1 else ''}


def trick_room(history) -> bool:
    on = False
    for line in history or []:
        if re.match(r'^\|-fieldstart\|(?:move: )?Trick Room', line):
            on = True
        if re.match(r'^\|-fieldend\|(?:move: )?Trick Room', line):
            on = False
    return on


class Features:
    def __init__(self, dex: dict, schema: dict | None = None):
        self.species = {s['id']: s for s in dex['species']}
        self.moves = {m['id']: m for m in dex['moves']}
        self.items = {i['id']: i for i in dex.get('items', [])}
        self.chart = dex['typechart']
        schema = schema or json.loads(SCHEMA.read_text())
        if schema['version'] != VERSION:
            raise ValueError('feature schema version mismatch')
        self.state_names, self.action_names = schema['state'], schema['action']

    @classmethod
    def load(cls, dex: Path = DEX, schema: Path = SCHEMA):
        return cls(json.loads(Path(dex).read_text()), json.loads(Path(schema).read_text()))

    # --- dex helpers (the JS reads the live dex; values agree with the export) ---
    def species_of(self, name):
        return self.species.get(to_id(name))

    def move_of(self, name):
        return self.moves.get(to_id(name))

    def item_of(self, name):
        return self.items.get(to_id(name))

    def effectiveness(self, type_: str, types: list[str]):
        total = 0.
        for t in types:
            k = self.chart.get(type_, {}).get(t, 1)
            if k == 0:
                return None
            total += {2: 1, .5: -1}.get(k, 0)
        return total

    def stat(self, mon: dict, s: str, boosts: dict | None, own: bool) -> float:
        spec = self.species_of(mon['details'].split(',')[0] if own else mon.get('species'))
        base = spec['baseStats'][s] if spec else 80
        if own and mon.get('stats') and mon['stats'].get(s) is not None:
            value = float(mon['stats'][s])
        else:
            value = float(math.floor((2 * base + 31 + 16) / 2) + 5)
        value *= stage((boosts or {}).get(s, 0) or 0)
        if s == 'spe' and health(mon.get('condition'))['status'] == 'par':
            value *= .5
        if s == 'spe' and to_id(mon.get('item')) == 'choicescarf':
            value *= 1.5
        return value

    def max_hp(self, mon: dict, own: bool) -> float:
        if own:
            return health(mon.get('condition'))['maxhp']
        spec = self.species_of(mon.get('species'))
        return float(math.floor((2 * (spec['baseStats']['hp'] if spec else 80) + 31 + 16) / 2) + 60)

    def hp(self, mon: dict, own: bool) -> float:
        h = health(mon.get('condition'))
        return h['hp'] if own else h['fraction'] * self.max_hp(mon, own)

    def types_of(self, mon: dict, own: bool) -> list[str]:
        spec = self.species_of(mon['details'].split(',')[0] if own else mon.get('species'))
        return list(spec['types']) if spec else []

    def damage(self, source, target, move, source_boosts, target_boosts, source_own, target_own) -> float:
        if not move or move['category'] == 'Status':
            return 0.
        types = self.types_of(target, target_own)
        eff = self.effectiveness(move['type'], types)
        if eff is None:
            return 0.
        ability, item = to_id(target.get('ability') or ''), to_id(target.get('item') or '')
        if ((move['type'] == 'Ground' and (ability == 'levitate' or item == 'airballoon')) or
                (move['type'] == 'Water' and ability in ('waterabsorb', 'stormdrain')) or
                (move['type'] == 'Fire' and ability == 'flashfire')):
            return 0.
        physical = move['category'] == 'Physical'
        power = move.get('basePower') or (80 if move['id'] == 'bodypress' else 60)
        multihit = move.get('multihit')
        power *= 3 if isinstance(multihit, list) else (multihit or 1)
        if to_id(target.get('ability') or '') == 'unaware':
            source_boosts = {}
        if to_id(source.get('ability') or '') == 'unaware':
            target_boosts = {}
        result = ((22 * power * self.stat(source, 'atk' if physical else 'spa', source_boosts, source_own) /
                   max(1., self.stat(target, 'def' if physical else 'spd', target_boosts, target_own))) / 50 + 2) * .925
        if move['type'] in self.types_of(source, source_own):
            result *= 1.5
        result *= 2 ** eff
        if physical and health(source.get('condition'))['status'] == 'brn':
            result *= .5
        if isinstance(move.get('accuracy'), (int, float)) and not isinstance(move.get('accuracy'), bool):
            result *= move['accuracy'] / 100
        return result

    # --- shared context ---
    def context(self, view: dict) -> dict:
        side = view['side']
        foe_side = 'p2' if side == 'p1' else 'p1'
        own = view['request']['side']['pokemon']
        me = next((m for m in own if m.get('active')), own[0])
        pub = view['public']
        my_public = pub['mons'][side].get(pub['active'][side]) or {'boosts': {}, 'volatiles': [], 'moves': []}
        foe = pub['mons'][foe_side].get(pub['active'][foe_side])
        known = [self.move_of(m) for m in foe['moves']] if foe else []
        known = [m for m in known if m and m['category'] != 'Status']
        foe_spec = self.species_of(foe['species']) if foe else None
        generic = [{'id': 'generic', 'type': t, 'basePower': 90, 'accuracy': 100,
                    'category': ('Physical' if foe_spec['baseStats']['atk'] >= foe_spec['baseStats']['spa']
                                 else 'Special')} for t in foe_spec['types']] if foe_spec else []
        threats = known + (generic if (not foe or len(foe['moves']) < 4) else [])
        room = trick_room(view.get('history'))

        def incoming(target):
            if not foe:
                return 0.
            return max([0.] + [self.damage(foe, target, m, foe.get('boosts'), my_public.get('boosts'), False, True)
                               for m in threats])

        def best_damage(source):
            if not foe:
                return 0.
            boosts = my_public.get('boosts') if source is me else {}
            return max([0.] + [self.damage(source, foe, self.move_of(mid), boosts, foe.get('boosts'), True, False)
                               for mid in source.get('moves', [])])

        def my_speed(m):
            return self.stat(m, 'spe', my_public.get('boosts') if m is me else {}, True)

        foe_speed = self.stat(foe, 'spe', foe.get('boosts'), False) if foe else 0.

        def faster(m):
            return my_speed(m) < foe_speed if room else my_speed(m) > foe_speed

        return {'side': side, 'foe_side': foe_side, 'own': own, 'me': me, 'pub': pub, 'my_public': my_public,
                'foe': foe, 'threats': threats, 'incoming': incoming, 'best_damage': best_damage, 'room': room,
                'my_speed': my_speed, 'foe_speed': foe_speed, 'faster': faster}

    def species_features(self, out, prefix, spec):
        for s in STATS:
            out[f'{prefix}.base_{s}'] = (spec['baseStats'][s] if spec else 0) / 255
        for t in TYPES:
            out[f'{prefix}.type_{t}'] = 1. if spec and t in spec['types'] else 0.
        out[f'{prefix}.mega'] = 1. if spec and spec.get('isMega') else 0.

    def item_features(self, out, prefix, item):
        i = to_id(item or '')
        for name in ITEMS:
            out[f'{prefix}.item_{name}'] = 1. if i == name else 0.
        data = self.item_of(i)
        out[f'{prefix}.mega_item'] = 1. if data and data.get('megaStone') else 0.
        out[f'{prefix}.berry'] = 1. if data and data.get('isBerry') else 0.

    def ability_features(self, out, prefix, ability):
        a = to_id(ability or '')
        for name in ABILITIES:
            out[f'{prefix}.ability_{name}'] = 1. if a == name else 0.

    def mega_item(self, mon) -> float:
        data = self.item_of(mon.get('item') or '') if mon else None
        return 1. if data and data.get('megaStone') else 0.

    def state(self, view: dict) -> dict:
        c = self.context(view)
        me, foe, my_public, pub, own = c['me'], c['foe'], c['my_public'], c['pub'], c['own']
        out: dict = {}
        out['turn'] = min(view['turn'], 100) / 100
        out['phase_switch'] = 1. if view['phase'] == 'switch' else 0.
        out['trick_room'] = 1. if c['room'] else 0.
        my_health = health(me.get('condition'))
        self.species_features(out, 'me', self.species_of(me['details'].split(',')[0]))
        out['me.hp'] = my_health['fraction']
        for s in STATUSES:
            out[f'me.status_{s}'] = 1. if my_health['status'] == s else 0.
        for b in BOOSTS:
            out[f'me.boost_{b}'] = ((my_public.get('boosts') or {}).get(b, 0) or 0) / 6
        for v in VOLATILES:
            out[f'me.volatile_{v}'] = 1. if v in (my_public.get('volatiles') or []) else 0.
        self.item_features(out, 'me', me.get('item'))
        self.ability_features(out, 'me', me.get('ability') or me.get('baseAbility'))
        active = (view['request'].get('active') or [{}])[0]
        out['me.can_mega'] = 1. if active.get('canMegaEvo') else 0.
        out['me.trapped'] = 1. if active.get('trapped') else 0.
        slots = active.get('moves') or []
        out['me.pp_min'] = min((m.get('pp') or 0) / max(1, m.get('maxpp') or 1) for m in slots) if slots else 0.
        out['me.moves'] = len(me.get('moves') or []) / 4
        out['me.faster'] = 1. if foe and c['faster'](me) else 0.
        out['me.speed_share'] = c['my_speed'](me) / (c['my_speed'](me) + c['foe_speed']) if foe else .5
        out['me.incoming'] = clip(c['incoming'](me) / max(1., self.hp(me, True)), 0, 2)
        out['me.outgoing'] = clip(c['best_damage'](me) / max(1., self.hp(foe, False)), 0, 2) if foe else 0.
        out['me.boosted'] = 1. if any(v > 0 for v in (my_public.get('boosts') or {}).values()) else 0.
        bench = [m for m in own if m is not me]
        for k in range(2):
            m = bench[k] if k < len(bench) else None
            prefix = f'bench{k}'
            h = health(m.get('condition')) if m else {'fraction': 0., 'status': 'fnt', 'hp': 0.}
            out[f'{prefix}.present'] = 1. if m else 0.
            out[f'{prefix}.alive'] = 1. if m and h['hp'] > 0 else 0.
            out[f'{prefix}.hp'] = h['fraction'] if m else 0.
            for s in STATUSES:
                out[f'{prefix}.status_{s}'] = 1. if m and h['status'] == s else 0.
            self.species_features(out, prefix, self.species_of(m['details'].split(',')[0]) if m else None)
            out[f'{prefix}.mega_item'] = self.mega_item(m) if m else 0.
            alive = bool(m) and h['hp'] > 0
            out[f'{prefix}.faster'] = 1. if m and foe and alive and c['faster'](m) else 0.
            out[f'{prefix}.incoming'] = clip(c['incoming'](m) / max(1., h['hp']), 0, 2) if m and foe and alive else 0.
            out[f'{prefix}.outgoing'] = (clip(c['best_damage'](m) / max(1., self.hp(foe, False)), 0, 2)
                                         if m and foe and alive else 0.)
        fh = health(foe.get('condition')) if foe else {'fraction': 0., 'status': ''}
        out['foe.present'] = 1. if foe else 0.
        self.species_features(out, 'foe', self.species_of(foe['species']) if foe else None)
        out['foe.hp'] = fh['fraction'] if foe else 0.
        for s in STATUSES:
            out[f'foe.status_{s}'] = 1. if foe and fh['status'] == s else 0.
        for b in BOOSTS:
            out[f'foe.boost_{b}'] = (((foe.get('boosts') or {}).get(b, 0) or 0) / 6) if foe else 0.
        for v in VOLATILES:
            out[f'foe.volatile_{v}'] = 1. if foe and v in (foe.get('volatiles') or []) else 0.
        foe_moves = [m for m in (self.move_of(x) for x in foe['moves']) if m] if foe else []
        out['foe.moves_seen'] = len(foe_moves) / 4
        for cat in CATEGORIES:
            out[f'foe.moves_{cat}'] = sum(m['category'] == cat for m in foe_moves) / 4
        out['foe.max_power'] = max([0] + [m.get('basePower') or 0 for m in foe_moves]) / 250
        out['foe.priority'] = max([0] + [m.get('priority') or 0 for m in foe_moves]) / 5
        out['foe.item_known'] = 1. if foe and 'item' in foe else 0.
        self.item_features(out, 'foe', foe.get('item') if foe else '')
        out['foe.ability_known'] = 1. if foe and 'ability' in foe else 0.
        self.ability_features(out, 'foe', foe.get('ability') if foe else '')
        last = self.move_of(foe['lastMove']) if foe and foe.get('lastMove') else None
        out['foe.last_known'] = 1. if last else 0.
        for cat in CATEGORIES:
            out[f'foe.last_{cat}'] = 1. if last and last['category'] == cat else 0.
        out['foe.boosted'] = 1. if foe and any(v > 0 for v in (foe.get('boosts') or {}).values()) else 0.
        out['foe.incoming_hint'] = clip(c['incoming'](me) / max(1., self.hp(me, True)), 0, 2) if foe else 0.
        foe_mons = [m for key, m in pub['mons'][c['foe_side']].items() if key != pub['active'][c['foe_side']]]
        out['foe.bench_seen'] = len(foe_mons) / 3
        out['foe.bench_alive'] = sum(health(m.get('condition'))['hp'] > 0 for m in foe_mons) / 3
        out['foe.bench_hp'] = sum(health(m.get('condition'))['fraction'] for m in foe_mons) / 3
        for s in STATS:
            out[f'foe.bench_base_{s}'] = sum((self.species_of(m['species']) or {'baseStats': {s: 0}})['baseStats'][s]
                                             for m in foe_mons) / (255 * 3)
        for t in TYPES:
            out[f'foe.bench_type_{t}'] = 1. if any(t in (self.species_of(m['species']) or {'types': []})['types']
                                                   for m in foe_mons) else 0.
        preview = [self.species_of(d.split(',')[0]) for d in pub['preview'][c['foe_side']]]
        out['foe.unseen'] = max(0, len(preview) - len(pub['mons'][c['foe_side']])) / 6
        for s in STATS:
            out[f'foe.preview_base_{s}'] = sum(sp['baseStats'][s] if sp else 0 for sp in preview) / (255 * 6)
        for t in TYPES:
            out[f'foe.preview_type_{t}'] = sum(1 for sp in preview if sp and t in sp['types']) / 6
        weather, terrain = effect_id(pub.get('weather')), effect_id(pub.get('terrain'))
        for w in WEATHERS:
            out[f'weather_{w}'] = 1. if weather == w else 0.
        for t in TERRAINS:
            out[f'terrain_{t}'] = 1. if terrain == t else 0.
        for label, role in (('own', c['side']), ('foe', c['foe_side'])):
            conditions = {effect_id(k): v for k, v in (pub.get('conditions', {}).get(role) or {}).items()}
            for cond in CONDITIONS:
                out[f'{label}.condition_{cond}'] = (conditions.get(cond, 0) or 0) / (
                    3 if cond == 'spikes' else 2 if cond == 'toxicspikes' else 1)
        return out

    def action(self, view: dict, choice: str, label: str, c: dict | None = None) -> dict:
        c = c or self.context(view)
        me, foe, my_public, own = c['me'], c['foe'], c['my_public'], c['own']
        parts = label.split(':')
        kind, mega = parts[0], len(parts) > 2 and parts[2] == 'mega'
        out: dict = {'kind_move': 1. if kind == 'move' else 0., 'kind_mega': 1. if mega else 0.,
                     'kind_switch': 1. if kind == 'switch' else 0.}
        move = self.move_of(parts[1]) if kind == 'move' else None
        known = bool(move)
        for t in TYPES:
            out[f'move.type_{t}'] = 1. if known and move['type'] == t else 0.
        for cat in CATEGORIES:
            out[f'move.{cat}'] = 1. if known and move['category'] == cat else 0.
        out['move.power'] = (move.get('basePower') or 0) / 200 if known else 0.
        acc = move.get('accuracy') if known else None
        out['move.accuracy'] = ((acc / 100) if isinstance(acc, (int, float)) and not isinstance(acc, bool) else 1.) \
            if known else 0.
        out['move.priority'] = (move.get('priority') or 0) / 5 if known else 0.
        slots = (view['request'].get('active') or [{}])[0].get('moves') or []
        slot = next((s for s in slots if s.get('id') == move['id']), None) if known else None
        out['move.pp'] = (slot.get('pp') or 0) / max(1, slot.get('maxpp') or 1) if slot else 0.
        flags = set(move.get('flags') or []) if known else set()
        out['move.heal'] = 1. if known and ('heal' in flags or move.get('drain')) else 0.
        out['move.protect'] = 1. if known and move['id'] in PROTECTS else 0.
        out['move.substitute'] = 1. if known and move['id'] == 'substitute' else 0.
        out['move.batonpass'] = 1. if known and move['id'] == 'batonpass' else 0.
        out['move.pivot'] = 1. if known and move.get('selfSwitch') and move['id'] != 'batonpass' else 0.
        out['move.phaze'] = 1. if known and move.get('forceSwitch') else 0.
        out['move.haze'] = 1. if known and move['id'] in HAZES else 0.
        boosts = {}
        if known:
            if move.get('target') == 'self' and move.get('boosts'):
                boosts = move['boosts']
            else:
                boosts = ((move.get('self') or {}).get('boosts')) or {}
        out['move.setup'] = 1. if any(v > 0 for v in boosts.values()) else 0.
        for b in BOOSTS[:5]:
            out[f'move.boost_{b}'] = max(0, boosts.get(b, 0) or 0) / 2
        out['move.status'] = 1. if known and move.get('status') else 0.
        out['move.sound'] = 1. if known and 'sound' in flags else 0.
        out['move.contact'] = 1. if known and 'contact' in flags else 0.
        out['move.multihit'] = 1. if known and move.get('multihit') else 0.
        out['move.recoil'] = 1. if known and move.get('recoil') else 0.
        out['move.last'] = 1. if known and my_public.get('lastMove') == move['id'] else 0.
        out['move.stab'] = 1. if known and move['type'] in self.types_of(me, True) else 0.
        eff = None
        if known and foe and move['category'] != 'Status':
            eff = self.effectiveness(move['type'], self.types_of(foe, False))
        out['move.immune'] = 1. if known and foe and move['category'] != 'Status' and eff is None else 0.
        out['move.effectiveness'] = 0. if eff is None else eff / 2
        dmg = (self.damage(me, foe, move, my_public.get('boosts'), foe.get('boosts'), True, False)
               if known and foe else 0.)
        foe_hp = max(1., self.hp(foe, False)) if foe else 1.
        out['move.damage'] = clip(dmg / foe_hp, 0, 2)
        out['move.kills'] = 1. if foe and dmg >= self.hp(foe, False) else 0.
        target = own[int(choice.split(' ')[1]) - 1] if kind == 'switch' else None
        th = health(target.get('condition')) if target else {'fraction': 0., 'status': '', 'hp': 0.}
        self.species_features(out, 'switch', self.species_of(target['details'].split(',')[0]) if target else None)
        out['switch.hp'] = th['fraction'] if target else 0.
        for s in STATUSES:
            out[f'switch.status_{s}'] = 1. if target and th['status'] == s else 0.
        out['switch.incoming'] = clip(c['incoming'](target) / max(1., th['hp']), 0, 2) if target and foe else 0.
        out['switch.outgoing'] = clip(c['best_damage'](target) / foe_hp, 0, 2) if target and foe else 0.
        out['switch.faster'] = 1. if target and foe and c['faster'](target) else 0.
        out['switch.mega_item'] = self.mega_item(target) if target else 0.
        return out

    def vectors(self, view: dict, legal: list) -> tuple[np.ndarray, np.ndarray]:
        """State vector and one action vector per legal (choice, label) pair, in schema order."""
        c = self.context(view)
        state = self.state(view)
        s = np.array([state[name] for name in self.state_names], dtype=np.float32)
        a = np.array([[self.action(view, choice, label, c)[name] for name in self.action_names]
                      for choice, label in legal], dtype=np.float32)
        return s, a
