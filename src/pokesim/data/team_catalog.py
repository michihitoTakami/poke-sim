"""Import complete public team sheets from the GameWith listing; keep joint sets and source provenance.

The listing's numeric ids belong to its own masters, not the national dex or Showdown.
Only exact name/stat/form matches and teams accepted by the current format validator are retained.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sqlalchemy import select

from pokesim.config import FORMAT, PROCESSED_DIR, SHOWDOWN_DIR
from pokesim.data.fetch_build_links import BASE, HEADERS
from pokesim.db.models import Item, Species
from pokesim.db.session import Session
from pokesim.lookup.names import default_index
from pokesim.runtime.bundle import git_commit
from pokesim.sim.teams import STATS, validate

NATURES = ['lonely', 'brave', 'adamant', 'naughty', 'bold', 'relaxed', 'impish', 'lax', 'timid', 'hasty',
           'serious', 'jolly', 'naive', 'modest', 'mild', 'quiet', 'rash', 'calm', 'gentle', 'sassy', 'careful',
           'hardy', 'docile', 'bashful', 'quirky']
SOURCE = 'https://gamewith.jp/pokemon-champions/560474'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def decode(rows, masters, index, species, items):
    maps = {k: {str(r['id']): r for r in table} for k, table in masters.items()}
    by_stats = {}
    for sp in species.values():
        if not sp.is_mega and not sp.battle_only:
            by_stats.setdefault((sp.num, tuple(sp.base_stats[s] for s in STATS)), []).append(sp)

    def exact(name, kind):
        hits = index.search(name, kind, limit=1, cutoff=1.)
        if not hits or hits[0].score != 1.:
            raise ValueError(f'unresolved-{kind}:{name}')
        return hits[0].ref_id

    def pokemon(row):
        data = maps['pokemon'][str(row['id'])]
        # National number is provided separately by the master icon, never by its internal id.
        num = int(re.match(r'\d+', data['i'])[0])
        choices = by_stats.get((num, tuple(data['bs'])), [])
        if len(choices) > 1:
            hit = index.search(data['n'], 'species', limit=1, cutoff=1.)
            choices = [sp for sp in choices if hit and sp.id == hit[0].ref_id and not data.get('f')]
        if len(choices) != 1:
            raise ValueError(f'ambiguous-form:{data["n"]}:{data.get("f")}')
        sp = choices[0]
        name = maps['item'][str(row['i'])]['n']
        if str(row['i']) == '0':
            item = ''
        else:
            try:
                item = exact(name, 'item')
            except ValueError:
                if 'ナイト' not in name:
                    raise
                megas = [m for m in species.values() if m.is_mega and m.changes_from == sp.name and
                         bool(m.name.endswith('-Z')) == name.endswith('Z')]
                possible = [i.id for i in items.values() if any(i.name == m.required_item for m in megas)]
                if len(possible) != 1:
                    raise ValueError(f'ambiguous-mega-item:{name}') from None
                item = possible[0]
        ability = exact(maps['ability'][str(row['a'])]['n'], 'ability')
        if ability not in sp.abilities:
            raise ValueError('ability-form-mismatch')
        moves = [exact(maps['move'][str(m)]['n'], 'move') for m in row['mo']]
        points = row['e']
        if (len(moves) != 4 or len(set(moves)) != 4 or len(points) != 6 or
                any(type(p) is not int or not 0 <= p <= 32 for p in points) or not 0 < sum(points) <= 66):
            raise ValueError('incomplete-moves-or-spread')
        nature = int(row['na'])
        if not 1 <= nature <= len(NATURES):
            raise ValueError('unknown-nature')
        return {'species': sp.name, 'item': items[item].name if item else '', 'ability': ability,
                'moves': moves, 'nature': NATURES[nature - 1], 'evs': dict(zip(STATS, points)), 'level': 50}

    teams, rejected, seen = [], [], set()
    for row in rows:
        try:
            if len(row.get('po', [])) != 6:
                raise ValueError('incomplete-six')
            sets = [pokemon(p) for p in row['po']]
            if len({s['species'] for s in sets}) != 6:
                raise ValueError('duplicate-species')
            key = digest(sorted(sets, key=lambda s: s['species']))
            if key in seen:
                continue
            seen.add(key)
            teams.append({'id': key[:20], 'season': row.get('s'), 'rank': row.get('r'),
                          'article': row.get('url'), 'listing': SOURCE, 'sets': sets})
        except (ValueError, KeyError, TypeError) as error:
            rejected.append({'rank': row.get('r'), 'reason': str(error)})
    return teams, rejected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--season', default='M-6')
    ap.add_argument('--out', type=Path, default=PROCESSED_DIR / 'top_teams.json')
    ap.add_argument('--offline', action='store_true')
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    if not args.offline:
        with httpx.Client(headers=HEADERS, timeout=30) as client:
            for kind in ['pokemon', 'item', 'ability', 'move']:
                reply = client.get(f'{BASE}/master-{kind}.json').raise_for_status()
                (args.cache / f'master-{kind}.json').write_text(reply.text)
            version = client.get(f'{BASE}/structure-data-{args.season}-single-update.json').raise_for_status().json()
            (args.cache / 'version.json').write_text(json.dumps(version))
            reply = client.get(f'{BASE}/structure-data-{args.season}-single.json',
                               params={'v': version['lastUpdate']}).raise_for_status()
            (args.cache / 'listing.json').write_text(reply.text)
    masters = {k: json.loads((args.cache / f'master-{k}.json').read_text())
               for k in ['pokemon', 'item', 'ability', 'move']}
    with Session() as session:
        species = {sp.id: sp for sp in session.scalars(select(Species))}
        items = {i.id: i for i in session.scalars(select(Item))}
    rows = json.loads((args.cache / 'listing.json').read_text())
    teams, rejected = decode(rows, masters, default_index(), species, items)
    results = validate([t['sets'] for t in teams])
    good = []
    for team, result in zip(teams, results, strict=True):
        if result['ok']:
            good.append(team)
        else:
            rejected.append({'rank': team['rank'], 'reason': result['problems']})
    report = {'schema_version': 1, 'format': FORMAT, 'battle_mode': 'single',
              'validation_engine': git_commit(str(SHOWDOWN_DIR)),
              'mega_abilities': {re.sub('[^a-z0-9]', '', item.name.lower()): sp.abilities
                                 for item in items.values() for sp in species.values()
                                 if sp.is_mega and sp.required_item == item.name},
              'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'listing': SOURCE,
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(args.cache.glob('*.json'))},
              'teams': good, 'teams_sha256': digest(good), 'rejected': rejected}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    print('legal complete teams', len(good), '/', len(rows), 'rejections', len(rejected))
    print(Counter(str(r['reason']).split(':')[0] for r in rejected).most_common(12))


if __name__ == '__main__':
    main()
