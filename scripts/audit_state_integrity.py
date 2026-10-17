"""Compare simulator reconstructions with facts in each recorded decision's observed prefix.

Run this same script with PYTHONPATH pointing to the before/after source trees. It never sends orders or
uses future revealed sets. Unknown sets are sampled without the live-record prior; their quality is not scored.
"""
import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from pokesim.config import FORMAT, ROOT, SHOWDOWN_DIR
from pokesim.evaluation.weakness import analyse_record, species_aliases
from pokesim.live.session import BattleSession
from pokesim.search.lookahead import Lookahead, SetSampler, field_state, to_id

INSPECT = r"""
const fs=require('fs'),path=require('path');
const {setUp}=require(path.resolve(process.argv[1],'services/sim-tools/state_codec.js')).createCodec(process.argv[2]);
const rows=JSON.parse(fs.readFileSync(0,'utf8'));
const out=rows.map(row=>{
 let b;
 try {
  b=setUp(row.sample,process.argv[3],1);
  return {p1:b.p1.pokemon.map(m=>({item:m.item,forme:m.species.name})),
   p2Forme:b.p2.active[0].species.name,faints:b.sides.map(s=>s.totalFainted)};
 } catch(e) {return {error:String(e.message||e)};}
 finally {if(b)b.destroy();}
});
process.stdout.write(JSON.stringify(out));
"""


def audit(args):
    sets = json.loads(args.team.read_text())
    engine = object.__new__(Lookahead)
    engine.sampler = SetSampler(['M-6'], observed={})
    sampler = engine.sampler
    own = {sampler.base_id(to_id(s['species'])): s for s in sets}
    dex = json.loads(args.dex.read_text())
    aliases = species_aliases(dex)
    snapshots, sources, excluded = [], {}, []
    outcomes = Counter()
    for path in sorted(args.records.glob('*.json')):
        rec = json.loads(path.read_text())
        if not rec.get('source', '').startswith('showdown ') or rec.get('result') not in ('win', 'loss'):
            continue
        if args.until and rec.get('started', '') > args.until:
            continue
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        outcomes[rec['result']] += 1
        if analyse_record(rec, aliases)['lineup_mismatch']:
            excluded.append({'record': path.name, 'reason': 'saved lineup differs from actual own appearances'})
            continue
        session = BattleSession(sets)
        session.preview(rec['preview']['species'])
        session.select([next(i for i, s in enumerate(sets) if to_id(s['species']) == to_id(n))
                        for n in rec['lineup']])
        items = {info['name']: info['item'] for info in session.info}
        formes, active, faints = {}, {}, Counter()
        known_moves, entries = defaultdict(set), {}
        turns = {a['turn'] for a in rec.get('advice', []) if not a.get('force')}
        start = next(i for i, line in enumerate(rec['log']) if line.startswith('|teampreview')) + 1
        for line in rec['log'][start:]:
            p = line.split('|')
            if len(p) < 3 or p[1] == 'error':
                continue
            kind, ident = p[1:3]
            side, _, name = ident.partition(': ')
            side = side[:2]
            if kind in {'switch', 'drag'} and len(p) > 3:
                entries[side] = {'name': name, 'before': set(known_moves[(side, name)]), 'moves': set()}
                active[side] = name
                formes[(side, name)] = p[3].split(',')[0]
            elif kind == 'replace' and len(p) > 3:
                entry = entries.pop(side, None)
                if entry:
                    known_moves[(side, entry['name'])] = entry['before']
                    known_moves[(side, name)].update(entry['moves'])
                active[side] = name
                formes[(side, name)] = p[3].split(',')[0]
            elif kind == 'move' and len(p) > 3 and '[from]' not in line:
                mid = to_id(p[3])
                known_moves[(side, name)].add(mid)
                if side in entries:
                    entries[side]['moves'].add(mid)
            elif kind == 'detailschange' and len(p) > 3:
                formes[(side, name)] = p[3].split(',')[0]
            elif kind == '-enditem' and side == 'p1':
                items[name] = ''
            elif kind == '-item' and side == 'p1' and len(p) > 3:
                items[name] = to_id(p[3])
            elif kind == 'faint':
                faints[side] += 1
            try:
                session.apply([line])
                if kind != 'turn' or int(p[2]) not in turns:
                    continue
                battle = session.decision()
                sampler.rng = random.Random(f'{path.name}:{battle.turn}')
                sample = {'p1': engine.own_side(battle, own), 'p2': engine.opponent_side(battle),
                          'field': field_state(battle)}
                selected = battle.last_request['side']['pokemon']
                names = [entry['ident'].partition(': ')[2] for entry in selected]
                # own_side puts active first, followed by the selected roster in request order.
                lead = battle.active_pokemon.name
                names = [lead] + [n for n in names if n != lead]
                snapshots.append({'record': path.name, 'turn': battle.turn, 'sample': sample,
                                  'items': [items[n] for n in names],
                                  'p1Formes': [formes.get(('p1', n)) for n in names],
                                  'p2Forme': formes.get(('p2', active.get('p2'))),
                                  'known_foe_moves': sorted(known_moves[('p2', active.get('p2'))]),
                                  'parsed_foe_moves': sorted(battle.opponent_active_pokemon.moves),
                                  'faints': [faints['p1'], faints['p2']]})
            except Exception as e:  # one bad prefix is explicit, never silently a matching state
                excluded.append({'record': path.name, 'line': line, 'reason': repr(e)[:200]})
                break
    result = subprocess.run(['node', '-e', INSPECT, str(ROOT), str(SHOWDOWN_DIR), FORMAT],
                            input=json.dumps(snapshots), text=True, capture_output=True, check=True, timeout=180)
    restored = json.loads(result.stdout)
    counts, affected, mismatches = Counter(), defaultdict(set), []
    for row, actual in zip(snapshots, restored, strict=True):
        errors = []
        missing = set(row['known_foe_moves']) - set(row['parsed_foe_moves'])
        if missing:
            errors.append({'kind': 'lost_opponent_moves', 'missing': sorted(missing)})
        if actual.get('error'):
            errors.append({'kind': 'reconstruction_error', 'actual': actual['error']})
        else:
            for i, (expected, mon) in enumerate(zip(row['items'], actual['p1'], strict=True)):
                if to_id(expected) != mon['item']:
                    errors.append({'kind': 'own_item', 'slot': i, 'expected': expected, 'actual': mon['item']})
            for side, expected, seen in [('p1', row['p1Formes'][0], actual['p1'][0]['forme']),
                                         ('p2', row['p2Forme'], actual['p2Forme'])]:
                if expected and '-Mega' in expected and expected != seen:
                    errors.append({'kind': 'mega_forme', 'side': side, 'expected': expected, 'actual': seen})
            for i, side in enumerate(['p1', 'p2']):
                if row['faints'][i] != actual['faints'][i]:
                    errors.append({'kind': 'cumulative_faints', 'side': side,
                                   'expected': row['faints'][i], 'actual': actual['faints'][i]})
        for kind in {e['kind'] for e in errors}:
            counts[kind] += 1
            affected[kind].add(row['record'])
        if errors:
            mismatches.append({'record': row['record'], 'turn': row['turn'], 'errors': errors})
    report = {'schema_version': 1, 'until': args.until, 'records': sources, 'outcomes': dict(outcomes),
              'code_root': str(ROOT), 'source_sha256': {
                  n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in (
                      'services/sim-tools/state_codec.js', 'src/pokesim/live/session.py',
                      'src/pokesim/search/lookahead.py', 'src/pokesim/state/codec.py',
                      'src/pokesim/state/tracker.py', 'src/pokesim/sim/formes.py',
                      'src/pokesim/belief/observe.py')},
              'decisions': len(snapshots), 'mismatching_decisions': len(mismatches),
              'affected_battles': len({r['record'] for r in mismatches}),
              'by_kind': {k: {'decisions': v, 'battles': len(affected[k])} for k, v in counts.items()},
              'excluded': excluded, 'mismatches': mismatches,
              'scope': 'Observed-prefix state fidelity, not counterfactual wins or online win-rate improvement. '
                       'One prior world per normal recorded decision; no belief or future-set reconstruction.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in (
        'records', 'mismatches', 'source_sha256', 'excluded')}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--records', type=Path, default=ROOT / 'data/raw/ladder')
    ap.add_argument('--team', type=Path, default=ROOT / 'data/my_teams/baton.json')
    ap.add_argument('--dex', type=Path, default=ROOT / 'data/processed/public_value/training-dex-v1.json')
    ap.add_argument('--until', required=True, help='freeze the data cutoff, YYYYMMDD-HHMMSS')
    ap.add_argument('--out', type=Path, required=True)
    audit(ap.parse_args())


if __name__ == '__main__':
    main()
