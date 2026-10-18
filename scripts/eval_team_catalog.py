"""Forward revealed-move inclusion diagnostics; published sets are hypotheses, not action probabilities."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from pokesim.belief.team_catalog import TeamCatalog, ident
from pokesim.config import ROOT
from pokesim.data.prior_bundle import load_bundle, sampler


def evaluate(spec, records, split, prior, draws):
    engine = sampler(prior)
    catalog = TeamCatalog.load(engine.base_id)
    rows = []
    for name, part in spec['assignments'].items():
        if part != split:
            continue
        rec = json.loads((records / name).read_text())
        preview = rec['preview']['species']
        known, names, events = {}, {}, []
        for line in rec['log']:
            p = line.split('|')
            if len(p) < 3 or not p[2].startswith('p2'):
                continue
            if p[1] in ['switch', 'drag', 'replace']:
                sid = engine.base_id(ident(p[3].split(',')[0]))
                names[p[2]] = sid
                known.setdefault(sid, {'moves': [], 'item': None, 'ability': None})
            sid = names.get(p[2])
            if sid is None:
                continue
            k = known[sid]
            if p[1] == 'move' and '[from]' not in line:
                move = ident(p[3])
                if move not in k['moves']:
                    event = {'species': sid, 'move': move, 'seen_moves': list(k['moves'])}
                    seed = int(hashlib.sha256((name + line).encode()).hexdigest()[:16], 16)
                    for variant in ['independent', 'catalog']:
                        engine.catalog = catalog if variant == 'catalog' else None
                        engine.rng = random.Random(seed)
                        samples = [engine.sample(sid, **k, preview=preview) for _ in range(draws)]
                        mass = np.mean([move in set(map(ident, s['moves'])) for s in samples])
                        event[variant] = {'inclusion': float(mass), 'log_loss': float(-np.log(max(.001, mass)))}
                    events.append(event)
                    k['moves'].append(move)
            if p[1] in ['-item', '-enditem']:
                k['item'] = ident(p[3]) if p[1] == '-item' else ''
            if p[1] == '-mega':
                k['item'] = ident(p[4])
            if p[1] == '-ability':
                k['ability'] = ident(p[3])
        if events:
            rows.append({'file': name, 'opponent': rec['opponent'],
                         'retrieval': catalog.info(preview), 'events': events})
    metrics = {}
    for variant in ['independent', 'catalog']:
        metrics[variant] = {key: float(np.mean([np.mean([e[variant][key] for e in r['events']]) for r in rows]))
                            for key in ['inclusion', 'log_loss']}
    return {'scope': 'public-prefix forecasts of next revealed move adoption; not calibrated action probability',
            'split': split, 'battles': len(rows), 'draws': draws, 'catalog_sha': catalog.sha,
            'matched_previews': sum(r['retrieval']['matched'] > 0 for r in rows), 'metrics': metrics, 'rows': rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split-manifest', type=Path, required=True)
    ap.add_argument('--split', choices=['train', 'development', 'calibration'], required=True)
    ap.add_argument('--records', type=Path, default=ROOT / 'data/raw/ladder')
    ap.add_argument('--priors', type=Path, required=True)
    ap.add_argument('--draws', type=int, default=192)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    report = evaluate(json.loads(args.split_manifest.read_text()), args.records, args.split,
                      load_bundle(args.priors), args.draws)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    print({k: v for k, v in report.items() if k != 'rows'})


if __name__ == '__main__':
    main()
