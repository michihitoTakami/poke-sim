"""Rebuild learning rows from a frozen opponent-group split, preserving mixed-control provenance."""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from pokesim.config import ROOT
from pokesim.data.dataset import write_metadata
from pokesim.data.prior_bundle import load_bundle, sampler
from pokesim.evaluation.branch_matches import digest
from pokesim.learn.records import rows_of
from pokesim.runtime.bundle import file_hash
from pokesim.search.lookahead import Lookahead, to_id


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', type=Path, required=True)
    ap.add_argument('--priors', type=Path, required=True)
    ap.add_argument('--records', type=Path, default=ROOT / 'data/raw/ladder')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    spec = json.loads(args.split.read_text())
    groups = {}
    for name, split in spec['assignments'].items():
        opponent = spec['opponent_groups'][name]
        if opponent in groups and groups[opponent] != split:
            raise ValueError('an opponent appears in multiple splits')
        groups[opponent] = split
        if hashlib.sha256((args.records / name).read_bytes()).hexdigest() != spec['records'][name]:
            raise ValueError('source record changed')
    sets = json.loads((ROOT / 'data/my_teams/baton.json').read_text())
    engine = Lookahead(sampler(load_bundle(args.priors)))
    own = {engine.sampler.base_id(to_id(s['species'])): s for s in sets}
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        for split in ['train', 'development', 'calibration']:
            path = args.out / f'{split}.jsonl.gz'
            files = [name for name, s in spec['assignments'].items() if s == split]
            skipped_all, total = Counter(), 0
            with gzip.open(path, 'wt', encoding='utf-8') as out:
                for name in files:
                    rec = json.loads((args.records / name).read_text())
                    rows, skipped = rows_of(rec['source'], rec, sets, engine, own, split)
                    for row in rows:
                        out.write(json.dumps(row, ensure_ascii=False) + '\n')
                    total += len(rows)
                    skipped_all.update({k: v for k, v in skipped.items() if isinstance(v, int)})
                print(split, len(files), 'battles', total, 'rows', dict(skipped_all), flush=True)
            write_metadata(path, {'schema_version': 2, 'split': split, 'split_sha': file_hash(args.split),
                                 'records': {name: spec['records'][name] for name in files},
                                 'records_sha': digest({name: spec['records'][name] for name in files}),
                                 'rows': total, 'battles': len(files),
                                 'skipped': dict(skipped_all), 'information_scope': 'public_actor_observation',
                                 'control_mode': 'unknown; executed moves are not optimal-action labels',
                                 'prior_sha': file_hash(args.priors)})
    finally:
        engine.close()


if __name__ == '__main__':
    main()
