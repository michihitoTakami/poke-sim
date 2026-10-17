"""Write enriched copies of historical records; preserve recommendations and unknown actors without inference."""
import argparse
import json
from collections import Counter
from pathlib import Path

from pokesim.evaluation.branch_matches import atomic_json, file_sha
from pokesim.live.provenance import actual_lineup, execution_rows


def migrate(records: Path, out: Path):
    if out.exists():
        raise ValueError('choose a new output directory; source records are never overwritten')
    counts = Counter()
    manifest = {}
    for path in sorted(records.glob('*.json')):
        rec = json.loads(path.read_text())
        if not rec.get('log'):
            continue
        enriched = dict(rec, suggested_lineup=rec.get('suggested_lineup', rec.get('lineup')),
                        actual_lineup=actual_lineup(rec), executions=execution_rows(rec),
                        provenance_migration={'original_sha': file_sha(path), 'original_file': path.name})
        counts['records'] += 1
        counts['complete_lineups' if enriched['actual_lineup']['complete'] else 'partial_lineups'] += 1
        if (enriched['actual_lineup']['complete'] and
                set(enriched['actual_lineup']['species']) != set(rec.get('lineup') or [])):
            counts['recovered_different_from_suggestion'] += 1
        counts.update('actor:' + e['source'] for e in enriched['executions'])
        atomic_json(out / path.name, enriched)
        manifest[path.name] = file_sha(path)
    report = {'counts': dict(counts), 'original_records': manifest,
              'rule': 'own confirmed membership only; opponent history and unknown actors are unchanged'}
    atomic_json(out / 'migration.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.records, args.out)['counts'], ensure_ascii=False))


if __name__ == '__main__':
    main()
