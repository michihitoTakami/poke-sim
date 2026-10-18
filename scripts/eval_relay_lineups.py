"""Preview-only development diagnostics: static shortlist versus full relay admission on identical trials."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from pokesim.config import ROOT
from pokesim.data.prior_bundle import load_bundle, sampler
from pokesim.evaluation.branch_matches import atomic_json, file_sha, hash_seed, tree_sha
from pokesim.live.lineup_search import LineupSearch
from pokesim.live.preview_matchups import PreviewCoverage
from pokesim.live.selection import LadderLineupModel, PickModel
from pokesim.search.lookahead import Lookahead


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--records', type=Path, default=ROOT / 'data/raw/ladder')
    ap.add_argument('--priors', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    hash_seed()
    sets = json.loads((ROOT / 'data/my_teams/baton.json').read_text())
    roster = [s['species'] for s in sets]
    engine = Lookahead(sampler(load_bundle(args.priors)))
    engine.query_timeout = 120.
    selection, coverage = LineupSearch(sets, engine), PreviewCoverage(sets, engine)
    # The 93 records are a development dataset, including for the historical selector.
    model = LadderLineupModel.from_records(args.records, roster)
    pick_model = PickModel()
    report = {'scope': 'development preview diagnostics, not held-out or calibrated win rates',
              'records': str(args.records), 'priors_sha': file_sha(args.priors),
              'worker_sha': tree_sha(ROOT / 'services/sim-tools', {'.js'}), 'cases': []}
    seen = set()
    try:
        for case in json.loads(args.cases.read_text()):
            path = next(args.records.glob(case['stamp'] + '*.json'))
            if path.name in seen:
                continue
            seen.add(path.name)
            record = json.loads(path.read_text())
            preview = record['preview']['species']
            picks = pick_model.predict(preview, roster=roster, records=args.records)
            started = time.monotonic()
            ranking = model.rank(roster, picks, coverage.profile(picks))
            ranked, info = selection.rerank(ranking, picks)
            extra = info['additional_relay_candidates']
            static = [row for row in ranked if row[0] not in extra]
            winner = ranked[0][0]
            evidence = next(e for e in info['candidates'] if e['lineup'] == winner)
            transfers = [r.get('transfers', []) for r in evidence['relay']]
            trials = next(v['trials'] for k, v in selection.cache.items() if k.startswith(info['profile']))
            row = {'file': path.name, 'sha256': file_sha(path), 'preview': preview,
                   'chosen': winner, 'restricted_static_shortlist_choice': static[0][0],
                   'elapsed_seconds': time.monotonic() - started,
                   'outcome_counts': dict(Counter(t['outcome'] for t in trials)),
                   'chosen_payload_games': sum(bool(x) for x in transfers),
                   'chosen_repeated_transfer_games': sum(len(x) > 1 for x in transfers),
                   'evidence': info}
            report['cases'].append(row)
            atomic_json(args.out, report)
            print(path.name, row['chosen'], 'static-only', static[0][0],
                  f"{row['elapsed_seconds']:.2f}s", row['outcome_counts'], flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
