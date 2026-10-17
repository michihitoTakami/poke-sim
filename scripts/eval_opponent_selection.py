"""Chronological opponent-preview predictions: fit each forecast on earlier completed games only."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from scipy.special import softmax

from pokesim.config import ROOT
from pokesim.live.opponent_selection import OpponentLineups, options, pick_rows, prior_logits
from pokesim.live.selection import LINEUP_PRIOR, LadderLineupModel, PickModel, ladder_rows
from pokesim.rl.myteam import to_id


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--records', type=Path, default=ROOT / 'data/raw/ladder')
    ap.add_argument('--team', type=Path, default=ROOT / 'data/my_teams/baton.json')
    ap.add_argument('--until', required=True)
    ap.add_argument('--last', type=int, default=60)
    ap.add_argument('--choices', action='store_true',
                    help='compare our recommended lineups with identical history/coverage')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    roster = [s['species'] for s in json.loads(args.team.read_text())]
    rows = [r for r in pick_rows(args.records, roster) if r['record'][:15] <= args.until]
    base, comparisons = PickModel(), []
    engine = coverage = None
    if args.choices:
        from pokesim.live.preview_matchups import PreviewCoverage
        from pokesim.search.lookahead import Lookahead, SetSampler

        team = json.loads(args.team.read_text())
        engine = Lookahead(SetSampler(['M-6'], observed={}))
        coverage = PreviewCoverage(team, engine)
        own_rows = ladder_rows(args.records, roster)
        prior = json.loads(LINEUP_PRIOR.read_text()) if LINEUP_PRIOR.exists() else None
    for i in reversed(range(min(args.last, max(0, len(rows) - 30)))):
        row, earlier = rows[i], rows[i + 1:]
        old = base.predict(row['preview'])
        new = OpponentLineups(earlier, base).predict(old)
        opts = options(old.species)
        mask = np.array([old.species[o[0]] in row['lead'] and all(
            set(shown) & {old.species[j] for j in o} for shown in row['shown']) for o in opts])
        if not mask.any():
            raise ValueError(f"no compatible lineup: {row['record']}")
        legacy = softmax(prior_logits(old))
        by_lineup = {tuple(to_id(s) for s in names): p for names, p in new.lineups}
        candidate = np.array([by_lineup[tuple(old.species[j] for j in o)] for o in opts])
        summary = {'record': row['record'], 'earlier_games': len(earlier),
                   'compatible_lineups': int(mask.sum()), 'preview': old.species,
                   'old_event_log_loss': float(-np.log(legacy[mask].sum())),
                   'new_event_log_loss': float(-np.log(candidate[mask].sum())),
                   'old_lead': old.species[int(old.lead.argmax())],
                   'new_lead': new.species[int(new.lead.argmax())],
                   'actual_lead': row['lead'], 'top_lineups': new.lineups[:3]}
        if len(row['lead']) == 1:
            summary['old_lead_correct'] = summary['old_lead'] in row['lead']
            summary['new_lead_correct'] = summary['new_lead'] in row['lead']
        if len(row['shown']) == 3 and all(len(s) == 1 for s in row['shown']):
            actual = {s[0] for s in row['shown']}
            target = np.array([s in actual for s in row['preview']], float)
            summary['old_member_brier'] = float(np.mean((old.brought - target) ** 2))
            summary['new_member_brier'] = float(np.mean((new.brought - target) ** 2))
        if coverage is not None:
            model = LadderLineupModel([r for r in own_rows if r['name'] < row['record']], prior=prior)
            profile = coverage.profile(old)
            summary['old_choice'], _ = model.choose(roster, random.Random(0), picks=old,
                                                    explore=False, coverage=profile)
            summary['new_choice'], info = model.choose(roster, random.Random(0), picks=new,
                                                       explore=False, coverage=profile)
            summary['new_choice_coverage'] = info['coverage']
        comparisons.append(summary)
    if engine:
        engine.close()
    metrics = {}
    for key in ('event_log_loss', 'lead_correct', 'member_brier'):
        subset = [r for r in comparisons if f'old_{key}' in r]
        metrics[key] = {'games': len(subset), **{
            name: float(np.mean([r[f'{name}_{key}'] for r in subset])) if subset else None
            for name in ('old', 'new')}}
    report = {'until': args.until, 'available_games': len(rows), 'metrics': metrics, 'comparisons': comparisons,
              'record_sha256': {r['record']: hashlib.sha256((args.records / r['record']).read_bytes()).hexdigest()
                                for r in rows},
              'source_sha256': {n: hashlib.sha256((ROOT / 'src/pokesim/live' / n).read_bytes()).hexdigest()
                                for n in ('selection.py', 'opponent_selection.py', 'preview_matchups.py')},
              'prior_sha': base.prior_sha,
              'changed_own_choices': sum(r.get('old_choice') != r.get('new_choice') for r in comparisons),
              'scope': 'Descriptive forward prediction on inspected ladder histories. Each forecast fits only '
                       'earlier games of the same visible own six. Partial appearances are censored. '
                       'This does not measure the effect of changing our lineup or human-game win rate.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(report, f, indent=1)
    print(json.dumps(metrics, indent=1))


if __name__ == '__main__':
    main()
