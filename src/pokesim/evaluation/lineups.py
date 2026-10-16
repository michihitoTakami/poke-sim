"""Forward-chaining comparison of team-preview lineup predictors on this roster's rated ladder records.

  uv run python -m pokesim.evaluation.lineups [--records data/raw/ladder] [--last 400] [--out report.json]

For each rated game t (in time order) every predictor is fitted on the games before t only and asked for the
win chance of the lineup that was actually brought against that opponent's preview. Mean log loss and Brier over
the last `--last` games compare: the roster's overall rate, the current rule (simulated matchup table shrunk to
the trio's last-200 results), the ordered Beta posterior (pokesim.live.selection.LadderLineupModel), and that
posterior with a ridge logistic adjustment on the opponent's preview species by lead. Also reported: how often
Thompson sampling would have brought a different lineup than the greedy posterior, and the posterior means of
the lineups it would have brought.

Lineups were chosen with the opponents in view, so every number here is a selection estimate; a predictor that
explains past choices well is not thereby a causal comparison between lineups.
"""
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np

from pokesim.config import ROOT
from pokesim.live.selection import LADDER_PRIOR, LADDER_WINDOW, LadderLineupModel, MatchupModel, PickModel, ladder_rows

TEAM = ROOT / 'data/my_teams/baton.json'


def clip(p: float) -> float:
    return min(1 - 1e-3, max(1e-3, p))



def current_rule(previous: list[dict], row: dict, matchups: MatchupModel, picks: PickModel) -> float | None:
    """The deployed rule: simulated win chance shrunk to the trio's results in the last LADDER_WINDOW games."""
    p = matchups.win_prob(','.join(row['ordered']), picks.predict(row['preview']))
    if p is None:
        return None
    recent = previous[:LADDER_WINDOW]
    wins = sum(r['win'] for r in recent if r['trio'] == row['trio'])
    games = sum(r['trio'] == row['trio'] for r in recent)
    return (LADDER_PRIOR * p + wins) / (LADDER_PRIOR + games)


def evaluate(rows: list[dict], last: int, roster: list[str], refit: int = 20, seed: int = 0) -> dict:
    """rows: most recent first (as ladder_rows). Fits use rows after index t (older games) only."""
    matchups, picks = MatchupModel(), PickModel()
    rng = random.Random(seed)
    model = fitted_logistic = None
    preds: dict[str, list[tuple[float, bool]]] = {k: [] for k in ('global', 'current', 'beta', 'beta+preview')}
    explored, thompson_means, greedy_means, actual_means = 0, [], [], []
    targets = list(range(min(last, len(rows) - 1)))  # evaluation games, most recent first
    for n, t in enumerate(targets):
        previous = rows[t + 1:]
        row = rows[t]
        # The logistic part is refitted every `refit` games (the live model refits at every preview).
        if n % refit == 0 or model is None:
            model = LadderLineupModel(previous)
        else:
            model = LadderLineupModel(previous, preview=False)
            model.logistic = fitted_logistic
        fitted_logistic = model.logistic
        beta = model.mean(list(row['ordered']))
        preds['global'].append((clip(model.global_rate), row['win']))
        preds['beta'].append((clip(beta), row['win']))
        preds['beta+preview'].append((clip(model.mean(list(row['ordered']), row['preview'])), row['win']))
        p = current_rule(previous, row, matchups, picks)
        preds['current'].append((clip(p) if p is not None else clip(model.global_rate), row['win']))
        opponent = picks.predict(row['preview'])
        chosen, info = model.choose(roster, rng, picks=opponent)
        greedy, _ = model.choose(roster, rng, picks=opponent, explore=False)
        explored += chosen != greedy
        thompson_means.append(info['posterior_mean'])
        greedy_means.append(model.mean(greedy, row['preview']))
        actual_means.append(model.mean(list(row['ordered']), row['preview']))
    out = {'games': len(targets), 'records': len(rows), 'methods': {}}
    for name, pairs in preds.items():
        out['methods'][name] = {
            'log_loss': float(np.mean([-math.log(p if w else 1 - p) for p, w in pairs])),
            'brier': float(np.mean([(p - w) ** 2 for p, w in pairs])),
            'mean_prediction': float(np.mean([p for p, _ in pairs]))}
    out['actual_win_rate'] = float(np.mean([w for _, w in preds['beta']]))
    out['thompson'] = {'explored_share': explored / len(targets),
                       'posterior_mean_of_pick': float(np.mean(thompson_means)),
                       'posterior_mean_of_greedy_pick': float(np.mean(greedy_means)),
                       'posterior_mean_of_actual_lineup': float(np.mean(actual_means))}
    final = LadderLineupModel(rows, preview=False)
    out['lineups'] = {' / '.join(k): {'games': n, 'wins': sum(r['win'] for r in rows if r['ordered'] == k),
                                      'posterior': round(final.mean(list(k)), 3)}
                      for k, n in Counter(r['ordered'] for r in rows).most_common()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--records', default=str(ROOT / 'data/raw/ladder'))
    ap.add_argument('--team', default=str(TEAM))
    ap.add_argument('--last', type=int, default=400)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    roster = [s['species'] for s in json.loads(Path(args.team).read_text())]
    rows = ladder_rows(Path(args.records), roster)
    report = evaluate(rows, args.last, roster)
    print(f"{report['games']} evaluated games of {report['records']} records; actual win rate "
          f"{report['actual_win_rate']:.3f}")
    for name, m in report['methods'].items():
        print(f"  {name:14s} log loss {m['log_loss']:.4f}  Brier {m['brier']:.4f}  mean prediction "
              f"{m['mean_prediction']:.3f}")
    t = report['thompson']
    print(f"Thompson: explored {t['explored_share']:.0%} of previews; posterior mean of its pick "
          f"{t['posterior_mean_of_pick']:.3f}, greedy pick {t['posterior_mean_of_greedy_pick']:.3f}, "
          f"actual lineup {t['posterior_mean_of_actual_lineup']:.3f}")
    print('posterior now (ordered lineup: games, wins, mean):')
    for k, v in report['lineups'].items():
        print(f"  {v['games']:4d} {v['wins']:4d}  {v['posterior']:.3f}  {k}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
