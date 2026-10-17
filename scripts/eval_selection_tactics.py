"""Replay preview and move choices from completed records without sending any Showdown orders.

Every preview model uses earlier records only. Tactical A/B decisions share the exact same search matrices.
Optional continuations are simulator diagnostics with hypothetical sets, not estimates of human win rate.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

from pokesim.config import ROOT
from pokesim.live.preview_matchups import PreviewCoverage
from pokesim.live.selection import LINEUP_PRIOR, LadderLineupModel, PickModel, ladder_rows
from pokesim.search.lookahead import Lookahead, SetSampler


def save(path, value):
    path.write_text(json.dumps(value, indent=1))


def preview(args):
    source = subprocess.check_output(['git', 'show', f'{args.baseline_ref}:src/pokesim/live/selection.py'], text=True)
    path = args.out / 'selection-before.py'
    path.write_text(source)
    spec = importlib.util.spec_from_file_location('selection_before', path)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    sets = json.loads(args.team.read_text())
    roster = [s['species'] for s in sets]
    rows = [r for r in ladder_rows(args.records, roster) if not args.until or r['started'] <= args.until]
    prior = json.loads(LINEUP_PRIOR.read_text()) if LINEUP_PRIOR.exists() else None
    engine = Lookahead(SetSampler(['M-6'], observed={}))
    coverage = PreviewCoverage(sets, engine)
    pick_model = PickModel()
    comparisons, matches = [], []
    try:
        for t in reversed(range(min(args.last, max(0, len(rows) - 20)))):
            row, previous = rows[t], rows[t + 1:]
            picks = pick_model.predict(row['preview'])
            before = legacy.LadderLineupModel(previous, prior=prior)
            after = LadderLineupModel(previous, prior=prior)
            old, _ = before.choose(roster, random.Random(0), picks=picks, explore=False)
            new, info = after.choose(roster, random.Random(0), picks=picks, explore=False,
                                     coverage=coverage.profile(picks))
            comparisons.append({'record': row['name'], 'preview': row['preview'], 'result': row['win'],
                                'before': old, 'after': new, 'coverage': info['coverage'],
                                'before_p_actual': before.mean(list(row['ordered']), row['preview']),
                                'after_p_actual': after.mean(list(row['ordered']), row['preview'])})
            print(row['name'], old, '->', new, flush=True)
        save(args.out / 'selection-comparison.json', comparisons)
        if args.simulate:
            changed = [r for r in comparisons if r['before'] != r['after']][:12]
            for index, row in enumerate(changed):
                picks = pick_model.predict(row['preview'])
                rng, foes = random.Random(index + 170018), []
                for k in range(3):
                    lead = rng.choices(picks.species, weights=picks.lead)[0]
                    bench, chosen = [s for s in picks.species if s != lead], [lead]
                    for _ in range(2):
                        sid = rng.choices(bench, weights=[float(picks.brought[picks.species.index(s)])
                                                        for s in bench])[0]
                        chosen.append(sid)
                        bench.remove(sid)
                    engine.sampler.rng = random.Random(index * 10 + k)
                    foes.append({'team': [engine.sampler.sample(s) for s in chosen], 'mons': [{'hp': 1}] * 3})
                outcomes = {}
                for name in ('before', 'after'):
                    team = [next(s for s in sets if s['species'] == n) for n in row[name]]
                    req = {'format': 'gen9championsbssregmc', 'samples': [
                        {'p1': {'team': team, 'mons': [{'hp': 1}] * 3}, 'p2': foe, 'field': {}} for foe in foes],
                        'plans': {'candidates': ['expert'], 'opponents': ['expert', 'disrupt'],
                                  'seeds': 3, 'maxTurns': 100}}
                    outcomes[name] = engine._query(req)['plans']['expert']
                matches.append({'record': row['record'], 'before': row['before'], 'after': row['after'],
                                'outcomes': outcomes})
                save(args.out / 'selection-matches.json', matches)
                print('simulation', matches[-1], flush=True)
    finally:
        engine.close()


def moves(args):
    from pokesim.evaluation.shadow import position
    from pokesim.live.advisor import Advisor
    from pokesim.search.planner import plan, predict
    from pokesim.search.terminal import search

    sets = json.loads(args.team.read_text())
    cases = json.loads(args.cases.read_text())
    advisor = Advisor(sets, policy=None, planner={'mode': 'on'}, belief=True,
                      sampler=SetSampler(['M-6'], observed={}), opponent_policy=args.opponent_policy,
                      value_run=args.value_run)
    rows, branches = [], []
    try:
        for case in cases:
            stamp, turn = case['stamp'], case['turn']
            path = next(args.records.glob(stamp + '*.json'))
            record = json.loads(path.read_text())
            session, battle = position(record, sets, turn)
            belief = advisor.beliefs.get(stamp, session.log, 'p1')
            req = advisor.engine.query(battle, advisor.own_sets, None, advisor.planner['search'],
                                       belief=belief, log=session.log, our_role='p1', value=True, capture=True)
            req.pop('value_model', None)
            req.update(decisionMs=args.decision_ms, workers=2)
            fixture = args.out / f'{stamp}-T{turn}.json'
            save(fixture, req)
            results = search(advisor.engine, copy.deepcopy(req), predictor=lambda m, labels, prior=None:
                             predict(m, labels, {}, session.last_move('p2'), None, .5, prior))
            before = copy.deepcopy(results)
            for item in before:
                item.pop('rootProgress', None)
            old = plan(before, pure=True, last_move=session.last_move('p2'))
            new = plan(results, pure=True, last_move=session.last_move('p2'))
            summary = results[0]['terminal_search']
            row = {'record': path.name, 'turn': turn, 'before': old.ranked()[0][0],
                   'after': new.ranked()[0][0], 'seconds': summary['elapsed_ms'] / 1000,
                   'failed': summary['failed'], 'fixture': str(fixture)}
            rows.append(row)
            save(args.out / 'move-comparison.json', rows)
            print(row, flush=True)
            if args.simulate and row['before'] != row['after']:
                reply = advisor.engine._query({'format': req['format'], 'samples': req['samples'][:3], 'branches': {
                    'seeds': [0, 1, 2, 3], 'continuations': ['expert/tactical', 'expert/disrupt'],
                    'maxTurns': 100, 'actions': [row['before'], row['after']]}})
                outcomes = {action: dict(Counter(t['outcome'] for t in reply['trials'] if t['action'] == action))
                            for action in (row['before'], row['after'])}
                branches.append({**row, 'outcomes': outcomes})
                save(args.out / 'move-branches.json', branches)
    finally:
        advisor.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['preview', 'moves', 'all'], default='all')
    parser.add_argument('--records', type=Path, default=ROOT / 'data/raw/ladder')
    parser.add_argument('--team', type=Path, default=ROOT / 'data/my_teams/baton.json')
    parser.add_argument('--cases', type=Path, default=ROOT / 'docs/audits/20261018-selection-tactical-cases.json')
    parser.add_argument('--baseline-ref', default='5f7b33c')
    parser.add_argument('--last', type=int, default=40)
    parser.add_argument('--until', default='', help='inclusive timestamp cutoff for a reproducible snapshot')
    parser.add_argument('--decision-ms', type=int, default=4500)
    parser.add_argument('--value-run', type=Path, default=ROOT / 'runs/value-replays-v1')
    parser.add_argument('--opponent-policy', default=f'net:{ROOT}/runs/opp-replays-v1/opp-seed0.json')
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {'settings': {k: str(v) for k, v in vars(args).items()}, 'records': {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.records.glob('*.json'))
        if not args.until or p.name[:15] <= args.until},
        'scope': 'development replay and hypothetical-set simulation only; no live orders, no human win-rate claim'}
    save(args.out / 'manifest.json', manifest)
    if args.mode in ('all', 'preview'):
        preview(args)
    if args.mode in ('all', 'moves'):
        moves(args)


if __name__ == '__main__':
    main()
