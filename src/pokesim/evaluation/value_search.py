"""Frozen action-ranking validation on source battles unused for value training/epoch selection.

Development can be used for debugging. Calibration validation consumes those battles for this integration gate;
it is recorded explicitly and must not be described later as untouched calibration. Final test stays unopened.
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
from poke_env.environment import SinglesEnv

from pokesim.config import FORMAT, ROOT
from pokesim.data.dataset import DatasetScope
from pokesim.data.prior_bundle import load_bundle, sampler
from pokesim.evaluation.branch_matches import atomic_json, digest, file_sha, tree_sha, validate_trials
from pokesim.evaluation.shadow import position
from pokesim.live.advisor import OPPONENT_MODEL
from pokesim.live.opponent import OpponentPolicy
from pokesim.search.lookahead import Lookahead, field_state, label_to_action, to_id
from pokesim.search.planner import plan
from pokesim.value.evaluate import intervals
from pokesim.value.runtime import ValueEnsemble


def selected(results, battle, cloned):
    advice = plan(results, OPPONENT_MODEL, rho=.3, cloned=cloned or None, cloned_weight=.5)
    mask = SinglesEnv.get_action_mask(battle)
    return next(label for label, _, _ in advice.ranked()
                if (action := label_to_action(label, battle)) is not None and mask[action])


def validate(args):
    if args.out.exists():
        raise ValueError('choose a new validation output directory')
    scope = DatasetScope(root=args.root, ledger=args.root / 'data/curated/battles.jsonl',
                         manifest=args.root / 'data/curated/split_manifest.json')
    selected_battles = [b for b in scope.select(args.split) if b.source == 'showdown'][:args.limit]
    ids = {b.battle_id for b in selected_battles}
    records = dict(scope.records(args.split))
    excluded = [b.battle_id for b in selected_battles
                if '|turn|1' not in records[b.battle_id].get('log', []) or not records[b.battle_id].get('lineup')]
    selected_battles = [b for b in selected_battles if b.battle_id not in excluded]
    ids = {b.battle_id for b in selected_battles}
    if not selected_battles:
        raise ValueError('no eligible normal-turn source records')
    ensemble = ValueEnsemble(args.value_run)
    if args.split != 'development':
        for split in ('train', 'development'):
            meta = json.loads((args.root / f'data/processed/public_value/reference-{split}-v1.jsonl.manifest.json')
                              .read_text())
            if ids & set(meta['groups']):
                raise ValueError('validation source was used for value training or epoch selection')
    sets = json.loads((args.root / 'data/my_teams/baton.json').read_text())
    bundle_path = args.root / 'data/processed/scoped/priors.json'
    engine = Lookahead(sampler(load_bundle(bundle_path)), value_model=ensemble)
    cloned_path = args.root / 'checkpoints/bc_human.zip'
    opponent = OpponentPolicy(cloned_path) if cloned_path.exists() else None
    own = {engine.sampler.base_id(to_id(s['species'])): s for s in sets}
    contract = {'split': args.split, 'source': scope.metadata(args.split), 'battles': sorted(ids),
                'excluded_no_turn_or_lineup': excluded,
                'purpose': 'action-ranking development gate; calibration sources consumed, final test unopened',
                'value': ensemble.version, 'worlds': args.worlds, 'root_seed_base': 1000,
                'repeat_seed_base': 2000, 'terminal_seed_base': 32, 'terminal_seeds': args.seeds,
                'rollout': args.rollout, 'prior_sha': file_sha(bundle_path),
                'opponent_policy_sha': file_sha(cloned_path),
                'code_sha': tree_sha(ROOT / 'src/pokesim', {'.py'}),
                'worker_sha': tree_sha(ROOT / 'services/sim-tools', {'.js'})}
    args.out.mkdir(parents=True)
    atomic_json(args.out / 'manifest.json', contract)
    rows = []
    try:
        for battle_meta in selected_battles:
            rec = records[battle_meta.battle_id]
            turn = 3 if '|turn|3' in rec.get('log', []) else 1
            session, battle = position(rec, sets, turn)
            cloned = dict(opponent.predict(session.log, 'p1')) if opponent else {}
            seed = int(digest([battle_meta.battle_id, turn, 20261016])[:12], 16)

            def worlds(offset):
                engine.sampler.rng = random.Random(seed + offset)
                p1 = engine.own_side(battle, own)
                return [{'p1': p1, 'p2': engine.opponent_side(battle), 'field': field_state(battle)}
                        for _ in range(args.worlds)]

            samples = worlds(0)
            req = {'format': FORMAT, 'samples': samples, 'valueLeaves': True, 'observationLog': session.log,
                   'seeds': 1, 'seedBase': 1000, 'rollout': args.rollout,
                   'continuations': ['expert/tactical', 'expert/disrupt']}
            start = time.perf_counter()
            results = ensemble.apply(engine._query(req))
            seconds = time.perf_counter() - start
            learned = selected(results, battle, cloned)
            hp = copy.deepcopy(results)
            for result in hp:
                result['payoff'] = result['hpDiagnostic']
                result.pop('byContinuation', None)
            baseline = selected(hp, battle, cloned)
            req['seedBase'] = 2000
            repeated = selected(ensemble.apply(engine._query(req)), battle, cloned)
            actions = sorted({learned, baseline})
            trial_req = {'format': FORMAT, 'samples': worlds(999999), 'branches': {
                'informationScope': 'public_observation', 'observationLog': session.log,
                'actions': actions, 'seeds': list(range(32, 32 + args.seeds)), 'maxTurns': 200,
                'continuations': req['continuations']}}
            trials = engine._query(trial_req)
            validate_trials(trials, args.worlds, trial_req['branches']['seeds'], req['continuations'])
            outcomes = {}
            for action in actions:
                action_rows = [r for r in trials['trials'] if r['action'] == action]
                complete = all(r['target'] is not None for r in action_rows)
                known = sum(r['target'] for r in action_rows if r['target'] is not None)
                unknown = sum(r['target'] is None for r in action_rows)
                outcomes[action] = {'mean': float(np.mean([r['target'] for r in action_rows])) if complete else None,
                                    'bounds': [known / len(action_rows), (known + unknown) / len(action_rows)],
                                    'trials': len(action_rows),
                                    'failed': sum(r['outcome'] == 'failed' for r in action_rows),
                                    'unfinished': sum(r['outcome'] == 'unfinished' for r in action_rows)}
            row = {'battle_id': battle_meta.battle_id, 'turn': turn, 'learned': learned, 'hp': baseline,
                   'repeat': repeated, 'search_seconds': seconds, 'outcomes': outcomes,
                   'value_counts': {k: sum(r['value_status'][k] for r in results)
                                    for k in ('learned', 'terminal', 'failed')},
                   'trial_request': trial_req, 'trials': trials}
            rows.append(row)
            atomic_json(args.out / f'{len(rows):03d}.json', row)
            print(f"{len(rows)}/{len(selected_battles)} T{turn} learned={learned} hp={baseline} "
                  f"terminal={[outcomes[a]['mean'] for a in actions]} seconds={seconds:.2f}", flush=True)
    finally:
        engine.close()
    complete = [r for r in rows if all(v['mean'] is not None for v in r['outcomes'].values())]
    delta = [r['outcomes'][r['learned']]['mean'] - r['outcomes'][r['hp']]['mean'] for r in complete]
    bounds = [[0., 0.] if r['learned'] == r['hp'] else
              [r['outcomes'][r['learned']]['bounds'][0] - r['outcomes'][r['hp']]['bounds'][1],
               r['outcomes'][r['learned']]['bounds'][1] - r['outcomes'][r['hp']]['bounds'][0]] for r in rows]
    report = {'sources': len(rows), 'complete_sources': len(complete), 'disagreements': sum(r['learned'] != r['hp']
                                                                                                for r in rows),
              'repeat_changes': sum(r['learned'] != r['repeat'] for r in rows),
              'learned_mean': (float(np.mean([r['outcomes'][r['learned']]['mean'] for r in complete]))
                               if complete else None),
              'hp_mean': float(np.mean([r['outcomes'][r['hp']]['mean'] for r in complete])) if complete else None,
              'difference': float(np.mean(delta)) if delta else None,
              'difference_bounds_all_sources': np.mean(bounds, axis=0).tolist(),
              'failed_trials': sum(v['failed'] for r in rows for v in r['outcomes'].values()),
              'unfinished_trials': sum(v['unfinished'] for r in rows for v in r['outcomes'].values()),
              'paired_battle_ci95': intervals(delta) if delta else None,
              'search_seconds': {'median': float(np.median([r['search_seconds'] for r in rows])),
                                 'max': max(r['search_seconds'] for r in rows)},
              'contract_sha': digest(contract), 'model_frozen': True, 'live_win_rate_measured': False}
    atomic_json(args.out / 'report.json', report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--value-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=['development', 'calibration'], default='development')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--worlds', type=int, default=6)
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--rollout', type=int, default=2)
    args = parser.parse_args()
    if min(args.limit, args.worlds, args.seeds) < 1 or args.rollout < 0:
        parser.error('invalid validation budget')
    validate(args)


if __name__ == '__main__':
    main()
