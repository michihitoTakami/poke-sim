"""Replay frozen previews and compare public adversaries; never submit orders or fit a model."""
import argparse
import atexit
import copy
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path

from eval_relay_tempo import ReplayWorker

from pokesim.belief.team_catalog import TeamCatalog
from pokesim.config import ROOT
from pokesim.evaluation.branch_matches import atomic_json, file_sha, hash_seed, tree_sha
from pokesim.live.lineup_search import LineupSearch, counterplay_worlds
from pokesim.live.opponent_selection import OpponentLineups, pick_rows
from pokesim.live.preview_matchups import PreviewCoverage
from pokesim.live.selection import LadderLineupModel, PickModel, ladder_rows
from pokesim.search.lookahead import Lookahead, SetSampler


def validate_choices(engine, prior, sets, pick_model, opps, report, out):
    """Fresh hypothetical worlds, separate from ranking; still the same inspected development previews."""
    result = {'scope': 'fresh paired hypothetical worlds; inspected previews, not unseen ladder games', 'cases': []}
    for case in report['cases']:
        preview = case['preview']
        picks = opps.predict(pick_model.predict(preview))
        seed = int(hashlib.sha256(json.dumps(['preview-validation-v1', preview]).encode()).hexdigest()[:16], 16)
        prior.rng = random.Random(seed)
        options, weights = zip(*picks.lineups)
        worlds = []
        for _ in range(8):
            names = prior.rng.choices(options, weights=weights, k=1)[0]
            joint = prior.catalog.draw(preview, {}, prior.rng)
            worlds.append([joint.get(prior.base_id(s)) or prior.sample(s) for s in names])
        stress, tags = counterplay_worlds(prior, picks)
        worlds.extend(stress)
        by_id = {prior.base_id(s['species']): s for s in sets}
        req = {'format': 'gen9championsbssregmc', 'lineupMatches': {
            'ours': [[by_id[prior.base_id(s)] for s in case[key]] for key in ['old_pick', 'new_pick']],
            'worlds': worlds, 'maxTurns': 160, 'policies': ['counterplay', 'disrupt', 'expert'],
            'preview': {'p1': [s['species'] for s in sets], 'p2': preview}}}
        reply = engine._query(req)
        assert reply['informationScope'] == 'public_observation'
        outcomes = {}
        for i, key in enumerate(['old_pick', 'new_pick']):
            rows = [r for r in reply['trials'] if r['candidate'] == i]
            assert len(rows) == len(worlds) * 3
            outcomes[key] = {k: sum(r['outcome'] == k for r in rows)
                             for k in ['win', 'loss', 'draw', 'unfinished', 'failed']}
        entry = {'file': case['file'], 'seed': seed, 'worlds': len(worlds), 'stress_tags': tags, 'outcomes': outcomes}
        result['cases'].append(entry)
        atomic_json(out / (case['file'][:15] + '.validation.json'), {'request': req, 'reply': reply})
        atomic_json(out / 'validation.json', result)
        print('fresh-world validation', case['file'], outcomes, flush=True)


def main():
    # poke-env's notebook cleanup waits on a loop unused by these offline replays.
    import poke_env.concurrency as concurrency

    atexit.unregister(concurrency.__clear_loop)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--records', type=Path, required=True)
    ap.add_argument('--baseline-root', type=Path, required=True)
    ap.add_argument('--observed', type=Path, required=True, help='frozen pre-cohort live prior')
    ap.add_argument('--cutoff', required=True, help='exclude all history at or after this timestamp')
    ap.add_argument('--stamps', nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--validate-only', action='store_true', help='compare saved selections on fresh worlds')
    args = ap.parse_args()
    hash_seed()
    spec = importlib.util.spec_from_file_location('baseline_lineup',
        args.baseline_root / 'src/pokesim/live/lineup_search.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sets = json.loads((ROOT / 'data/my_teams/baton.json').read_text())
    roster = [s['species'] for s in sets]
    prior = SetSampler(['M-6'], observed=json.loads(args.observed.read_text()))
    prior.catalog = TeamCatalog.load(prior.base_id, path=ROOT / 'data/processed/top_teams.json')
    after, before = Lookahead(prior), Lookahead(prior)
    before.worker.kill()
    before.worker = ReplayWorker(args.baseline_root / 'services/sim-tools/lookahead.js')
    before.query_timeout = after.query_timeout = 120
    model = LadderLineupModel([r for r in ladder_rows(args.records, roster) if r['name'] < args.cutoff],
                             prior=json.loads((ROOT / 'data/lineup_prior.json').read_text()))
    pick_model = PickModel(ROOT / 'data/processed/pick_rates.json')
    opps = OpponentLineups([r for r in pick_rows(args.records, roster) if r['record'] < args.cutoff],
                          pick_model, signature=args.cutoff)
    selection, old_selection = LineupSearch(sets, after), module.LineupSearch(sets, before)
    coverage = PreviewCoverage(sets, after)
    report = {'scope': 'inspected development previews; shared public-only stress worlds, not prospective win rates',
              'history_before': args.cutoff, 'observed_sha': file_sha(args.observed),
              'worker_sha': {'before': tree_sha(args.baseline_root / 'services/sim-tools', {'.js'}),
                             'after': tree_sha(ROOT / 'services/sim-tools', {'.js'})}, 'cases': []}
    try:
        if args.validate_only:
            report = json.loads((args.out / 'report.json').read_text())
            validate_choices(after, prior, sets, pick_model, opps, report, args.out)
            return
        for stamp in args.stamps:
            path, = args.records.glob(stamp + '*.json')
            rec = json.loads(path.read_text())
            picks = opps.predict(pick_model.predict(rec['preview']['species']))
            ranking = model.rank(roster, picks, coverage.profile(picks))
            t = time.monotonic()
            old_ranked, old_info = old_selection.rerank(ranking, picks)
            old_seconds = time.monotonic() - t
            calls, original = [], after._query
            def capture(req):
                t = time.monotonic()
                reply = original(req)
                calls.append({'request': copy.deepcopy(req), 'seconds': time.monotonic() - t})
                return reply
            after._query = capture
            t = time.monotonic()
            ranked, info = selection.rerank(ranking, picks)
            seconds = time.monotonic() - t
            after._query = original
            final = next(c for c in calls if not c['request']['lineupMatches'].get('oursPolicy'))
            req = final['request']
            paired_old = before._query(req)
            new = selection.cache[next(reversed(selection.cache))]
            # The old selector's trio is also compared against the same new adversarial worlds.
            paired = []
            for tag, names in [('old_pick', old_ranked[0][0]), ('new_pick', ranked[0][0])]:
                teams = [[s['species'] for s in team] for team in req['lineupMatches']['ours']]
                i = teams.index(names)
                for worker, reply in [('old_opponents', paired_old), ('new_opponents', new)]:
                    trials = [r for r in reply['trials'] if r['candidate'] == i]
                    paired.append({'pick': tag, 'lineup': names, 'worker': worker,
                                   'outcomes': {k: sum(r['outcome'] == k for r in trials)
                                                for k in ['win', 'loss', 'draw', 'unfinished', 'failed']},
                                   'trials': len(trials)})
            entry = {'file': path.name, 'sha256': file_sha(path), 'preview': rec['preview']['species'],
                     'old_pick': old_ranked[0][0], 'new_pick': ranked[0][0],
                     'old_seconds': old_seconds, 'seconds': seconds,
                     'query_seconds': [c['seconds'] for c in calls], 'paired': paired,
                     'old_info': old_info, 'info': info}
            atomic_json(args.out / (stamp + '.requests.json'), calls)
            atomic_json(args.out / (stamp + '.paired.json'), {'old': paired_old, 'new': new})
            report['cases'].append(entry)
            atomic_json(args.out / 'report.json', report)
            print(path.name, old_ranked[0][0], '->', ranked[0][0], round(seconds, 2), paired, flush=True)
    finally:
        before.close()
        after.close()


if __name__ == '__main__':
    main()
