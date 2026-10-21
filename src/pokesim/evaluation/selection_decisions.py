"""Separate lineup effects from action effects in paired public-information simulations."""
import argparse
import copy
import itertools
import json
import random
from pathlib import Path

import numpy as np

from pokesim.config import FORMAT, ROOT
from pokesim.evaluation.branch_matches import atomic_json, digest, ensure_manifest, file_sha, run_case, tally, tree_sha
from pokesim.learn.action_value import guarded_index, load_models, packed, probabilities
from pokesim.learn.branches import CONTINUATIONS, confident_pairs, encode_ranks, paired_interval
from pokesim.learn.curriculum import components, decisions
from pokesim.live.preview_matchups import PreviewCoverage
from pokesim.live.selection import OpponentPicks
from pokesim.rl.myteam import to_id

VERSION = 'selection-and-decision-audit-v1'
OPPONENTS = ['tactical', 'disrupt', 'expert', 'counterplay']


def save_result(path, value):
    atomic_json(path, dict(value, payload_sha=digest(value)))


def read_result(path):
    value = json.loads(path.read_text())
    expected = value.pop('payload_sha')
    if digest(value) != expected:
        raise ValueError('modified audit result')
    return value


def paired_gain(before, after, key_fields):
    """Bound unresolved pairs rather than silently calling them losses or dropping them."""
    def indexed(rows):
        out = {tuple(r[k] for k in key_fields): r['target'] for r in rows}
        if len(out) != len(rows):
            raise ValueError('duplicate paired comparison')
        return out
    a, b = indexed(before), indexed(after)
    if not a or a.keys() != b.keys():
        raise ValueError('missing paired comparison')
    lower, upper = [], []
    for key in a:
        old, new = a[key], b[key]
        lower.append((0 if new is None else new) - (1 if old is None else old))
        upper.append((1 if new is None else new) - (0 if old is None else old))
    return {'bounds': [float(np.mean(lower)), float(np.mean(upper))],
            'complete': all(a[k] is not None and b[k] is not None for k in a), 'pairs': len(a)}


def freeze_cases(records, commit, before, limit):
    eligible = {}
    for p in sorted(records.glob('*.json')):
        if p.name >= before:
            continue
        rec = json.loads(p.read_text())
        if (rec.get('versions', {}).get('code', {}).get('commit', '').startswith(commit) and
                rec.get('suggested_lineup') and decisions(rec, 1)):
            eligible[p.name] = rec
    cases = []
    normalized = {name: dict(rec, opponent=to_id(rec.get('opponent', ''))) for name, rec in eligible.items()}
    for names in sorted(components(normalized), key=digest)[:limit]:
        name = min(names, key=digest)
        rec = eligible[name]
        cases.append({'file': name, 'sha256': file_sha(records / name), 'group': digest(names)[:20],
                      'members': names, 'turn': decisions(rec, 1)[0]})
    if not cases:
        raise ValueError('no eligible independent source groups')
    return cases


def worlds(prior, catalog, preview, group, phase, count):
    prior.rng = random.Random(int(digest([VERSION, group, phase])[:16], 16))
    out = []
    for _ in range(count):
        trio = prior.rng.sample(preview, 3)
        joint = catalog.draw(preview, {}, prior.rng) if catalog is not None else {}
        out.append([joint.get(prior.base_id(s)) or prior.sample(s) for s in trio])
    return out


def candidates(sets, baseline, profile, picks):
    """Every three-member combination, with a lead shortlisted by static preview coverage."""
    pool = [baseline]
    for trio in itertools.combinations([s['species'] for s in sets], 3):
        orders = [[lead, *[s for s in trio if s != lead]] for lead in trio]
        best = max(orders, key=lambda names: profile.score(names, picks)[0])
        if best not in pool:
            pool.append(best)
    return pool


def lineup_audit(engine, prior, catalog, sets, case, rec, model, directory, worlds_count, max_turns,
                 reference_directory=None):
    path = directory / 'lineups' / (case['group'] + '.json')
    if path.exists():
        return read_result(path)['summary']
    if reference_directory is not None:
        original_path = reference_directory / 'lineups' / (case['group'] + '.json')
        saved = read_result(original_path)
        fixed_request = saved['fixed']['request']
        fixed = engine._query(fixed_request)
        if digest(fixed) != digest(saved['fixed']['result']):
            raise ValueError('reference actor no longer reproduces the frozen paired games')
        learned_request = copy.deepcopy(fixed_request)
        learned_request['lineupMatches']['actionValueRun'] = str(model.resolve())
        learned = engine._query(learned_request)
        if any(t['outcome'] == 'failed' for t in learned['trials']):
            raise ValueError('candidate actor simulation failed')
        cells = {name: [t for t in reply['trials'] if t['candidate'] == i]
                 for name, reply, i in [('original_reference', fixed, 0), ('alternative_reference', fixed, 1),
                                        ('original_q', learned, 0), ('alternative_q', learned, 1)]}
        summary = dict(saved['summary'])
        summary.update(cells={name: tally(rows) for name, rows in cells.items()},
            decision_effect_fixed_lineup=paired_gain(cells['original_reference'], cells['original_q'],
                                                     ('world', 'seed', 'opponent')),
            combined_effect=paired_gain(cells['original_reference'], cells['alternative_q'],
                                       ('world', 'seed', 'opponent')),
            q_changes=sum(t.get('actionValueChanges', 0) for t in learned['trials']),
            reference_actor_reproduced=True)
        save_result(path, {'summary': summary, 'discovery': saved['discovery'],
                          'fixed': {'request': fixed_request, 'result': fixed},
                          'learned': {'request': learned_request, 'result': learned},
                          'reused_source_sha': file_sha(original_path)})
        return summary
    preview, baseline = rec['preview']['species'], rec['suggested_lineup']
    picks = OpponentPicks(preview, np.full(6, .5), np.full(6, 1 / 6))
    profile = PreviewCoverage(sets, engine).profile(picks)
    pool = candidates(sets, baseline, profile, picks)
    own = {s['species']: s for s in sets}
    discovery = worlds(prior, catalog, preview, case['group'], 'discovery', worlds_count)
    evaluation = worlds(prior, catalog, preview, case['group'], 'evaluation', worlds_count)
    if digest(discovery) == digest(evaluation):
        raise ValueError('discovery and evaluation worlds are identical')

    def query(names, world_sets, action_value=False):
        cfg = {'ours': [[own[s] for s in trio] for trio in names], 'worlds': world_sets,
               'preview': {'p1': [s['species'] for s in sets], 'p2': preview},
               'maxTurns': max_turns, 'policies': OPPONENTS, 'seeds': [0, 1]}
        if action_value:
            cfg['actionValueRun'] = str(model.resolve())
        request = {'format': FORMAT, 'lineupMatches': cfg}
        reply = engine._query(request)
        if reply.get('informationScope') != 'public_observation':
            raise ValueError('lineup actor must only see public observation and own request')
        if any(r['outcome'] == 'failed' for r in reply['trials']):
            raise ValueError('lineup simulation failed')
        return request, reply

    discovery_request, found = query(pool, discovery)
    summaries = [tally([r for r in found['trials'] if r['candidate'] == i]) for i in range(len(pool))]
    selected = max(range(len(pool)), key=lambda i: (summaries[i]['target_bounds'][0], -i))
    chosen = [baseline, pool[selected]]
    fixed_request, fixed = query(chosen, evaluation)
    learned_request, learned = query(chosen, evaluation, True)
    cells = {name: [r for r in reply['trials'] if r['candidate'] == i]
             for name, reply, i in [('original_reference', fixed, 0), ('alternative_reference', fixed, 1),
                                    ('original_q', learned, 0), ('alternative_q', learned, 1)]}
    key = ('world', 'seed', 'opponent')
    summary = {'file': case['file'], 'group': case['group'], 'preview': preview,
        'ai_suggested_lineup': baseline, 'actual_lineup': rec.get('actual_lineup'),
        'alternative': pool[selected], 'candidates': len(pool),
        'cells': {name: tally(rows) for name, rows in cells.items()},
        'selection_effect_fixed_reference': paired_gain(
            cells['original_reference'], cells['alternative_reference'], key),
        'decision_effect_fixed_lineup': paired_gain(cells['original_reference'], cells['original_q'], key),
        'combined_effect': paired_gain(cells['original_reference'], cells['alternative_q'], key),
        'q_changes': sum(r.get('actionValueChanges', 0) for r in learned['trials']),
        'scope': 'fresh evaluation worlds after discovery; fixed public reference/Q actors, not live 15s search'}
    save_result(path, {'summary': summary, 'discovery': {'request': discovery_request, 'result': found},
                      'fixed': {'request': fixed_request, 'result': fixed},
                      'learned': {'request': learned_request, 'result': learned}})
    return summary


def decision_audit(engine, sets, own, case, rec, models, model_report, manifest, directory,
                   reference_directory=None):
    path = directory / 'decisions' / (case['group'] + '.json')
    if path.exists():
        return read_result(path)['summary']
    root_case = {'file': case['file'], 'turn': case['turn'], 'id': case['group'], 'actions': None}
    if reference_directory is None:
        row = run_case(engine, sets, own, rec, root_case, manifest)
    else:
        row = read_result(reference_directory / 'decisions' / (case['group'] + '.json'))['branch_result']
        if row['case'] != root_case:
            raise ValueError('reused candidate comparisons belong to another decision')
    reply, trials = row['result'], row['result']['trials']
    if any(t['outcome'] == 'failed' for t in trials):
        raise ValueError('candidate action simulation failed')
    a = next(a for a in rec['advice'] if a['turn'] == case['turn'] and not a.get('force'))
    ai = a.get('suggested') or a.get('chosen')
    actions = reply['actions']
    summary = {'file': case['file'], 'group': case['group'], 'turn': case['turn'], 'ai_proposal': ai,
               'control_mode': rec.get('control_mode', 'unknown'), 'actual_selected_team_fixed': True}
    if ai not in actions or any(t['target'] is None for t in trials):
        summary.update(status='incomplete', outcomes={name: tally([t for t in trials if t['action'] == name])
                                                    for name in actions})
    else:
        root = next(d for d in trials[0]['trajectory'] if d['side'] == 'p1' and d['rootIntervention'])
        targets = [float(np.mean([t['target'] for t in trials if t['action'] == name])) for name in actions]
        encoded = encode_ranks([{'view': root['observation'], 'actions': actions, 'targets': targets,
                                'pair_weights': np.zeros((len(actions), len(actions))).tolist(),
                                'group': case['group']}])
        probs = probabilities(models, packed(encoded))[:, 0]
        baseline = actions.index(ai)
        selected = guarded_index(probs, baseline, **model_report['guard_args'])
        best = int(np.argmax(targets))
        weights = confident_pairs(trials, actions, world_keys=[digest(s) for s in row['simulation_request']['samples']])
        summary.update(status='complete', learned_choice=actions[selected], conditional_best=actions[best],
            ai_target=targets[baseline], learned_target=targets[selected], best_target=targets[best],
            gain=targets[selected] - targets[baseline], regret=targets[best] - targets[baseline],
            robust_better_action=bool(weights[best][baseline]),
            scope='same selected team and observed prefix; all root actions share worlds/seeds/reference continuation')
    save_result(path, {'summary': summary, 'branch_result': row})
    return summary


def aggregate(lineups, actions):
    report = {'groups': len(lineups), 'lineup_comparison': {}, 'decision_comparison': {}}
    for field in ['selection_effect_fixed_reference', 'decision_effect_fixed_lineup', 'combined_effect']:
        rows = [r[field] for r in lineups]
        complete = [r['bounds'][0] for r in rows if r['complete']]
        report['lineup_comparison'][field] = {
            'bounds': np.mean([r['bounds'] for r in rows], axis=0).tolist(),
            'complete_groups': len(complete),
            'ci90_complete_groups': paired_interval(complete, list(range(len(complete)))) if complete else None}
    good = [r for r in actions if r['status'] == 'complete']
    report['decision_comparison'] = {'complete_groups': len(good),
        'robust_better_action_groups': sum(r['robust_better_action'] for r in good),
        'mean_conditional_regret': float(np.mean([r['regret'] for r in good])) if good else None,
        'candidate_gain': float(np.mean([r['gain'] for r in good])) if good else None,
        'candidate_gain_ci90': paired_interval([r['gain'] for r in good], [r['group'] for r in good]) if good else None}
    return report


def main():
    import atexit

    import poke_env.concurrency as concurrency

    from pokesim.belief.team_catalog import TeamCatalog
    from pokesim.data.dataset import sidecar
    from pokesim.data.prior_bundle import load_bundle, sampler
    from pokesim.search.lookahead import Lookahead

    atexit.unregister(concurrency.__clear_loop)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--records', type=Path, required=True)
    ap.add_argument('--commit', required=True)
    ap.add_argument('--before', required=True)
    ap.add_argument('--priors', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--team', type=Path, default=ROOT / 'data/my_teams/baton.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--groups', type=int, default=24)
    ap.add_argument('--worlds', type=int, default=4)
    ap.add_argument('--max-turns', type=int, default=160)
    ap.add_argument('--reuse', type=Path,
                    help='freeze old discovery/branches, reproduce reference games, recheck actor')
    args = ap.parse_args()
    if json.loads(sidecar(args.priors).read_text()).get('records'):
        raise ValueError('audit requires published-only priors')
    cases = freeze_cases(args.records, args.commit, args.before, args.groups)
    contract = {'version': VERSION, 'cases': cases, 'prior_sha': file_sha(args.priors),
                'model_sha': file_sha(args.model / 'report.json'), 'team_sha': file_sha(args.team),
                'python_code_sha': tree_sha(ROOT / 'src/pokesim', {'.py'}),
                'worker_code_sha': tree_sha(ROOT / 'services/sim-tools', {'.js'}),
                'catalog_sha': file_sha(ROOT / 'data/processed/top_teams.json'),
                'worlds': args.worlds, 'max_turns': args.max_turns, 'continuations': CONTINUATIONS,
                'seeds': [0, 1], 'world_seed': 20261021, 'information_scope': 'public_observation',
                'restore_private_request': True, 'live_policies': True, 'trajectory_sample': {'continuations': 1}}
    if args.reuse:
        old = json.loads((args.reuse / 'manifest.json').read_text())
        for key in ['cases', 'prior_sha', 'team_sha', 'worlds', 'max_turns', 'continuations', 'seeds']:
            if old['contract'][key] != contract[key]:
                raise ValueError('reused comparison conditions differ: ' + key)
        contract['reused_manifest_sha'] = file_sha(args.reuse / 'manifest.json')
    manifest = ensure_manifest(args.out, contract, {
        'purpose': 'historical diagnosis; no human labels or best-version baseline'})
    sets, models = json.loads(args.team.read_text()), load_models(args.model)
    models, model_report = models
    model_report['guard_args'] = {k: model_report['guard'][k] for k in ['margin', 'disagreement']}
    prior = sampler(load_bundle(args.priors))
    prior.catalog = TeamCatalog.load(prior.base_id)
    engine = Lookahead(prior)
    engine.query_timeout = 300
    own = {prior.base_id(to_id(s['species'])): s for s in sets}
    lineups, actions = [], []
    try:
        for i, case in enumerate(cases):
            if file_sha(args.records / case['file']) != case['sha256']:
                raise ValueError('source record modified')
            rec = json.loads((args.records / case['file']).read_text())
            lineups.append(lineup_audit(engine, prior, prior.catalog, sets, case, rec, args.model,
                                       args.out, args.worlds, args.max_turns, args.reuse))
            actions.append(decision_audit(engine, sets, own, case, rec, models, model_report, manifest, args.out,
                                         args.reuse))
            report = aggregate(lineups, actions)
            report.update(cases=lineups, actions=actions, gate={'passed': False,
                'reason': 'historical sources and simulator reference actors; no prospective adoption'})
            atomic_json(args.out / 'report.json', report)
            print(i + 1, '/', len(cases), case['file'], lineups[-1]['selection_effect_fixed_reference'],
                  actions[-1].get('regret'), flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
