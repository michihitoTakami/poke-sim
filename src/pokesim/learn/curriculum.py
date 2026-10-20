"""Frozen, opponent/team-grouped branch curriculum. Never labels played moves as correct."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from pokesim.config import ROOT, SHOWDOWN_DIR
from pokesim.evaluation.branch_matches import (
    atomic_json,
    digest,
    ensure_manifest,
    file_sha,
    hash_seed,
    load_result,
    normalize_cases,
    run_case,
    tree_sha,
)
from pokesim.learn.branches import CONTINUATIONS, development_files

VERSION = 'grouped-branch-curriculum-v1'
SPLITS = ('train', 'development', 'calibration')


def components(records):
    """Transitive grouping: the same opponent OR the same preview cannot cross splits."""
    parent = {name: name for name in records}
    def root(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name
    seen = {}
    for name, rec in sorted(records.items()):
        for key in [('opponent', rec.get('opponent', '').lower()),
                    ('preview', tuple(sorted(rec.get('preview', {}).get('species', []))))]:
            if not key[1]:
                continue
            if key in seen:
                parent[root(name)] = root(seen[key])
            else:
                seen[key] = name
    groups = defaultdict(list)
    for name in records:
        groups[root(name)].append(name)
    return [sorted(g) for g in groups.values()]


def decisions(rec, limit=2):
    """Prioritize planning disagreements and difficult decisions, with equal access to wins and losses."""
    rows = {}
    for a in rec.get('advice', []):
        turn = a.get('turn')
        request = a.get('request') or {}
        if not isinstance(turn, int) or a.get('force') or not request.get('active') or turn in rows:
            continue
        if len(request.get('side', {}).get('pokemon', [])) != 3 or f'|turn|{turn}' not in rec.get('log', []):
            continue
        search = a.get('status', {}).get('search', {})
        relay = search.get('relay_plan') or {}
        priority = int(bool(relay) and not relay.get('followed', True)) * 4
        priority += min(3., max((e.get('sampling_error', 0) or 0 for e in search.get('root_evidence', [])), default=0))
        # Use no final result, future opponent moves, or manual/recommendation mismatch to choose a teacher.
        rows[turn] = priority
    ranked = sorted(rows, key=lambda t: (-rows[t], digest([VERSION, t])))
    return ranked[:limit]


def freeze(args):
    if any(n < 1 for n in args.groups) or args.positions_per_battle < 1:
        raise ValueError('positive group and position counts required')
    if (args.out / 'cohort.json').exists():
        raise ValueError('use a new frozen curriculum directory')
    records = {}
    for path in sorted(args.records.glob('*.json')):
        if path.name >= args.before:
            continue
        rec = json.loads(path.read_text())
        if rec.get('result') in {'win', 'loss'} and decisions(rec, args.positions_per_battle):
            records[path.name] = rec
    inherited = json.loads(args.inherit.read_text())['assignments'] if args.inherit else {}
    inspected = development_files()
    # These manifests are preserved, not rewritten. Conflicting connected groups are quarantined.
    groups, quarantined = [], []
    for names in components(records):
        restrictions = {inherited[n] for n in names if n in inherited}
        if set(names) & inspected:
            restrictions.add('development')
        if len(restrictions) > 1 or (restrictions and not restrictions <= set(SPLITS)):
            quarantined.extend(names)
            continue
        group = digest([VERSION, names])[:20]
        groups.append((group, names, next(iter(restrictions), None)))
    assignments, cases, analysis_groups = {}, [], {}
    quotas = dict(zip(SPLITS, args.groups))
    for split in SPLITS:
        count = 0
        for group, names, required in sorted(groups, key=lambda g: digest([args.seed, g[0]])):
            if group in assignments or required not in (None, split):
                continue
            assignments[group] = split
            # At most one source battle from a connected group; all turns and branches share the group.
            name = min(names, key=lambda n: digest([args.seed, n]))
            analysis_groups[name] = group
            for turn in decisions(records[name], args.positions_per_battle):
                cases.append({'file': name, 'turn': turn, 'split': split})
            count += 1
            if count == quotas[split]:
                break
        if count < quotas[split]:
            raise ValueError(f'not enough independent {split} groups: {count}/{quotas[split]}')
    names = sorted({r['file'] for r in cases})
    cohort = {'version': VERSION, 'before': args.before, 'seed': args.seed, 'cases': cases,
              'analysis_groups': analysis_groups, 'group_splits': assignments,
              'connected_members': {g: ns for g, ns, _ in groups if g in assignments},
              'records': {n: file_sha(args.records / n) for n in names}, 'quarantined': quarantined,
              'inherited_split_sha': file_sha(args.inherit) if args.inherit else None,
              'scope': 'historical grouped holdout; inspected development data are not a new prospective ladder test'}
    atomic_json(args.out / 'cohort.json', cohort)
    print(Counter(row['split'] for row in cases), 'positions from', len(names), 'independent groups', flush=True)


def generate(args):
    from pokesim.data.dataset import sidecar
    from pokesim.data.prior_bundle import load_bundle, sampler
    from pokesim.search.lookahead import Lookahead, to_id

    fixed_seed = hash_seed()
    cohort = json.loads((args.cohort / 'cohort.json').read_text())
    if cohort['version'] != VERSION:
        raise ValueError('unsupported curriculum')
    for name, expected in cohort['records'].items():
        if file_sha(args.records / name) != expected:
            raise ValueError('a frozen source record changed')
    prior = load_bundle(args.priors)
    meta = json.loads(sidecar(args.priors).read_text())
    if meta.get('split') != 'train' or meta.get('records'):
        raise ValueError('this grouped curriculum requires a published-only prior, without source battles')
    sets = json.loads(args.team.read_text())
    engine = Lookahead(sampler(prior, args.seed))
    engine.query_timeout = 180
    own = {engine.sampler.base_id(to_id(s['species'])): s for s in sets}
    splits = json.loads((args.out / 'split.json').read_text()) if (args.out / 'split.json').exists() else None
    split_contract = {'curriculum_sha': file_sha(args.cohort / 'cohort.json'),
                      'assignments': {n: cohort['group_splits'][g] for n, g in cohort['analysis_groups'].items()}}
    if splits is not None and splits != split_contract:
        raise ValueError('curriculum changed; use a new output directory')
    atomic_json(args.out / 'split.json', split_contract)
    try:
        for split in SPLITS:
            if split not in getattr(args, 'splits', SPLITS):
                continue
            cases = normalize_cases([c for c in cohort['cases'] if c['split'] == split])
            cfg = {'schema_version': 1, 'format': 'gen9championsbssregmc', 'positions': cases, 'split': split,
                   'information_scope': 'public_observation', 'worlds': args.worlds, 'seeds': [0, 1],
                   'world_seed': args.seed, 'max_turns': args.max_turns, 'continuations': CONTINUATIONS,
                   'restore_private_request': True, 'live_policies': True,
                   'trajectory_sample': {'continuations': args.value_steps},
                   'analysis_groups': cohort['analysis_groups'],
                   'battle_ids': {n: json.loads((args.records / n).read_text()).get('source', n)
                                  for n in {c['file'] for c in cases}}, 'records': cohort['records'],
                   'prior_sha': file_sha(args.priors), 'prior_manifest_sha': file_sha(sidecar(args.priors)),
                   'team_sha': file_sha(args.team), 'python_hash_seed': fixed_seed,
                   'python_code_sha': tree_sha(ROOT / 'src/pokesim', {'.py'}),
                   'worker_code_sha': tree_sha(ROOT / 'services/sim-tools', {'.js'}),
                   'engine_dist_sha': tree_sha(SHOWDOWN_DIR / 'dist', {'.js', '.json'}),
                   'continuation_net_sha': file_sha(args.policy) if args.policy else None,
                   'policy_nets': {'ours': str(args.policy.resolve())} if args.policy else {}}
            directory = args.out / split
            manifest = ensure_manifest(directory, cfg, {'purpose': VERSION})
            for case in cases:
                path = directory / 'positions' / f"{case['id']}.json"
                if path.exists():
                    load_result(path, manifest, case)
                    continue
                rec = json.loads((args.records / case['file']).read_text())
                try:
                    result = run_case(engine, sets, own, rec, case, manifest)
                except ValueError as error:
                    atomic_json(directory / 'skipped' / path.name, {'case': case, 'reason': str(error)})
                    print('skip', split, case['file'], case['turn'], str(error), flush=True)
                    continue
                from pokesim.live.provenance import execution_rows

                result['root_provenance'] = next((r for r in execution_rows(rec) if r['turn'] == case['turn']), {})
                result['payload_sha'] = digest({k: v for k, v in result.items() if k != 'payload_sha'})
                atomic_json(path, result)
                tally = dict(Counter(t['outcome'] for t in result['result']['trials']))
                print(split, case['file'], case['turn'], tally, flush=True)
    finally:
        engine.close()


def main():
    import atexit

    import poke_env.concurrency as concurrency

    atexit.unregister(concurrency.__clear_loop)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze_parser = sub.add_parser('freeze')
    freeze_parser.add_argument('--before', required=True)
    freeze_parser.add_argument('--groups', type=int, nargs=3, default=[60, 20, 20])
    freeze_parser.add_argument('--positions-per-battle', type=int, default=2)
    freeze_parser.add_argument('--inherit', type=Path)
    gen = sub.add_parser('generate')
    gen.add_argument('--cohort', type=Path, required=True)
    gen.add_argument('--priors', type=Path, required=True)
    gen.add_argument('--team', type=Path, default=ROOT / 'data/my_teams/baton.json')
    gen.add_argument('--policy', type=Path)
    gen.add_argument('--worlds', type=int, default=4)
    gen.add_argument('--max-turns', type=int, default=160)
    gen.add_argument('--value-steps', type=int, default=3)
    gen.add_argument('--splits', choices=SPLITS, nargs='+', default=list(SPLITS))
    for p in [freeze_parser, gen]:
        p.add_argument('--records', type=Path, required=True)
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--seed', type=int, default=20261021)
    args = parser.parse_args()
    (freeze if args.command == 'freeze' else generate)(args)


if __name__ == '__main__':
    main()
