"""Export cached Showdown games as public-state value pretraining data, without inventing private sets.

These human-policy terminal outcomes are auxiliary targets, not labels for the fixed simulation reference policy.
Uses the same stable 15% replay-ID holdout as rl.human; excludes all identifiable own ladder games.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from pokesim.config import FORMAT, ROOT, SHOWDOWN_DIR
from pokesim.evaluation.branch_matches import atomic_json, file_sha, tree_sha
from pokesim.runtime.bundle import describe


def battle_key(value: str) -> str:
    match = re.search(r'(gen\d+[a-z0-9]+-\d+)(?:-|$)', value)
    if not match:
        raise ValueError('missing Showdown battle ID')
    return match[1]


def replay_split(key: str) -> str:
    # Match rl.human.split exactly for ordinary public replay IDs. Both perspectives stay together.
    held = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 16**8 < .15
    return 'development' if held else 'train'


def outcome(log: str) -> float:
    lines = log.splitlines()
    players = {}
    results = []
    previous_turn = 0
    for line in lines:
        p = line.split('|')
        if len(p) > 3 and p[1] == 'player' and p[2] in {'p1', 'p2'} and p[3] and not results:
            players[p[2]] = p[3]
        if len(p) > 2 and p[1] == 'turn':
            if results or int(p[2]) <= previous_turn:
                raise ValueError('nonmonotonic or post-terminal turns')
            previous_turn = int(p[2])
        if len(p) > 2 and p[1] == 'win':
            results.append(p[2])
        elif len(p) > 1 and p[1] == 'tie':
            results.append(None)
    if (not previous_turn or len(results) != 1 or set(players) != {'p1', 'p2'} or
            not all(players.values()) or players['p1'] == players['p2']):
        raise ValueError('missing or ambiguous terminal outcome')
    winner = results[0]
    if winner is None:
        return .5
    if winner not in players.values():
        raise ValueError('winner is not a player')
    return float(winner == players['p1'])


def own_games(root: Path) -> tuple[set[str], dict]:
    ledger = root / 'data/curated/battles.jsonl'
    split = root / 'data/curated/split_manifest.json'
    # Require the existing frozen evaluation contract even if no own games overlap this corpus.
    json.loads(split.read_text())
    keys = set()
    hashes = {}
    for line in ledger.read_text().splitlines():
        row = json.loads(line)
        if row['source'] == 'showdown':
            keys.add(battle_key(row['battle_id']))
    # Include newly recorded games outside the old ledger, so a later download cannot bypass that exclusion.
    for path in sorted((root / 'data/raw/ladder').glob('*.json')):
        data = json.loads(path.read_text())
        if data.get('source'):
            keys.add(battle_key(data['source']))
        hashes[path.name] = file_sha(path)
    return keys, {'ledger_sha': file_sha(ledger), 'split_manifest_sha': file_sha(split),
                  'live_ladder_sha': hashes, 'excluded_battle_ids': sorted(keys)}


def build(root: Path, out: Path, engine: Path = SHOWDOWN_DIR) -> dict:
    root, out, engine = Path(root), Path(out), Path(engine)
    if out.exists():
        raise ValueError('refusing to overwrite an existing replay dataset directory')
    own, own_meta = own_games(root)
    selected, skipped, seen, logs = [], Counter(), set(), set()
    sources = {}
    files = sorted((root / 'data/raw/replays').glob('*.json'))
    for path in files:
        raw = path.read_bytes()
        entry = {'sha': hashlib.sha256(raw).hexdigest()}
        sources[path.name] = entry
        try:
            data = json.loads(raw)
            key = battle_key(data.get('id', path.stem))
            if key != battle_key(path.stem):
                raise ValueError('file and replay identity disagree')
            log = data['log']
            if data.get('formatid') != FORMAT or '|gametype|singles' not in log.splitlines():
                raise ValueError('unsupported format')
            target = outcome(log)
            log_sha = hashlib.sha256(log.encode()).hexdigest()
            entry.update(battle_id=key, log_sha=log_sha)
            if key in own:
                reason = 'own_ladder'
            elif key in seen or log_sha in logs:
                reason = 'duplicate'
            else:
                reason = None
            if reason:
                skipped[reason] += 1
                entry['excluded'] = reason
                continue
            seen.add(key)
            logs.add(log_sha)
            split = replay_split(key)
            entry['split'] = split
            selected.append((path, entry['sha'], key, split, target))
        except (KeyError, ValueError, TypeError) as exc:
            skipped['invalid'] += 1
            entry['excluded'] = str(exc)
    if not selected:
        raise ValueError('no eligible terminal Showdown replays')
    out.mkdir(parents=True)
    worker = ROOT / 'services/sim-tools/public_replays.js'
    counts, groups = Counter(), {'train': set(), 'development': set()}
    outputs = {s: out / f'{s}.jsonl.gz' for s in groups}
    process = subprocess.Popen(['node', str(worker), str(engine)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, text=True)
    handles = {}
    try:
        handles = {s: gzip.open(p.with_suffix('.gz.tmp'), 'wt', encoding='utf-8') for s, p in outputs.items()}
        for index, (path, expected, key, split, target) in enumerate(selected, 1):
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError('source replay changed during generation')
            log = json.loads(raw)['log']
            # The parser receives neither a target nor the terminal suffix. It only sees event prefixes.
            prefix = re.split(r'\n\|(?:win|tie)(?:\||\n|$)', '\n' + log, maxsplit=1)[0]
            process.stdin.write(json.dumps({'format': FORMAT, 'log': prefix}) + '\n')
            process.stdin.flush()
            answer = process.stdout.readline()
            if not answer:
                raise RuntimeError('public replay worker exited unexpectedly')
            reply = json.loads(answer)
            if reply.get('error'):
                skipped['public_parse_error'] += 1
                sources[path.name]['excluded'] = reply['error']
                sources[path.name].pop('split', None)
                continue
            states = reply['states']
            if not states or any(len(states[0]['preview'][s]) != 6 for s in ('p1', 'p2')):
                skipped['incomplete_preview'] += 1
                sources[path.name]['excluded'] = 'incomplete_preview'
                sources[path.name].pop('split', None)
                continue
            for state in states:
                if not all(state['active'].values()):
                    raise ValueError('turn boundary without both active Pokemon')
                for side in ('p1', 'p2'):
                    sample = {'schema_version': 1, 'split': split, 'group': f'public:{key}',
                              'source': 'showdown_replay', 'target_kind': 'terminal_outcome_under_replay_players',
                              'observation': {'schema': 1, 'side': side, 'turn': state['turn'],
                                              'phase': 'move', 'public': state, 'request': None},
                              'target': target if side == 'p1' else 1 - target,
                              'sample_weight': 1 / (2 * len(states))}
                    handles[split].write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + '\n')
                    counts[split] += 1
            groups[split].add(key)
            if index % 500 == 0:
                print(f'{index}/{len(selected)} public replays processed', flush=True)
        for handle in handles.values():
            handle.close()
        process.stdin.close()
        if process.wait(timeout=30):
            raise RuntimeError('public replay worker failed')
        if any(not groups[s] for s in groups):
            raise ValueError('need nonempty train and development public datasets')
        meta = {'schema_version': 1, 'information_scope': 'public_state_only',
                'purpose': 'auxiliary_public_value_pretraining',
                'target': 'terminal_outcome_under_replay_players', 'format': FORMAT,
                'private_information': 'unavailable; request=null; no reconstructed future sets',
                'split_rule': 'rl.human SHA1(replay_id) first32bits / 2**32 < .15 -> development',
                'groups': {s: sorted(ids) for s, ids in groups.items()}, 'rows': dict(counts),
                'source_files': sources, 'source_count': len(files), 'excluded': dict(skipped),
                'own_exclusion': own_meta, 'provenance': describe(),
                'python_code_sha': tree_sha(ROOT / 'src/pokesim', {'.py'}),
                'worker_code_sha': tree_sha(ROOT / 'services/sim-tools', {'.js'}),
                'engine_dist_sha': tree_sha(engine / 'dist', {'.js', '.json'}),
                'datasets': {s: {'file': p.name, 'sha': file_sha(p.with_suffix('.gz.tmp'))}
                             for s, p in outputs.items()}}
        atomic_json(out / 'manifest.json', meta)
        for p in outputs.values():
            p.with_suffix('.gz.tmp').rename(p)
        return meta
    finally:
        for handle in handles.values():
            handle.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        for p in outputs.values():
            p.with_suffix('.gz.tmp').unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    meta = build(args.data_root, args.out)
    print(json.dumps({k: meta[k] for k in ('source_count', 'rows', 'excluded')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
