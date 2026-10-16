"""Reproducible Baton Pass audit from completed protocol logs (attempts and handoffs counted separately).

uv run python -m pokesim.evaluation.baton --since 20261016 --until 20261016-104000 --out /tmp/baton.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from pokesim.config import RAW_DIR
from pokesim.rl.myteam import to_id

PASSERS = {'scolipede', 'umbreon'}


def audit(records: list[tuple[str, dict]]) -> dict:
    totals, moves, recipients, payloads, modes = Counter(), defaultdict(Counter), Counter(), Counter(), Counter()
    lineups, results = defaultdict(Counter), defaultdict(Counter)
    examples = defaultdict(list)

    def count(kind, file, turn, species, boosts):
        totals[kind] += 1
        if len(examples[kind]) < 5:
            examples[kind].append(dict(file=file, turn=turn, species=species, boosts=dict(boosts)))

    for file, rec in records:
        if rec.get('result') not in ('win', 'loss'):
            continue
        totals['battles'] += 1
        lineup = tuple(sorted(map(to_id, rec.get('lineup') or [])))
        lineups['/'.join(lineup)][rec['result']] += 1
        selected = PASSERS.intersection(lineup)
        for species in selected:
            totals[f'selected:{species}'] += 1
        turn, active, boosts, substitute, fainted, pending, last = 0, None, {}, False, False, None, None
        used, appeared, encore = False, False, False
        search = rec.get('versions', {}).get('planner', {}).get('search', {})
        modes['terminal' if search.get('terminalSearch') else 'other'] += 1
        for line in rec.get('log', []):
            e = line.split('|')
            if e[1:2] == ['-clearallboost']:
                boosts = {}
                continue
            if len(e) < 3:
                continue
            kind = e[1]
            positive = {s: v for s, v in boosts.items() if v > 0}
            if kind == 'turn':
                turn, pending = int(e[2]), None
                continue
            if not e[2].startswith('p1'):
                continue
            if kind in ('switch', 'drag'):
                incoming = to_id(e[3].split(',')[0])
                if pending and kind == 'switch' and not fainted:
                    totals['completed_passes'] += 1
                    recipients[incoming] += 1
                    payloads[','.join(f'{s}:{v:+d}' for s, v in sorted(pending['boosts'].items()) if v) or 'none'] += 1
                    boosts = pending['boosts'].copy()
                    substitute = pending['substitute']
                else:
                    if active and positive and not fainted and kind == 'switch':
                        totals['all_boosts_lost:switch'] += 1
                    if active in PASSERS and positive and not fainted:
                        count('boosts_lost:drag' if kind == 'drag' else 'boosts_lost:switch',
                              file, turn, active, boosts)
                    boosts, substitute = {}, False
                active, fainted, pending, last, encore = incoming, False, None, None, False
                appeared |= incoming in PASSERS
            elif kind == 'move' and '[from]' not in line:
                move = to_id(e[3])
                moves[active][move] += 1
                consecutive = (move in ('protect', 'detect') and last and last[0] == turn - 1 and
                               last[1] in ('protect', 'detect'))
                if consecutive:
                    count('consecutive_protect:encore' if encore else 'consecutive_protect',
                          file, turn, active, boosts)
                    if last[2] == 1:
                        totals['consecutive_protect_runs'] += 1
                last = turn, move, last[2] + 1 if consecutive else 1
                if move == 'batonpass':
                    used = True
                    totals['attempted_passes'] += 1
                    if not positive:
                        count('pass_without_positive_stages', file, turn, active, boosts)
                        if not substitute:
                            count('pass_without_stages_or_substitute', file, turn, active, boosts)
                    pending = dict(boosts=dict(boosts), substitute=substitute)
            elif kind == 'faint':
                if active in PASSERS:
                    totals[f'fainted:{active}'] += 1
                    if positive:
                        count('boosts_lost:faint', file, turn, active, boosts)
                fainted, pending = True, None
            elif kind in ('-fail', 'cant'):
                pending = None
            elif kind in ('-boost', '-unboost', '-setboost'):
                value = int(e[4])
                boosts[e[3]] = value if kind == '-setboost' else max(-6, min(6, boosts.get(e[3], 0) +
                                                                           value * (1 if kind == '-boost' else -1)))
            elif kind == '-clearboost':
                boosts = {}
            elif kind == '-clearpositiveboost':
                boosts = {s: min(0, v) for s, v in boosts.items()}
            elif kind == '-clearnegativeboost':
                boosts = {s: max(0, v) for s, v in boosts.items()}
            elif kind == '-invertboost':
                boosts = {s: -v for s, v in boosts.items()}
            elif kind in ('-start', '-end') and len(e) > 3 and e[3] == 'Substitute':
                substitute = kind == '-start'
            elif kind in ('-start', '-end') and len(e) > 3 and e[3] in ('Encore', 'move: Encore'):
                encore = kind == '-start'
        if used:
            results['used_baton'][rec['result']] += 1
        elif appeared:
            results['appeared_passer_without_baton'][rec['result']] += 1
    totals['uncompleted_pass_attempts'] = totals['attempted_passes'] - totals['completed_passes']
    return dict(totals=dict(totals), moves={s: dict(c) for s, c in moves.items()},
                recipients=dict(recipients), payloads=dict(payloads), modes=dict(modes),
                lineups={s: dict(c) for s, c in lineups.items()}, results={s: dict(c) for s, c in results.items()},
                examples=dict(examples))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dir', type=Path, default=RAW_DIR / 'ladder')
    ap.add_argument('--since', default='')
    ap.add_argument('--until', default='')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    records = [(p.name, json.loads(p.read_text())) for p in sorted(args.dir.glob('*.json'))
               if p.stem >= args.since and (not args.until or p.stem < args.until)]
    result = audit(records)
    result['files'] = [name for name, _ in records]
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
