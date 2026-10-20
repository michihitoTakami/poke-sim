"""Re-run saved private requests and their public prefixes, without playing or treating saved moves as labels."""
import argparse
import atexit
import copy
import json
import time
from pathlib import Path

from pokesim.belief.team_catalog import TeamCatalog
from pokesim.config import ROOT
from pokesim.evaluation.branch_matches import atomic_json, file_sha, hash_seed, tree_sha
from pokesim.live.advisor import Advisor
from pokesim.live.provenance import base, reconstruction_lineup
from pokesim.live.session import BattleSession
from pokesim.search import review
from pokesim.search.lookahead import SetSampler


def main():
    import poke_env.concurrency as concurrency

    atexit.unregister(concurrency.__clear_loop)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--records', type=Path, required=True)
    ap.add_argument('--observed', type=Path, required=True, help='frozen pre-cohort prior')
    ap.add_argument('--cases', nargs='+', required=True, help='YYYYMMDD-HHMMSS:turn')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    hash_seed()
    sets = json.loads((ROOT / 'data/my_teams/baton.json').read_text())
    prior = SetSampler(['M-6'], observed=json.loads(args.observed.read_text()))
    prior.catalog = TeamCatalog.load(prior.base_id)
    advisor = Advisor(sets, policy=None, planner={'mode': 'on'}, source='showdown', sampler=prior,
                      opponent_policy='net:runs/opp-replays-v1/opp-seed0.json', value_run='runs/value-replays-v1')
    report = {'scope': 'inspected development prefixes; saved choices are not paired trials or teacher labels',
              'observed_sha': file_sha(args.observed),
              'worker_sha': tree_sha(ROOT / 'services/sim-tools', {'.js'}), 'decisions': []}
    original, requests = review.search_with_review, []
    def capture(engine, request, **kwargs):
        requests.append(copy.deepcopy(request))
        return original(engine, request, **kwargs)
    review.search_with_review = capture
    try:
        for case in args.cases:
            stamp, turn_text = case.split(':')
            turn = int(turn_text)
            path, = args.records.glob(stamp + '*.json')
            rec = json.loads(path.read_text())
            old = next(a for a in rec['advice'] if a['turn'] == turn and not a['force'])
            session = BattleSession(sets)
            session.preview(rec['preview']['species'])
            names, _ = reconstruction_lineup(rec, turn)
            session.select([next(i for i, s in enumerate(sets) if base(s['species']) == base(n)) for n in names])
            log = rec['log']
            start = next(i for i, line in enumerate(log) if line.startswith('|teampreview')) + 1
            end = log.index(f'|turn|{turn}') + 1
            session.apply([line for line in log[start:end] if not line.startswith('|error|')])
            battle = session.decision()
            request = copy.deepcopy(old['request'])
            request['side']['id'] = 'p1'
            for mon in request['side']['pokemon']:
                mon['ident'] = 'p1:' + mon['ident'].split(':', 1)[1]
            battle.parse_request(request)
            requests.clear()
            started = time.monotonic()
            answer = advisor.advise(battle, session.last_move(), log=session.log)
            chosen = answer.actions[0][0] if answer.actions else None
            row = {'record': path.name, 'sha256': file_sha(path), 'turn': turn,
                   'old_saved_choice': old.get('chosen'), 'chosen': chosen,
                   'seconds': time.monotonic() - started, 'status': answer.status}
            assert chosen and not answer.status.get('fallback'), row
            assert not answer.status['search']['terminal_search']['failed'], row
            atomic_json(args.out / f'{stamp}-T{turn}.request.json', requests[0])
            report['decisions'].append(row)
            atomic_json(args.out / 'report.json', report)
            print(stamp, turn, old.get('chosen'), '->', chosen, round(row['seconds'], 2), flush=True)
    finally:
        review.search_with_review = original
        advisor.close()


if __name__ == '__main__':
    main()
