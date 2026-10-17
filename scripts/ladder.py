"""Play our team on a Showdown server with the advisor (policy + lookahead), saving every battle as a record.

  # public server (needs a registered account): credentials in ~/.config/poke-sim/showdown.env
  #   PS_USERNAME=...   PS_PASSWORD=...
  uv run python scripts/ladder.py --battles 30
  # A/B: battles alternate between the policies (the record says which one played)
  # On the public server it exits between battles when the code or priors change, so a loop restarting it keeps
  # playing the latest version (see /data/poke-sim/ladder_logs/run3h.sh).
  uv run python scripts/ladder.py --battles 30 --policy checkpoints/ppo_baton_v2.zip checkpoints/ppo_baton_items.zip
  # local test against a random player
  uv run python scripts/ladder.py --local --battles 2

Team preview: a lineup drawn from the posterior of this roster's own rated results against the opponent's 6
(pokesim.live.selection.LadderLineupModel; --lineup matchup: the simulated matchup table instead). Each turn: the
advisor's top legal action, or with --sample an action drawn from its distribution (the seed and the chosen
action's probability are saved: a mixed strategy played as argmax is no longer mixed). --planner shadow|on runs
the common-strategy planner (pokesim.search.planner) next to / instead of the current advice. Each record says
which code, policy and lineup model played (pokesim.runtime.bundle).
Records go to data/raw/ladder/<time>.json in the format of data/raw/live/logs (opponent preview, lineup, protocol
log, result), so pokesim.live.observed and scripts/eval_records.py can use them.
Showdown is not the Switch game: mechanics follow Showdown's Champions format and opponents are Showdown players.
"""
import argparse
import asyncio
import json
import os
import random
import re
import time
import uuid
from pathlib import Path

import numpy as np
from poke_env import AccountConfiguration, LocalhostServerConfiguration, ShowdownServerConfiguration
from poke_env.environment import SinglesEnv
from poke_env.player import RandomPlayer
from sb3_contrib import MaskablePPO

from pokesim.config import FORMAT, PROCESSED_DIR, RAW_DIR, ROOT
from pokesim.live.advisor import Advisor
from pokesim.live.preview_matchups import PreviewCoverage
from pokesim.live.selection import LadderLineupModel, MatchupModel, PickModel
from pokesim.rl.bc import CounterPickHeuristic
from pokesim.rl.myteam import FixedTeam, TeamFeatures, counter_preview, to_id
from pokesim.rl.opponents import LeaguePlayer
from pokesim.runtime.bundle import describe, file_hash
from pokesim.search.lookahead import label_to_action
from pokesim.sim.pool import TeamPool, load_teams
from pokesim.state.player import TrackedPlayer

OUT = RAW_DIR / "ladder"
CREDENTIALS = Path.home() / ".config" / "poke-sim" / "showdown.env"


def search_quality_issues(status: dict, elapsed: float, expected_nets: dict | None = None) -> list[str]:
    """Deployment gate for terminal search; failures must not silently become another collection batch."""
    issues = []
    if status.get('fallback'):
        issues.append('search fallback')
    tree = status.get('search', {}).get('terminal_search', {})
    if tree.get('failed', 0):
        issues.append('simulation failed')
    if expected_nets and tree and set(expected_nets) - set(tree.get('policy_nets') or {}):
        issues.append('policy net missing')
    if elapsed > 10:
        issues.append('decision exceeded 10 seconds')
    return issues


def credentials() -> tuple[str, str] | None:
    env = dict(os.environ)
    if CREDENTIALS.exists():
        for line in CREDENTIALS.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    if env.get("PS_USERNAME") and env.get("PS_PASSWORD"):
        return env["PS_USERNAME"], env["PS_PASSWORD"]
    return None


class AdvisorPlayer(TrackedPlayer):
    """With several policies, battles alternate between them (A/B comparison on the same ladder)."""

    def __init__(self, sets: list[dict], policies: list[str], out: Path = OUT, combine: dict | None = None,
                 planner: dict | None = None, sample: bool = False, seed: int | None = None,
                 opponent_policy: str = "default", belief: bool = True, value_run=None, lineup: str = "ladder",
                 preview_coverage: bool = True, control_mode: str = "unknown",
                 **kwargs):
        """lineup: "matchup" ranks lineups by the simulated table shrunk towards recent ladder results;
        "ladder" draws them from the posterior of this roster's own rated results (LadderLineupModel)."""
        self.sets, self.out = sets, out
        self.control_mode = control_mode
        if lineup not in ("matchup", "ladder"):
            raise ValueError(f"unknown lineup method {lineup}")
        self.lineup_method = lineup
        self.selection: dict[str, dict] = {}  # battle tag -> how the lineup was chosen
        self.belief = belief
        self.advisor = Advisor(sets, policy=policies[0], combine=combine, planner=planner,
                               source="showdown", sample=sample, opponent_policy=opponent_policy,
                               belief=belief, value_run=value_run)
        self.planner, self.sample = self.advisor.planner, sample
        self.preview_coverage = PreviewCoverage(sets, self.advisor.engine) \
            if preview_coverage and self.advisor.engine else None
        from pokesim.live.lineup_search import LineupSearch

        self.lineup_search = LineupSearch(sets, self.advisor.engine) if self.preview_coverage else None
        self.seed = seed if seed is not None else random.SystemRandom().randrange(2 ** 31)
        self.rng = random.Random(self.seed)
        roster = [s["species"] for s in sets]
        self.policies = {policies[0]: (self.advisor.model, self.advisor.extra)}
        self.calibrations = {policies[0]: self.advisor.calibration}
        for path in [p for p in policies[1:] if p]:
            model = MaskablePPO.load(path, device="cpu")
            self.policies[path] = (model, TeamFeatures.for_model(model, roster))
            self.calibrations[path] = self.advisor.calibrations.get("showdown", file_hash(path),
                                                                 self.advisor.calibration_context)
        self.assigned: dict[str, str] = {}  # battle tag -> policy path
        self.advice: dict[str, list[dict]] = {}  # battle tag -> per-decision advice
        self.picks, self.matchups = PickModel(), MatchupModel()
        for path in policies:
            if path and (warning := self.matchups.check(path)):
                print(f"lineup table vs {Path(path).name}:", warning, flush=True)
        self.lineups: dict[str, list[str]] = {}
        self.logs: dict[str, list[str]] = {}  # battle tag -> protocol lines
        self.value_logged: set[str] = set()
        self.quality_issues: list[str] = []
        self.sent_requests: dict[str, tuple] = {}
        self.retried_requests: dict[str, tuple] = {}
        self.last_activity = time.monotonic()  # last battle message: the inactivity timeout in run() watches it
        super().__init__(team=FixedTeam(sets), **kwargs)

    async def _handle_battle_message(self, split_messages):
        self.last_activity = time.monotonic()
        tag = split_messages[0][0].lstrip(">").strip() if split_messages and split_messages[0] else ""
        # Append only the prefix already processed. A request inside a packet must not see later events.
        for message in split_messages[1:]:
            battle = self._battles.get(tag)
            kind = message[1] if len(message) > 1 else ''
            if battle is not None and battle.finished:
                if kind in {'request', 'error'}:
                    self.diagnostic(battle, 'finished-message', message=message)
                continue
            if kind == 'error' and len(message) > 2 and any(
                text in message[2] for text in ("There's nothing to choose", 'too late to make a different move')
            ):
                if battle is not None:
                    self.diagnostic(battle, 'stale-choice-error', message=message)
                continue
            if kind == 'request' and len(message) > 2 and message[2] and battle is not None:
                incoming_id = json.loads(message[2]).get('rqid')
                current_id = battle.last_request.get('rqid')
                if isinstance(incoming_id, int) and isinstance(current_id, int) and incoming_id < current_id:
                    self.diagnostic(battle, 'stale-request', incoming_rqid=incoming_id)
                    continue
            if tag and kind and kind not in {'request', 't:', 'j', 'l', 'c'}:
                self.logs.setdefault(tag, []).append('|'.join(message))
            await super()._handle_battle_message([split_messages[0], message])

    @staticmethod
    def request_key(battle) -> tuple:
        return battle.last_request.get('rqid'), battle.turn, bool(battle.force_switch)

    def diagnostic(self, battle, kind: str, **details) -> None:
        """Survives record finalization, including errors and requests received after the battle ends."""
        self.out.mkdir(parents=True, exist_ok=True)
        record = {'time': time.time(), 'battle': battle.battle_tag, 'turn': battle.turn,
                  'rqid': battle.last_request.get('rqid'), 'kind': kind, **details}
        with (self.out / 'diagnostics.jsonl').open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')

    def record_quality(self, battle, entry: dict, issues: list[str]) -> None:
        self.quality_issues.extend(issues)
        if issues:
            self.diagnostic(battle, 'quality-failure', issues=issues, advice=entry)

    async def _handle_battle_request(self, battle, maybe_default_order=False):
        if battle.finished:
            return
        if battle._wait:
            self._waiting.set()
            return
        key = self.request_key(battle)
        tag = battle.battle_tag
        sent_key = self.sent_requests.get(tag)
        if sent_key and key[0] is not None and sent_key[0] is not None and key[0] < sent_key[0]:
            self.diagnostic(battle, 'stale-request', incoming_rqid=key[0])
            return
        if sent_key == key or (sent_key and key[0] is not None and sent_key[0] == key[0]):
            if not maybe_default_order or self.retried_requests.get(tag) == key:
                self.diagnostic(battle, 'duplicate-request', retry=maybe_default_order)
                return
            # A genuinely rejected current choice gets one legal server default. Stale errors are filtered above.
            self.retried_requests[tag] = key
            message = self.choose_default_move().message
            entry = {'turn': battle.turn, 'force': bool(battle.force_switch), 'rqid': key[0],
                     'chosen': 'default', 'order': message, 'retry': True}
            self.advice.setdefault(tag, []).append(entry)
            if self.advisor.engine is not None and self.advisor.engine.value_model is not None:
                self.record_quality(battle, entry, ['choice rejected'])
        elif battle.teampreview:
            message = self.teampreview(battle)
        else:
            order = await self.choose_move(battle)
            message = order.message if order is not None else ''
        if not message or battle.finished or battle._wait or self.request_key(battle) != key:
            self.diagnostic(battle, 'discarded-order', order=message, request_key=key)
            return
        # Bind every choice to the request it was computed for. The server rejects a late command instead of
        # applying it to a different turn (poke-env's default sender omits rqid).
        command = message + (f'|{key[0]}' if key[0] is not None else '')
        self.diagnostic(battle, 'send-attempt', command=command, request_key=key)
        await self.ps_client.send_message(command, tag)
        self.sent_requests[tag] = key
        if self.advice.get(tag):
            self.advice[tag][-1]['sent'] = command
            self.advice[tag][-1]['submission'] = {
                'source': 'ai', 'action': self.advice[tag][-1].get('chosen'), 'command': command, 'rqid': key[0]}
        self.diagnostic(battle, 'sent-order', command=command, request_key=key)

    def policy_for(self, battle) -> str:
        if battle.battle_tag not in self.assigned:
            paths = list(self.policies)  # alternate across restarts too: by the number of records saved so far
            done = len(list(self.out.glob("*.json"))) if self.out.exists() else 0  # one battle at a time
            self.assigned[battle.battle_tag] = paths[done % len(paths)]
        return self.assigned[battle.battle_tag]

    def teampreview(self, battle) -> str:
        self.policy_for(battle)
        if self.advisor.engine is not None and self.advisor.engine.value_model is not None:
            print(f'{battle.battle_tag}: matched; value model '
                  f'{self.advisor.engine.value_model.version["dataset_id"][:12]}', flush=True)
        opp = [m.species for m in battle.teampreview_opponent_team]
        roster = [s["species"] for s in self.sets]
        picks = self.picks.predict(opp, roster=roster, records=OUT)
        if self.advisor.engine is not None:
            self.advisor.engine.set_opponent_picks(battle.battle_tag, picks)
        # Lineup evidence is this roster's rated ladder history (OUT), also when records go elsewhere (--out,
        # local games); unrated local games never count.
        if self.lineup_method == "ladder":
            model = LadderLineupModel.from_records(OUT, roster)
            coverage, coverage_error = None, None
            if self.preview_coverage:
                try:
                    coverage = self.preview_coverage.profile(picks)
                except Exception as e:  # noqa: BLE001 - retain a valid historical selection if static estimates fail
                    coverage_error = repr(e)[:200]
            lineup, info = model.choose(roster, self.rng, self.matchups, picks, coverage=coverage)
            if self.lineup_search:
                try:
                    ranking, continuation = self.lineup_search.rerank(model.rank(roster, picks, coverage), picks)
                    lineup = ranking[0][0]
                    info = {'method': 'public-continuation', 'continuation': continuation, 'static_selector': info,
                            'selection_score': ranking[0][1], 'coverage': ranking[0][2],
                            'score_kind': continuation.get('score_kind', 'static-preview-fallback')}
                except Exception as e:  # noqa: BLE001 - preserve valid preview submission
                    info['continuation_error'] = repr(e)[:200]
            if coverage_error:
                info['coverage_error'] = coverage_error
            self.selection[battle.battle_tag] = info
        else:
            self.matchups.observe_ladder(OUT, roster)
            ranking = self.matchups.rank(roster, picks)
            lineup = ranking[0][0] if ranking else [s["species"] for s in self.sets[:3]]
            self.selection[battle.battle_tag] = {"method": "matchup", "win_prob": ranking[0][1] if ranking else None}
        self.selection[battle.battle_tag]['opponent_selection'] = picks.context
        self.selection[battle.battle_tag]['opponent_lineups'] = picks.lineups[:5]
        self.lineups[battle.battle_tag] = lineup
        order = [next(i for i, s in enumerate(self.sets) if to_id(s["species"]) == to_id(x)) for x in lineup]
        order += [i for i in range(len(self.sets)) if i not in order]
        return "/team " + "".join(str(i + 1) for i in order)

    def last_opponent_move(self, battle) -> str | None:
        role = battle.player_role
        for line in reversed(self.logs.get(battle.battle_tag, [])):
            ev = line.split("|")
            if len(ev) > 3 and ev[1] in ("switch", "drag") and not ev[2].startswith(role):
                return None
            if len(ev) > 3 and ev[1] == "move" and not ev[2].startswith(role):
                return "move:" + to_id(ev[3])
        return None

    def activate_policy(self, policy_path: str) -> None:
        """Switch both the model and its calibration together, including clearing an unsupported calibration."""
        self.advisor.model, self.advisor.extra = self.policies[policy_path]
        self.advisor.calibration = self.calibrations[policy_path]

    async def choose_move(self, battle):
        if battle.finished or battle._wait:
            self.diagnostic(battle, 'discarded-request')
            return None
        key = self.request_key(battle)
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
        self.activate_policy(self.policy_for(battle))
        active = battle.active_pokemon
        entry = {"turn": battle.turn, "request": battle.last_request,
                 "force": bool(battle.force_switch), 'rqid': key[0],
                 'active': None if active is None else {'species': active.species, 'hp': active.current_hp,
                     'boosts': dict(active.boosts), 'effects': [e.name for e in active.effects]}}
        self.advice.setdefault(battle.battle_tag, []).append(entry)
        try:
            advice = await asyncio.to_thread(self.advisor.advise, battle, self.last_opponent_move(battle),
                                              list(self.logs.get(battle.battle_tag, [])), battle.player_role)
            if battle.finished or battle._wait or self.request_key(battle) != key:
                self.diagnostic(battle, 'discarded-result', request_key=key, elapsed=advice.elapsed)
                return None
            value = advice.status.get('search', {}).get('value', {})
            tree = advice.status.get('search', {}).get('terminal_search', {})
            if tree and battle.battle_tag not in self.value_logged:
                self.value_logged.add(battle.battle_tag)
                print(f'{battle.battle_tag}: terminal search active ({tree["terminal"]} ends, '
                      f'{tree["censored"]} capped, {tree["unfinished_at_deadline"]} unfinished; '
                      f'{advice.elapsed:.2f}s)', flush=True)
            elif value.get('learned', 0) and battle.battle_tag not in self.value_logged:
                self.value_logged.add(battle.battle_tag)
                print(f'{battle.battle_tag}: value active ({value["learned"]} leaves, '
                      f'{advice.elapsed:.2f}s)', flush=True)
            entry.update({"actions": rounded(advice.actions[:5]), "win": advice.win_raw, "win_cal": advice.win_prob,
                          "policy": rounded(sorted(advice.policy.items(), key=lambda kv: -kv[1])[:3]),
                          "lookahead": [(lab, round(p, 3), round(v, 3)) for lab, p, v in advice.lookahead[:3]],
                          "opponent": rounded(advice.opponent[:4]), "secs": round(advice.elapsed, 2),
                          "status": advice.status})
            entry['evaluations'] = advice.evaluations
            entry['win_kind'] = advice.win_kind
            if self.advisor.engine is not None and self.advisor.engine.value_model is not None:
                nets = ((getattr(self, 'planner', None) or {}).get('search') or {}).get('policyNets')
                self.record_quality(battle, entry, search_quality_issues(advice.status, advice.elapsed, nets))
            if advice.shadow:
                entry["shadow"] = [(lab, round(p, 3), round(v, 3)) for lab, p, v in advice.shadow[:5]]
            legal = [(label, p, a) for label, p in advice.actions
                     if (a := label_to_action(label, battle)) is not None and mask[a]]
            if legal:
                label, p, action = legal[0]
                if self.sample and sum(q for _, q, _ in legal) > 0:
                    label, p, action = self.rng.choices(legal, weights=[q for _, q, _ in legal])[0]
                entry.update({"chosen": label, "suggested": label, "chosen_p": round(float(p), 4)})
                order = SinglesEnv.action_to_order(np.int64(action), battle, strict=True)
                entry['order'] = order.message
                return order
        except Exception as e:  # noqa: BLE001 - never stall a live battle
            if battle.finished or battle._wait or self.request_key(battle) != key:
                self.diagnostic(battle, 'discarded-error', request_key=key, error=repr(e))
                return None
            print("advice failed:", repr(e)[:200], flush=True)
            entry["error"] = repr(e)[:200]
        if self.advisor.engine is not None and self.advisor.engine.value_model is not None:
            self.record_quality(battle, entry, ['advice failed or had no legal action'])
        entry["chosen"] = "random"
        order = self.choose_random_move(battle)
        entry['order'] = order.message
        return order

    def _battle_finished_callback(self, battle) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        log = self.logs.pop(battle.battle_tag, [])
        if battle.player_role == "p2":  # records always have us as p1 (BattleSession, observed, eval_records)
            log = [re.sub(r"\bp([12])(?=[a-z]?[:|]|$)", lambda m: "p" + ("2" if m.group(1) == "1" else "1"), line)
                   for line in log]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        record = {"control_mode": self.control_mode, "started": stamp,
                  "result": "win" if battle.won else "loss" if battle.lost else None,
                  "team": [s["species"] for s in self.sets],
                  "preview": {"species": [m.species for m in battle.teampreview_opponent_team]},
                  "lineup": self.lineups.get(battle.battle_tag), "log": log,
                  "advice": self.advice.pop(battle.battle_tag, []),
                  "policy": Path(self.assigned.get(battle.battle_tag) or "").stem or "none",
                  "source": f"showdown {battle.battle_tag}", "opponent": battle.opponent_username,
                  "rating": getattr(battle, "rating", None),
                  "versions": describe(self.assigned.get(battle.battle_tag), self.matchups.meta,
                                       planner=self.planner, sample=self.sample, seed=self.seed,
                                       opponent_policy=self.advisor.opponent_policy_version, belief=self.belief,
                                       calibration_context=self.advisor.calibration_context,
                                       selection=self.selection.pop(battle.battle_tag, None))}
        from pokesim.live.provenance import actual_lineup, execution_rows, request_lineup

        names = request_lineup(battle.last_request)
        record['suggested_lineup'] = record['lineup']
        record['actual_lineup'] = ({'species': names, 'complete': True, 'evidence': 'private_request',
                                    'source': 'ai' if self.control_mode == 'autonomous_ai' else 'unknown'}
                                   if names else actual_lineup(record))
        record['executions'] = execution_rows(record)
        (self.out / f"{stamp}-{battle.battle_tag.split('-')[-1]}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1))
        print(f"{battle.battle_tag}: {record['result']} vs {battle.opponent_username} [{record['policy']}]  "
              f"({self.n_won_battles}-{self.n_lost_battles})", flush=True)


def code_stamp(extra: list[Path] = ()) -> float:
    """Latest modification time of the code, the sim scripts, the priors the advisor reads and the model files
    in use (extra: policies, the value run, the opponent policy, the team), so a new model is picked up between
    battles. Not whole checkpoint or run directories: training writes there while the collector plays."""
    paths = [*(ROOT / "src").rglob("*.py"), *(ROOT / "services" / "sim-tools").glob("*.js"), Path(__file__),
             *(ROOT / "data" / "processed").glob("*_priors*.json"), *map(Path, extra)]
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


def model_files(args) -> list[Path]:
    files = [Path(p) for p in args.policy if p] + [Path(args.team), PROCESSED_DIR / "matchup_model.json",
                                             PROCESSED_DIR / "pick_rates.json"]
    if args.value_run:
        run = Path(args.value_run)
        files += [run / "report.json", *run.glob("*.pt")]
    if args.opponent_policy and args.opponent_policy != "default":
        files.append(Path(args.opponent_policy.removeprefix("net:")))
    else:
        from pokesim.live.opponent import DEFAULT_POLICY

        files.append(DEFAULT_POLICY)
    return files


def rounded(pairs) -> list:
    return [(k, round(float(v), 3)) for k, v in pairs]


async def run(args) -> None:
    sets = json.loads(Path(args.team).read_text())
    # --policy none: no PPO policy at all; the planner's search decides and a failed search plays a random
    # legal move (recorded as such). Collection can start before any policy is trained.
    args.policy = [None if p in ("none", "") else p for p in args.policy]
    combine = json.loads(args.combine) if args.combine else None
    planner = None if args.planner == "off" else {"mode": args.planner}
    if planner and args.search:
        planner["search"] = json.loads(args.search)
    if planner and args.rho_setup is not None:
        planner["rho_by_class"] = {"setup": args.rho_setup}
    if planner and args.near is not None:
        planner["near"] = args.near
    play = {"combine": combine, "planner": planner, "sample": args.sample, "seed": args.seed,
            "opponent_policy": args.opponent_policy, "belief": args.belief, "value_run": args.value_run,
            "lineup": args.lineup, "preview_coverage": args.preview_coverage, "control_mode": args.control_mode}
    if args.local:
        tag = uuid.uuid4().hex[:6]
        me = AdvisorPlayer(sets, args.policy, out=Path(args.out), **play, battle_format=FORMAT,
                           server_configuration=LocalhostServerConfiguration, max_concurrent_battles=1,
                           account_configuration=AccountConfiguration(f"adv_{tag}", None))
        # the same seed: the same opponent teams; the published pool until our own battles produced observed teams
        pool = TeamPool(load_teams("observed") or load_teams("all"), args.seed or 0)
        if args.local_opponent == "heuristic":
            foe = CounterPickHeuristic(battle_format=FORMAT, team=pool, seed=0, p_counter=0.7,
                                       server_configuration=LocalhostServerConfiguration,
                                       account_configuration=AccountConfiguration(f"opp_{tag}", None))
        elif args.local_opponent:
            foe = LeaguePlayer(battle_format=FORMAT, team=pool, snapshots=[args.local_opponent], p_heuristic=0.0,
                               server_configuration=LocalhostServerConfiguration,
                               account_configuration=AccountConfiguration(f"opp_{tag}", None))
            rng = random.Random(0)
            foe.teampreview = lambda battle: counter_preview(battle, rng, 0.7)
        else:
            foe = RandomPlayer(battle_format=FORMAT, team=pool, server_configuration=LocalhostServerConfiguration,
                               account_configuration=AccountConfiguration(f"opp_{tag}", None))
        await me.battle_against(foe, n_battles=args.battles)
        print(f"done: {me.n_won_battles}-{me.n_lost_battles} combine={combine}", flush=True)
        return
    cred = credentials()
    if cred is None:
        raise SystemExit(f"no Showdown account: put PS_USERNAME / PS_PASSWORD in {CREDENTIALS}")
    me = AdvisorPlayer(sets, args.policy, out=Path(args.out), **play, battle_format=FORMAT,
                       account_configuration=AccountConfiguration(*cred),
                       server_configuration=ShowdownServerConfiguration, max_concurrent_battles=1,
                       start_timer_on_battle_start=True)
    watched = model_files(args)
    stamp = code_stamp(watched)
    for _ in range(args.battles):
        # A long battle is fine (60 decisions of 8 s take over 10 minutes); a silent connection is not.
        task = asyncio.ensure_future(me.ladder(1))
        me.last_activity = time.monotonic()
        while not task.done():
            await asyncio.wait({task}, timeout=15)
            if not task.done() and time.monotonic() - me.last_activity > args.battle_timeout:
                task.cancel()
                raise TimeoutError(f'no battle message for {args.battle_timeout:.0f} seconds')
        task.result()
        if me.quality_issues:
            print('terminal search quality gate: stop after this battle: ' +
                  ', '.join(sorted(set(me.quality_issues))), flush=True)
            raise SystemExit(78)
        if code_stamp(watched) != stamp:  # code, priors or models changed: exit so the caller restarts us
            print("code changed: exiting to reload", flush=True)
            break
    print(f"done: {me.n_won_battles}-{me.n_lost_battles}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=30)
    ap.add_argument("--team", default=str(ROOT / "data/my_teams/baton.json"))
    ap.add_argument("--opponent-policy", default="default",
                    help="cloned human policy for the opponent prediction (default: checkpoints/bc_human.zip), or "
                         "net:<exported policy json> for a learned public policy inside the search")
    ap.add_argument("--belief", action=argparse.BooleanOptionalAction, default=True,
                    help="infer opponent sets from damage and move order (default on; --no-belief disables)")
    ap.add_argument('--value-run', type=Path, help='validated terminal-value run; requires planner on/shadow')
    ap.add_argument('--battle-timeout', type=float, default=600,
                    help='restart when no battle message arrives for this long (seconds); a long battle itself '
                         'is not interrupted')
    ap.add_argument("--policy", nargs="+", default=[str(ROOT / "checkpoints/ppo_baton_v2.zip")],
                    help="one or more policies; with several, battles alternate between them")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--local", action="store_true", help="local server (localhost:8000) against a random player")
    ap.add_argument("--local-opponent", help="with --local: a saved opponent policy instead (counter-picks 70%%), "
                                              "or \"heuristic\" for the counter-picking heuristic bot")
    ap.add_argument("--combine", help='JSON overriding the advisor\'s COMBINE, e.g. \'{"mode": "linear"}\'')
    ap.add_argument("--planner", choices=["off", "shadow", "on"], default="off",
                    help="common-strategy planner: shadow records it next to the current advice, on plays it")
    ap.add_argument("--search", help='JSON overriding the planner\'s worker settings (PLANNER_SEARCH), e.g. '
                                      '\'{"continuations": ["expert/tactical", "expert/disrupt"]}\'')
    ap.add_argument("--lineup", choices=["matchup", "ladder"], default="ladder",
                    help="team preview: the simulated matchup table shrunk to ladder results, or the posterior of "
                         "this roster's own ladder results with Thompson sampling (pokesim.live.selection)")
    ap.add_argument('--control-mode', choices=['autonomous_ai', 'assisted', 'unknown'], default='unknown',
                    help='declare autonomous operation only when no manual choices can override the bot')
    ap.add_argument('--preview-coverage', action=argparse.BooleanOptionalAction, default=True,
                    help='rank whole-trio matchup coverage at preview (disable for an offline A/B)')
    ap.add_argument("--rho-setup", type=float, default=None,
                    help="worst-case share for set-up moves in the planner's ranking (default: the planner's rho)")
    ap.add_argument("--near", type=float, default=None,
                    help="with --sample: draw among actions within this objective margin of the best (softmax)")
    ap.add_argument("--sample", action="store_true", help="draw the action from the advice instead of its top")
    ap.add_argument("--seed", type=int, help="seed for --sample (default: random, saved in each record)")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
