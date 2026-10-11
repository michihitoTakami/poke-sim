"""Play a scripted Baton Pass plan with my team against the team pool and count what disrupted it.

The plans are in pokesim.rl.baton.
Opponents are the lookahead player (it will phaze/Encore when that scores better) or the heuristic bot.

Usage: uv run python scripts/eval_my_team.py --lineup Scolipede Gholdengo Salamence --battles 60 [--opponent lookahead]
"""
import argparse
import asyncio
import random
import uuid
from collections import Counter

import numpy as np
from poke_env import AccountConfiguration
from poke_env.environment import SinglesEnv
from poke_env.player import SimpleHeuristicsPlayer
from sb3_contrib import MaskablePPO

from pokesim.config import FORMAT, ROOT
from pokesim.rl.baton import BatonScript, compress, disruptions, my_actions, stint
from pokesim.rl.features import embed
from pokesim.rl.myteam import FixedTeam, TeamFeatures, counter_preview, load_my_team
from pokesim.search.player import LookaheadPlayer
from pokesim.sim.pool import TeamPool, load_teams


class PolicyPilot(BatonScript):
    """Plays a trained fixed-team policy (pokesim.rl.train --my-team) instead of the script; keeps the logging."""

    def __init__(self, *args, policy: str, roster: list[str], deterministic: bool = True, **kwargs):
        self.model = MaskablePPO.load(policy, device="cpu")
        self.extra, self.deterministic = TeamFeatures(roster), deterministic
        super().__init__(*args, **kwargs)

    def choose_move(self, battle):
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
        if not mask.any():
            return self.choose_random_move(battle)
        obs = np.concatenate([embed(battle), self.extra(battle)])
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=self.deterministic)
        return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)


def trace(me, lineup: list[str]) -> None:
    """Which moves each lineup member used and in what order, with win rates per opening."""
    names = [n.split("-")[0] for n in lineup]
    openings: dict[str, list[bool]] = {}
    later: dict[str, Counter] = {n: Counter() for n in names}
    after_pass, usage = Counter(), {n: Counter() for n in names}
    for battle in me.battles.values():
        acts = my_actions(me.logs.get(battle.battle_tag, []), battle.player_role)
        openings.setdefault(compress(stint(acts, names[0])), []).append(battle.won)
        for n in names[1:]:
            first = stint(acts, n)
            if first:
                later[n][compress(first[:5])] += 1
        for i, (_, name, act) in enumerate(acts):
            if name in usage and act not in ("switch", "drag", "faint"):
                usage[name][act] += 1
            if act == "Baton Pass":
                nxt = next((nm for _, nm, a in acts[i + 1:] if a == "switch"), None)
                after_pass[nxt] += 1
    print(f"  {names[0]}'s first stay (games, win rate):")
    for seq, res in sorted(openings.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"    {len(res):3d}  {sum(res) / len(res):4.0%}  {seq or '(no move)'}")
    print("  switched in after Baton Pass:", dict(after_pass.most_common()))
    for n in names:
        print(f"  {n} moves:", dict(usage[n].most_common()))
    for n in names[1:]:
        print(f"  {n}'s first stay (top):", "; ".join(f"{k}x {s}" for s, k in later[n].most_common(5)))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default=str(ROOT / "data/my_teams/baton.json"))
    parser.add_argument("--lineup", nargs=3, default=["Scolipede", "Gholdengo", "Salamence"])
    parser.add_argument("--battles", type=int, default=60)
    parser.add_argument("--opponent", default="lookahead", choices=["lookahead", "heuristic"])
    parser.add_argument("--pool", default="all")
    parser.add_argument("--plan", default="pass6", choices=["pass6", "sub"],
                        help="pass6: Iron Defense to +6 then pass; sub: Iron Defense once + Substitute, pass both, "
                             "recipient sets up to +6 behind Substitute")
    parser.add_argument("--pass-to", help="preferred Baton Pass recipient (species name)")
    parser.add_argument("--policy", help="play this trained policy instead of the script")
    parser.add_argument("--trace", action="store_true", help="summarise which moves we used, in what order")
    parser.add_argument("--counter-pick", type=float, default=0.0,
                        help="chance the opponent picks its 3 by counter_score after seeing our team (else random)")
    args = parser.parse_args()
    order = load_my_team(args.team, args.lineup)
    teams = load_teams(args.pool)
    tag = uuid.uuid4().hex[:6]
    common = {"battle_format": FORMAT, "team": FixedTeam(order), "plan": args.plan, "pass_to": args.pass_to,
              "account_configuration": AccountConfiguration(f"baton_{tag}", None)}
    if args.policy:
        me = PolicyPilot(policy=args.policy, roster=[s["species"] for s in order], **common)
    else:
        me = BatonScript(**common)
    if args.opponent == "lookahead":
        opp = LookaheadPlayer(battle_format=FORMAT, team=TeamPool(teams, 7), teams=teams, seasons=["M-6"],
                              account_configuration=AccountConfiguration(f"opp_{tag}", None))
    else:
        opp = SimpleHeuristicsPlayer(battle_format=FORMAT, team=TeamPool(teams, 7),
                                     account_configuration=AccountConfiguration(f"opp_{tag}", None))
    if args.counter_pick:
        rng = random.Random(0)
        opp.teampreview = lambda battle: counter_preview(battle, rng, args.counter_pick)
    await me.battle_against(opp, n_battles=args.battles)

    by_case: dict[str, list[bool]] = {}
    total = Counter()
    for battle in me.battles.values():
        role = battle.player_role
        d = disruptions(me.logs.get(battle.battle_tag, []), role)
        total.update(d)
        lost_boosts = d["dragged out"] or d["haze"] or d["encoreed"] or d["taunted"]
        by_case.setdefault("disrupted" if lost_boosts else "clean", []).append(battle.won)
    n = len(me.battles)
    won = me.n_won_battles
    who = f"policy {args.policy}" if args.policy else args.plan
    print(f"[{who}] {' / '.join(args.lineup)} vs {args.opponent} (counter-pick {args.counter_pick:.0%}): "
          f"{won}/{n} ({won / n:.0%})")
    for case, results in by_case.items():
        print(f"  {case:9s}: {sum(results)}/{len(results)} won ({sum(results) / len(results):.0%})")
    print("  events:", dict(total.most_common()))
    if args.trace:
        trace(me, args.lineup)
    if args.opponent == "lookahead":
        opp.engine.close()


if __name__ == "__main__":
    asyncio.run(main())
