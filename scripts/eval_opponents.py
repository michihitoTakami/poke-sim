"""Score opponent AIs by how well they predict what human players actually did.

Decisions come from pokesim.rl.human, seen from the deciding player's side (own team = what was revealed plus the
set data, the other side = only what had been revealed):
  records   the opponents in our own battles (data/processed/replayed, the Switch ladder)
  replays   held-out Showdown replays (the test split of pokesim.rl.human.split)
Candidates: uniform over legal actions, "repeat the previous move", poke-env's heuristic bot, saved policies
(--policy, e.g. the general league policy or the cloned human policy) and, with --lookahead, the search
(its equilibrium mix from the deciding side, with the other side's sets sampled from the set data).

  uv run python scripts/eval_opponents.py --policy checkpoints/ppo_m6_league3m.zip checkpoints/bc_human.zip
"""
import argparse
import math
from collections import defaultdict

import numpy as np
from poke_env.environment import SinglesEnv
from poke_env.player import SimpleHeuristicsPlayer

from pokesim.config import FORMAT
from pokesim.rl.human import Sides, decisions, record_logs, replay_logs, split
from pokesim.rl.myteam import to_id

REPEAT = 0.56  # how often our opponents repeated their previous move (scripts/eval_records.py)
N_ACTIONS = SinglesEnv.get_action_space_size(9)


class Probe:
    """Other players' choices at each human decision: the heuristic bot and optionally the search."""

    def __init__(self, lookahead: bool):
        self.bot = SimpleHeuristicsPlayer(battle_format=FORMAT, start_listening=False)
        self.engine = None
        if lookahead:
            from pokesim.search.lookahead import Lookahead, SetSampler

            self.engine = Lookahead(SetSampler(["M-6"], observed={}), samples=4, seeds=1, rollout=3)

    def __call__(self, battle, sets):
        out = {}
        try:
            out["heuristic"] = int(SinglesEnv.order_to_action(self.bot.choose_move(battle), battle, strict=False))
        except Exception:  # noqa: BLE001
            pass
        if self.engine and not battle.force_switch and battle.active_pokemon and battle.opponent_active_pokemon:
            from pokesim.search.lookahead import label_to_action

            try:
                own = {self.engine.sampler.base_id(to_id(s["species"])): s for s in sets}
                adv = self.engine.advise(battle, own)
                dist = np.zeros(N_ACTIONS)
                for label, p in zip(adv.labels, adv.prob):
                    a = label_to_action(label, battle)
                    if a is not None:
                        dist[a] += p
                out["lookahead"] = dist
            except Exception as e:  # noqa: BLE001
                print("lookahead failed:", repr(e)[:120], flush=True)
        return out


def collect(items, roles, sides, probe):
    rows = []
    for stem, log in items:
        for role in roles:
            try:
                rows += decisions(log, role, sides, probe=probe)[0]
            except Exception as e:  # noqa: BLE001
                print("skip", stem, role, repr(e)[:100])
    return rows


def policy_dists(path, rows):
    import torch
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(path, device="cpu")
    obs = torch.tensor(np.array([d.obs for d in rows]))
    with torch.no_grad():
        return model.policy.get_distribution(obs, action_masks=np.array([d.mask for d in rows])
                                             ).distribution.probs.numpy()


def uniform(d):
    return d.mask / d.mask.sum()


def peaked(d, a, share):
    """`share` on action a (if legal), the rest uniform."""
    u = uniform(d)
    if a is None or not d.mask[a]:
        return u
    return (1 - share) * u + share * np.eye(len(u))[a]


def score(name, rows, dists) -> None:
    stats = defaultdict(lambda: [0, 0, 0, []])
    for d, p in zip(rows, dists):
        p = np.where(d.mask, p, 0)
        p = p / p.sum() if p.sum() > 0 else uniform(d)
        ranked = np.argsort(-p)
        for key in ("all", d.kind):
            s = stats[key]
            s[0] += 1
            s[1] += ranked[0] == d.action
            s[2] += d.action in ranked[:3]
            s[3].append(math.log(max(p[d.action], 1e-3)))
    parts = []
    for key in ("all", "turn", "force"):
        n, h1, h3, ll = stats[key]
        if n:
            parts.append(f"{key} n={n} top1 {h1 / n:.0%} top3 {h3 / n:.0%} ll {np.mean(ll):.2f}")
    print(f"  {name:26s} " + " | ".join(parts), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", nargs="*", default=[])
    ap.add_argument("--lookahead", action="store_true", help="also score the search on our records (slow)")
    ap.add_argument("--replays", type=int, default=0, help="limit the number of held-out replays")
    args = ap.parse_args()

    sides = Sides(seed=1)
    replays = replay_logs()
    _, test = split(np.array([f"{stem}:p1" for stem, _ in replays]))
    test_items = [replays[i] for i in test][:args.replays or None]
    with_search, plain = Probe(args.lookahead), Probe(False)
    for title, items, roles, probe in (("records (our opponents, Switch)", record_logs(), ["p2"], with_search),
                                       ("held-out Showdown replays", test_items, ["p1", "p2"], plain)):
        rows = collect(items, roles, sides, probe)
        print(f"{title}: {len(rows)} decisions")
        score("uniform", rows, [uniform(d) for d in rows])
        score("repeat previous move", rows, [peaked(d, d.last_action, REPEAT) for d in rows])
        score("heuristic bot", rows, [peaked(d, d.extra.get("heuristic"), 0.8) for d in rows])
        if any("lookahead" in d.extra for d in rows):
            score("lookahead (equilibrium)", rows, [d.extra.get("lookahead", uniform(d)) for d in rows])
        for path in args.policy:
            score(path.split("/")[-1], rows, list(policy_dists(path, rows)))
    if with_search.engine:
        with_search.engine.close()


if __name__ == "__main__":
    main()
