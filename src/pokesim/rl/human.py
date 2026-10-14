"""Human decisions from battle logs, seen from one player's side: data for imitating (and scoring) opponents.

A Showdown replay (data/raw/replays) or one of our own records (data/processed/replayed, us = p1) is replayed
through `BattleSession` from the chosen side. That side's own team is what the battle revealed (moves used,
item, ability) filled in from the set data; the other side is only what a player would see at that point
(the 6 at team preview, then whatever was revealed). At every decision the player actually made we keep:
  turn decisions      the first move / switch of that side after `|turn|`
  forced switches     any later switch of that side in the same turn (after a faint, U-turn, Baton Pass, ...)
with poke-env's observation (pokesim.rl.features.embed), action mask and the action taken (SinglesEnv index).
The mask leaves out moves the log never showed (see honest_mask).

  uv run python -m pokesim.rl.human build      # -> data/processed/human_replays.npz + human_records.npz
  uv run python -m pokesim.rl.human fit --init checkpoints/ppo_m6_league3m.zip --out checkpoints/bc_human
  # fine-tune on the people we faced (Switch + Showdown ladder), held out by battle:
  uv run python -m pokesim.rl.human build --records-only
  uv run python -m pokesim.rl.human fit --init checkpoints/bc_human.zip --records data/processed/human_records.npz \
      --out checkpoints/bc_ours --epochs 10 --lr 1e-4
"""
import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from poke_env.environment import SinglesEnv
from poke_env.player import Player
from sqlalchemy import select

from pokesim.config import PROCESSED_DIR, RAW_DIR
from pokesim.data.parse_replays import parse_log
from pokesim.db.models import Species
from pokesim.db.session import Session
from pokesim.live.session import BattleSession
from pokesim.rl.features import embed
from pokesim.rl.myteam import to_id
from pokesim.search.lookahead import SetSampler

REPLAYS = RAW_DIR / "replays"
RECORDS = PROCESSED_DIR / "replayed"
# our battles against people, us = p1: Switch battles rebuilt from saved frames, Showdown ladder (scripts/ladder.py),
# and Switch battles only read live (data/raw/live/logs) when no rebuilt record exists for them
RECORD_SOURCES = [RECORDS, RAW_DIR / "ladder"]
LIVE_LOGS = RAW_DIR / "live" / "logs"
SKIP = {"player", "poke", "clearpoke", "teampreview", "teamsize", "gen", "tier", "rule", "gametype", "badge",
        "uhtml", "j", "l", "c", "raw", "inactive", "inactiveoff", "t:", "rated", "start", "win", "tie", "html",
        "title", "timestamp", "seed", "n", "chat", "debug", "message", "bigerror", "error"}


@dataclass
class Decision:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    label: str  # "move:<id>[:mega]" / "switch:<species id>"
    kind: str  # "turn" / "force"
    turn: int
    last_move: str | None  # the side's active's previous move, for the repeat baseline
    last_action: int | None = None  # that previous move as an action index, if still legal
    extra: object = None  # probe(battle, sets) at decision time, when a probe is given


class Sides:
    """Fills the unrevealed parts of each side's sets from the set data (one sampler shared by all battles)."""

    def __init__(self, seed: int = 0):
        self.sampler = SetSampler(["M-6"], seed=seed, observed={})
        with Session() as s:
            self.species = {sp.id: sp for sp in s.scalars(select(Species))}

    def revealed(self, log: str, role: str) -> dict[str, set[str]]:
        """Moves each of the side's Pokémon used in the battle (the only moves we know it had)."""
        return {sid: set(m.moves) for sid, m in parse_log(log, self.species)[role].mons.items()}

    def sets(self, log: str, role: str) -> list[dict]:
        side = parse_log(log, self.species)[role]
        out = []
        for sid in side.preview:
            mon = side.mons.get(sid)
            if sid not in self.sampler.species:
                raise KeyError(sid)
            out.append(self.sampler.sample(sid, mon.moves if mon else [], mon.item if mon else None,
                                           mon.ability if mon else None))
        return out


def canonical(lines: list[str], role: str, names: dict[str, str], species: dict) -> list[str]:
    """Rename the side's nicknames to its set names (BattleSession looks Pokémon up by them)."""
    nick: dict[str, str] = {}
    out = []
    for line in lines:
        parts = line.split("|")
        if len(parts) > 3 and parts[1] in ("switch", "drag", "replace") and parts[2].startswith(role):
            sid = to_id(parts[3].split(",")[0])
            sp = species.get(sid)
            if sp is not None and sp.battle_only:
                sid = to_id(sp.battle_only)
            elif sp is not None and sp.is_mega and sp.changes_from:
                sid = to_id(sp.changes_from)
            if sid in names:
                nick[parts[2].split(": ", 1)[1]] = names[sid]
        if nick:
            line = re.sub(rf"({role}[a-z]?): ([^|]+)", lambda m: f"{m.group(1)}: {nick.get(m.group(2), m.group(2))}",
                          line)
        out.append(line)
    return out


def to_action(battle, label: str) -> int | None:
    kind, _, rest = label.partition(":")
    if kind == "switch":
        mon = next((m for m in battle.available_switches if to_id(m.species).startswith(rest)
                    or rest.startswith(to_id(m.species))), None)
        order = Player.create_order(mon) if mon else None
    else:
        mid, _, mega = rest.partition(":")
        move = next((m for m in battle.available_moves if m.id == mid), None)
        order = Player.create_order(move, mega=bool(mega) and battle.can_mega_evolve) if move else None
    if order is None:
        return None
    try:
        return int(SinglesEnv.order_to_action(order, battle, strict=False))
    except (ValueError, AssertionError, IndexError):
        return None


def honest_mask(battle, revealed: dict[str, set[str]]) -> np.ndarray:
    """The legal actions, without moves the log never showed: those were filled in from the set data and the
    player never picks them in that battle, which would teach (and score) "avoid the filled-in moves"."""
    mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
    mon = battle.active_pokemon
    if mon is not None:
        known = set().union(*(v for k, v in revealed.items() if to_id(mon.species).startswith(k)))
        for i, mid in enumerate(list(mon.moves)[:4]):
            if mid not in known:
                mask[6 + i] = mask[10 + i] = False
    return mask


def label_of(line: str, megas: set[str]) -> str:
    parts = line.split("|")
    if parts[1] == "move":
        return "move:" + to_id(parts[3]) + (":mega" if parts[2][:3] in megas else "")
    return "switch:" + to_id(parts[3].split(",")[0])


def base_species_id(details: str, species: dict) -> str:
    sid = to_id(details.split(",")[0])
    sp = species.get(sid)
    if sp is not None and sp.battle_only:
        return to_id(sp.battle_only)
    if sp is not None and sp.is_mega and sp.changes_from:
        return to_id(sp.changes_from)
    return sid


def open_session(lines: list[str], role: str, sets: list[dict], species: dict) -> tuple[BattleSession, list[str]]:
    """A session for `role` playing `sets` (team preview order) at team preview, and the battle lines to feed it
    (nicknames of that side renamed). The lineup is the side's Pokémon in order of appearance."""
    opp_role = "p2" if role == "p1" else "p1"
    session = BattleSession(sets, role=role)
    set_names = {to_id(s["species"]): i["name"] for s, i in zip(sets, session.info)}
    opp_preview = [ln.split("|")[3] for ln in lines if ln.startswith(f"|poke|{opp_role}|")]
    session.preview([d.split(",")[0] for d in opp_preview], opp_preview)
    lineup = []
    for ln in lines:
        if ln.startswith((f"|switch|{role}", f"|drag|{role}")):
            sid = base_species_id(ln.split("|")[3], species)
            if sid not in lineup:
                lineup.append(sid)
    ids = [to_id(s["species"]) for s in sets]
    lineup = [s for s in lineup if s in ids] + [s for s in ids if s not in lineup]
    session.select([ids.index(s) for s in lineup[:3]])
    body = [ln for ln in lines if len(ln.split("|")) > 1 and ln.split("|")[1] not in SKIP]
    return session, canonical(body, role, set_names, species)


def decisions(log: str, role: str, sides: Sides, probe=None) -> tuple[list[Decision], bool | None]:
    """All decisions `role` made in a battle log (full Showdown log text), and whether that side won.
    probe(battle, own_sets): called at each kept decision (e.g. another player's choice) -> Decision.extra."""
    species = sides.species
    lines = [ln for ln in log.split("\n") if ln.startswith("|")]
    winner_name = next((ln.split("|")[2] for ln in lines if ln.startswith("|win|")), None)
    names = {ln.split("|")[2]: ln.split("|")[3] for ln in lines if ln.startswith("|player|") and
             len(ln.split("|")) > 3 and ln.split("|")[3]}
    won = None if winner_name is None else winner_name == names.get(role, role)
    sets = sides.sets(log, role)
    revealed = sides.revealed(log, role)
    session, body = open_session(lines, role, sets, species)
    out: list[Decision] = []
    in_turn, acted, megas = False, False, set()
    turn = 0
    for line in body:
        parts = line.split("|")
        kind = parts[1]
        if kind == "turn":
            session.apply(line)
            in_turn, acted, megas = True, False, set()
            turn = int(parts[2])
            continue
        mine = len(parts) > 2 and parts[2].startswith(role)
        if in_turn and mine and kind == "-mega":
            megas.add(parts[2][:3])
        if in_turn and mine and kind in ("move", "switch") and "[from]" not in line and not line.endswith("[still]"):
            force = acted
            if kind == "move" and force:
                pass  # a second move in one turn (Dancer, Instruct...): not a choice
            else:
                label = label_of(line, megas)
                try:
                    battle = session.decision(force_switch=force)
                    action = to_action(battle, label) if battle.active_pokemon or force else None
                except Exception:  # noqa: BLE001 - a log our session cannot follow: skip the decision
                    action = None
                if action is not None:
                    mask = honest_mask(battle, revealed)
                    if mask[action]:
                        last = session.last_move(role) if not force else None
                        last_action = to_action(battle, last) if last else None
                        out.append(Decision(embed(battle), mask, action, label, "force" if force else "turn", turn,
                                            last, last_action, probe(battle, sets) if probe else None))
            if kind == "move":
                acted = True
            elif not force:
                acted = True
        if kind == "cant" and mine:
            acted = True
        try:
            session.apply(line)
        except Exception:  # noqa: BLE001
            return out, won
    return out, won


def replay_logs() -> list[tuple[str, str]]:
    out = []
    for p in sorted(REPLAYS.glob("*.json")):
        d = json.loads(p.read_text())
        out.append((p.stem, d["log"]))
    return out


def record_paths() -> list[Path]:
    """Record files from RECORD_SOURCES, plus live-only logs more than 5 minutes from any rebuilt record."""
    from datetime import datetime

    def stamp(p: Path) -> datetime | None:
        try:
            return datetime.strptime(p.stem[:15], "%Y%m%d-%H%M%S")
        except ValueError:
            return None

    paths = [p for d in RECORD_SOURCES for p in sorted(d.glob("*.json"))]
    rebuilt = [t for p in sorted(RECORDS.glob("*.json")) if (t := stamp(p))]
    for p in sorted(LIVE_LOGS.glob("*.json")):
        t = stamp(p)
        if t and all(abs((t - r).total_seconds()) > 300 for r in rebuilt):
            paths.append(p)
    return paths


def record_logs(paths: list[Path] | None = None) -> list[tuple[str, str]]:
    """Our records as logs with the opponent (p2) preview, in the replay's text form; ids are <source>/<stem>."""
    out = []
    for p in record_paths() if paths is None else paths:
        rec = json.loads(p.read_text())
        if not rec.get("log") or not rec.get("preview"):
            continue
        lines = [ln for ln in rec["log"] if not ln.startswith("|poke|p2|")]
        at = next((i for i, ln in enumerate(lines) if ln.startswith("|poke|p1|")), 0)
        lines[at:at] = [f"|poke|p2|{s}, L50|" for s in rec["preview"]["species"]]
        if rec.get("result") in ("win", "loss") and not any(ln.startswith("|win|") for ln in lines):
            names = {ln.split("|")[2]: ln.split("|")[3] for ln in lines
                     if ln.startswith("|player|") and len(ln.split("|")) > 3 and ln.split("|")[3]}
            winner = "p1" if rec["result"] == "win" else "p2"
            lines.append("|win|" + names.get(winner, winner))
        out.append((f"{p.parent.name}/{p.stem}", "\n".join(lines)))
    return out


def build(args) -> None:
    sides = Sides(seed=0)
    gamma = 0.97

    def dump(items, roles, path):
        rows = {"obs": [], "masks": [], "actions": [], "returns": [], "battle": [], "kind": [], "label": [],
                "last": [], "turn": []}
        bad = 0
        for stem, log in items:
            for role in roles:
                try:
                    ds, won = decisions(log, role, sides)
                except Exception as e:  # noqa: BLE001
                    bad += 1
                    print("skip", stem, role, repr(e)[:120], flush=True)
                    continue
                for k, d in enumerate(ds):
                    rows["obs"].append(d.obs)
                    rows["masks"].append(d.mask)
                    rows["actions"].append(d.action)
                    ret = 0.0 if won is None else (1.0 if won else -1.0) * gamma ** (len(ds) - 1 - k)
                    rows["returns"].append(ret)
                    rows["battle"].append(f"{stem}:{role}")
                    rows["kind"].append(d.kind)
                    rows["label"].append(d.label)
                    rows["last"].append(d.last_move or "")
                    rows["turn"].append(d.turn)
        np.savez_compressed(path, obs=np.array(rows["obs"], dtype=np.float32), masks=np.array(rows["masks"]),
                            actions=np.array(rows["actions"]), returns=np.array(rows["returns"], dtype=np.float32),
                            battle=np.array(rows["battle"]), kind=np.array(rows["kind"]),
                            label=np.array(rows["label"]), last=np.array(rows["last"]), turn=np.array(rows["turn"]))
        n_turn = sum(k == "turn" for k in rows["kind"])
        print(f"{path}: {len(rows['actions'])} decisions ({n_turn} turn, {len(rows['actions']) - n_turn} forced) "
              f"from {len(set(rows['battle']))} sides; {bad} sides skipped", flush=True)

    dump(record_logs(), ["p2"], PROCESSED_DIR / "human_records.npz")
    if args.records_only:
        return
    replays = replay_logs()
    if args.limit:
        replays = replays[:args.limit]
    dump(replays, ["p1", "p2"], PROCESSED_DIR / "human_replays.npz")


def split(battles: np.ndarray, test: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Train / test indices with whole replays on one side (both players of a replay together). By a hash of the
    replay id, so a replay stays on its side as more are downloaded."""
    held = {b: int(hashlib.sha1(b.split(":")[0].encode()).hexdigest()[:8], 16) / 16 ** 8 < test
            for b in set(battles)}
    is_test = np.array([held[b] for b in battles])
    return np.where(~is_test)[0], np.where(is_test)[0]


def fit(args) -> None:
    import torch
    from sb3_contrib import MaskablePPO

    from pokesim.rl.bc import SpacesOnly
    from pokesim.rl.train import new_model

    data = {k: v for k, v in np.load(args.data).items()}
    tr, te = split(data["battle"])
    te_rec = np.array([], dtype=int)
    if args.records:  # our opponents: held out by battle, the rest repeated --records-weight times
        rec = np.load(args.records)
        rtr, rte = split(rec["battle"], test=args.records_test)
        n = len(data["actions"])
        for k in ("obs", "masks", "actions", "returns"):
            data[k] = np.concatenate([data[k], rec[k]])
        tr = np.concatenate([tr] + [rtr + n] * args.records_weight)
        te_rec = rte + n
        print(f"records: {len(rtr)} train decisions (x{args.records_weight}), {len(rte)} held out", flush=True)
    obs, masks = torch.tensor(data["obs"]), data["masks"]
    actions, rets = torch.tensor(data["actions"]), torch.tensor(data["returns"])
    if args.init:
        model = MaskablePPO.load(args.init, device="cuda")
    else:
        model = new_model(SpacesOnly(obs.shape[1]))
    policy = model.policy
    device = policy.device
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def evaluate(idx):
        with torch.no_grad():
            o = obs[idx].to(device)
            dist = policy.get_distribution(o, action_masks=masks[idx])
            probs = dist.distribution.probs.cpu()
            a = actions[idx]
            top1 = (probs.argmax(1) == a).float().mean().item()
            ll = torch.log(probs[torch.arange(len(a)), a].clamp_min(1e-3)).mean().item()
            return top1, ll

    best, best_state = -1e9, None
    if len(te_rec):
        policy.eval()
        print("before: held-out replays top-1 {:.1%} log-lik {:.2f} | held-out records top-1 {:.1%} log-lik {:.2f}"
              .format(*evaluate(te), *evaluate(te_rec)), flush=True)
    for epoch in range(args.epochs):
        policy.train()
        order = tr[np.random.permutation(len(tr))]
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            o = obs[idx].to(device)
            dist = policy.get_distribution(o, action_masks=masks[idx])
            logp = dist.log_prob(actions[idx].to(device))
            v = policy.predict_values(o).squeeze(1)
            loss = -logp.mean() + args.vf_coef * torch.nn.functional.mse_loss(v, rets[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        policy.eval()
        top1, ll = evaluate(te)
        line = f"epoch {epoch + 1}: held-out replays top-1 {top1:.1%}  log-lik {ll:.2f}"
        if len(te_rec):  # select on our opponents
            top1_r, ll_r = evaluate(te_rec)
            line += f" | held-out records top-1 {top1_r:.1%}  log-lik {ll_r:.2f}"
            ll = ll_r
        print(line, flush=True)
        if ll > best:
            best, best_state = ll, {k: v.clone() for k, v in policy.state_dict().items()}
    policy.load_state_dict(best_state)
    model.save(args.out)
    print(f"saved {args.out}.zip (best held-out log-lik {best:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--records-only", action="store_true", help="only rebuild human_records.npz")
    f = sub.add_parser("fit")
    f.add_argument("--data", default=str(PROCESSED_DIR / "human_replays.npz"))
    f.add_argument("--init", help="start from this policy (e.g. checkpoints/ppo_m6_league3m.zip)")
    f.add_argument("--out", default="checkpoints/bc_human")
    f.add_argument("--epochs", type=int, default=30)
    f.add_argument("--batch", type=int, default=256)
    f.add_argument("--lr", type=float, default=3e-4)
    f.add_argument("--weight-decay", type=float, default=1e-4)
    f.add_argument("--vf-coef", type=float, default=0.1)
    f.add_argument("--records", help="also train on our opponents (data/processed/human_records.npz); the best "
                                     "epoch is then chosen on its held-out battles")
    f.add_argument("--records-weight", type=int, default=10)
    f.add_argument("--records-test", type=float, default=0.25)
    args = ap.parse_args()
    build(args) if args.cmd == "build" else fit(args)


if __name__ == "__main__":
    main()
