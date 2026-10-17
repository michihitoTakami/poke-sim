"""A live battle reconstructed from Showdown protocol lines, for advice outside a Showdown battle room.

Whatever the input (manual entry in the web UI, screen reading, a recorded replay), it is turned into protocol
lines (`|move|p2a: Garchomp|Earthquake|p1a: Scolipede`, `|-damage|p1a: Scolipede|120/165`, ...) and fed to a
poke-env `Battle`. Our own side is known from the team file, so the "request" Showdown would send us is built here
(actual stats, max HP and PP from the simulator via services/sim-tools/stats.js). The resulting `Battle` works with
the existing features, policies and lookahead unchanged.
"""
import hashlib
import json
import logging
import subprocess
import uuid
from pathlib import Path

from poke_env.battle import Battle, Effect, MoveCategory

from pokesim.config import FORMAT, RAW_DIR, ROOT, SHOWDOWN_DIR
from pokesim.rl.myteam import FixedTeam
from pokesim.state.tracker import TrackedBattle

STATS_CACHE = RAW_DIR / "live/team_stats"


def team_stats(sets: list[dict], format_id: str = FORMAT) -> list[dict]:
    """Simulator-computed details per set (cached by team contents)."""
    text = FixedTeam.to_showdown(sets)
    path = STATS_CACHE / f"{hashlib.sha1((format_id + text).encode()).hexdigest()[:16]}.json"
    if path.exists():
        return json.loads(path.read_text())
    out = subprocess.run(["node", str(ROOT / "services/sim-tools/stats.js"), str(SHOWDOWN_DIR), format_id],
                         input=text, capture_output=True, text=True, check=True).stdout
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out)
    return json.loads(out)


class BattleSession:
    """Our side is `role` (p1 by default) playing `sets` (6 sets, in team order)."""

    def __init__(self, sets: list[dict], username: str = "me", opponent: str = "opponent", role: str = "p1",
                 format_id: str = FORMAT):
        self.sets, self.username, self.role = sets, username, role
        self.format_id = format_id
        self.forme_stats = {}
        self.opp_role = "p2" if role == "p1" else "p1"
        self.info = team_stats(sets, format_id)
        self.battle = TrackedBattle(f"live-{uuid.uuid4().hex[:8]}", username, logging.getLogger("live"), gen=9)
        self.log: list[str] = []
        self.pp_used: dict[str, dict[str, int]] = {}
        self.rqid = 0
        self.partial_lineup = False
        self.selected: list[int] | None = None  # team indices brought (lead first), set at team preview
        self.apply([f"|player|{role}|{username}|", f"|player|{self.opp_role}|{opponent}|", "|gen|9",
                    "|gametype|singles"])

    # --- input -------------------------------------------------------------------------------------------------
    def apply(self, lines: list[str] | str) -> None:
        for line in [lines] if isinstance(lines, str) else lines:
            if not line.startswith("|"):
                continue
            self.log.append(line)
            split = line.split("|")
            kind = split[1] if len(split) > 1 else ""
            if self.partial_lineup and kind in {'switch', 'drag', 'replace'} and split[2].startswith(self.role):
                from pokesim.live.provenance import base

                index = next((i for i, s in enumerate(self.sets) if base(s['species']) == base(split[3])), None)
                if index is not None and index not in self.selected:
                    self.selected.append(index)
            if kind == '-transform' and split[3].startswith(self.role):
                target = self.battle.active_pokemon
                if target is not None:
                    info = next((i for i in self.info if i['name'] == target.name), None)
                    if info:
                        target.stats.update(self._current_stats(info, target))
            if kind == "win":
                self.battle.won_by(split[2])
            elif kind == "tie":
                self.battle.tied()
            elif kind == "move" and split[2].startswith(self.role) and "[from]" not in line:
                name = split[2].split(": ", 1)[1]
                move = split[3].lower().replace(" ", "").replace("-", "")
                used = self.pp_used.setdefault(name, {})
                foe = self.battle.opponent_active_pokemon
                pressure = foe is not None and foe.ability == "pressure" and len(split) > 4 and \
                    split[4].startswith(self.opp_role)
                used[move] = used.get(move, 0) + (2 if pressure else 1)
                self.battle.parse_message(split)
            elif kind in ("request", "teampreview", "t:", "j", "l", "c", "raw", "inactive", "uhtml", "badge"):
                continue
            else:
                self.battle.parse_message(split)

    def preview(self, opponent_species: list[str], opponent_details: list[str] | None = None) -> None:
        """Team preview: the opponent's 6 (species names, or full details strings like 'Garchomp, L50, M')."""
        details = opponent_details or [f"{s}, L50" for s in opponent_species]
        own = [f"|poke|{self.role}|{i['details']}|" for i in self.info]
        opp = [f"|poke|{self.opp_role}|{d}|" for d in details]
        self.apply(["|clearpoke", *own, *opp, "|teampreview|3"])
        self.battle.parse_request(self.request(team_preview=True))

    def select(self, order: list[int]) -> None:
        """The 3 we bring, as indices into our team, lead first."""
        self.selected = list(order)

    # --- our request -------------------------------------------------------------------------------------------
    def _mon(self, info: dict):
        return self.battle.team.get(f"{self.role}: {info['name']}")

    def _active_moves(self, info: dict, mon) -> list[dict]:
        used = self.pp_used.get(info["name"], {})
        effects = mon.effects if mon else {}
        last = None
        for line in reversed(self.log):  # since it came in: a Choice lock or Encore ends when it switches out
            if line.startswith((f"|switch|{self.role}a: ", f"|drag|{self.role}a: ", f"|replace|{self.role}a: ")):
                break
            if line.startswith(f"|move|{self.role}a: {info['name']}|") and "[from]" not in line:
                last = line.split("|")[3].lower().replace(" ", "").replace("-", "")
                break
        choice_locked = mon is not None and mon.item in ("choicescarf", "choiceband", "choicespecs") and last
        disabled_move = None
        for line in reversed(self.log):  # Disable on our active Pokémon (ends with |-end|...|Disable, or a switch)
            if line.startswith((f"|switch|{self.role}a: ", f"|drag|{self.role}a: ")):
                break
            if line.startswith(f"|-end|{self.role}a: {info['name']}|") and "Disable" in line:
                break
            if line.startswith(f"|-start|{self.role}a: {info['name']}|Disable|"):
                disabled_move = line.split("|")[4].lower().replace(" ", "").replace("-", "")
                break
        out = []
        locks = self.battle.state_tracker.locks(self.role, info["name"])
        encore_move = locks.get("encore", last)
        for m in info["moves"]:
            pp = max(0, m["maxpp"] - used.get(m["id"], 0))
            disabled = pp == 0 or m["id"] == disabled_move
            if Effect.ENCORE in effects and encore_move and m["id"] != encore_move:
                disabled = True
            if choice_locked and m["id"] != last:
                disabled = True
            known = mon.moves.get(m["id"]) if mon else None
            if known is not None:
                if Effect.TAUNT in effects and known.category == MoveCategory.STATUS:
                    disabled = True
                if Effect.HEAL_BLOCK in effects and "heal" in known.flags:
                    disabled = True
                if Effect.THROAT_CHOP in effects and "sound" in known.flags:
                    disabled = True
            out.append({"move": m["move"], "id": m["id"], "pp": pp, "maxpp": m["maxpp"], "target": m["target"],
                        "disabled": disabled})
        return out

    def _current_stats(self, info, mon):
        from pokesim.sim.formes import current_forme

        if mon is None or current_forme(mon) == info['species'].lower().replace('-', ''):
            return info['stats']
        key = (info['name'], current_forme(mon))
        if key not in self.forme_stats:
            original = self.sets[self.info.index(info)]
            self.forme_stats[key] = team_stats([dict(original, species=current_forme(mon))], self.format_id)[0]['stats']
        return self.forme_stats[key]

    def request(self, force_switch: bool = False, team_preview: bool = False) -> dict:
        self.rqid += 1
        pokemon, active_info = [], None
        brought = self.info if team_preview or self.selected is None else [self.info[i] for i in self.selected]
        for info in brought:
            mon = self._mon(info)
            maxhp = info["maxhp"]
            if mon is not None and mon.fainted:
                condition = "0 fnt"
            else:
                hp = maxhp if mon is None or mon.current_hp is None else round(mon.current_hp_fraction * maxhp)
                status = f" {mon.status.name.lower()}" if mon is not None and mon.status else ""
                condition = f"{hp}/{maxhp}{status}"
            active = bool(mon is not None and mon.active and not team_preview)
            if active:
                active_info = info
            item = info["item"] if mon is None or mon.item == "unknown_item" else (mon.item or "")
            # A receiver can already be Mega evolved. Reusing the original team-file details here reverts
            # its observed forme when poke-env refreshes the next request.
            details = mon._last_details if mon is not None and mon._last_details else info['details']
            pokemon.append({"ident": f"{self.role}: {info['name']}", "details": details,
                            "condition": condition, "active": active, "stats": self._current_stats(info, mon),
                            "moves": [m["id"] for m in info["moves"]], "baseAbility": info["ability"],
                            "item": item, "pokeball": "pokeball",
                            "ability": mon.ability if mon is not None and mon.ability else info["ability"]})
        pokemon.sort(key=lambda p: not p["active"])
        req: dict = {"side": {"name": self.username, "id": self.role, "pokemon": pokemon}, "rqid": self.rqid}
        if team_preview:
            req.update(teamPreview=True, maxTeamSize=3)
        elif force_switch or active_info is None:
            req["forceSwitch"] = [True]
        else:
            mon = self._mon(active_info)
            active = {"moves": self._active_moves(active_info, mon)}
            if active_info["megaStone"] and not self.battle.used_mega_evolve and "mega" not in mon.species:
                active["canMegaEvo"] = True
            req["active"] = [active]
        return req

    def last_move(self, role: str | None = None) -> str | None:
        """'move:<id>' last used by a side's active Pokémon since it came in (default: the opponent)."""
        role = role or self.opp_role
        for line in reversed(self.log):
            if line.startswith((f"|switch|{role}a", f"|drag|{role}a")):
                return None
            if line.startswith(f"|move|{role}a") and "[from]" not in line:
                return "move:" + line.split("|")[3].lower().replace(" ", "").replace("-", "")
        return None

    def decision(self, force_switch: bool = False) -> Battle:
        """Refresh our side from the team file and return the battle, ready for a decision."""
        self.battle.parse_request(self.request(force_switch=force_switch))
        return self.battle

    def save_log(self, path: str | Path) -> None:
        Path(path).write_text("\n".join(self.log) + "\n")
