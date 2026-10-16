"""The expert / disrupt continuation policies (the Baton Pass player's principles, docs/winning-ai-roadmap.md 8.2),
side-specific continuation pairs, strict terminal scoring and damage queries that leave the battle unchanged."""
import json
import subprocess

from pokesim.config import ROOT, SHOWDOWN_DIR

PHYS = {"hp": 32, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 2}
SPEC = {"hp": 32, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 2}
FAST = {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32}
BULK = {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 2, "spe": 0}
PASSER = {"species": "Scolipede", "item": "Mental Herb", "ability": "Speed Boost", "nature": "Jolly",
          "moves": ["Iron Defense", "Protect", "Substitute", "Baton Pass"],
          "evs": {"hp": 30, "atk": 0, "def": 13, "spa": 0, "spd": 0, "spe": 23}, "level": 50}
GHOLDENGO = {"species": "Gholdengo", "item": "Air Balloon", "ability": "Good as Gold", "nature": "Modest",
             "moves": ["Shadow Ball", "Flash Cannon", "Recover", "Nasty Plot"],
             "evs": {"hp": 13, "atk": 0, "def": 0, "spa": 21, "spd": 0, "spe": 32}, "level": 50}
FLOETTE = {"species": "Floette-Eternal", "item": "Leftovers", "ability": "Flower Veil", "nature": "Bold",
           "moves": ["Moonblast", "Substitute", "Synthesis", "Calm Mind"], "evs": BULK, "level": 50}
DIRGE = {"species": "Skeledirge", "item": "Sitrus Berry", "ability": "Unaware", "nature": "Bold",
         "moves": ["Torch Song", "Will-O-Wisp", "Shadow Ball", "Slack Off"], "evs": BULK, "level": 50}
CHOMP = {"species": "Garchomp", "item": "Life Orb", "ability": "Rough Skin", "nature": "Jolly",
         "moves": ["Earthquake", "Dragon Claw", "Swords Dance", "Protect"], "evs": PHYS, "level": 50}
HIPPO = {"species": "Hippowdon", "item": "Leftovers", "ability": "Sand Stream", "nature": "Impish",
         "moves": ["Earthquake", "Yawn", "Slack Off", "Stealth Rock"], "evs": PHYS, "level": 50}
TAILER = dict(HIPPO, species="Dragonite", ability="Multiscale", moves=["Dragon Tail", "Extreme Speed", "Roost",
                                                                        "Earthquake"])
SLOW_SPECIAL = {"species": "Hatterene", "item": "Leftovers", "ability": "Magic Bounce", "nature": "Quiet",
                "moves": ["Psychic", "Dazzling Gleam", "Mystical Fire", "Calm Mind"], "evs": SPEC, "level": 50}
WEAK = {"species": "Blissey", "item": "Leftovers", "ability": "Natural Cure", "nature": "Timid",
        "moves": ["Dazzling Gleam", "Soft-Boiled", "Protect", "Heal Bell"],
        "evs": {"hp": 32, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 32}, "level": 50}


def run(code: str, sample: dict) -> dict:
    js = """
const fs = require('fs'), vm = require('vm');
const sandbox = {__dirname: require('path').dirname(process.argv[1]),
    require: n => n === 'readline' ? {createInterface: () => ({on: () => {}})} : require(n),
    process: {argv: ['node', 'worker', process.argv[2]]}, module: {exports: {}}, console};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8') +
    '\\nmodule.exports={setUp,evaluate,tactical,expert,disrupt,receiver,'
    + 'expectedDamage,damageRoll,policy,speedOf};', sandbox);
const sample = JSON.parse(process.argv[3]);
const {setUp,evaluate,tactical,expert,disrupt,receiver,expectedDamage,damageRoll,policy,speedOf} =
    sandbox.module.exports;
const format = 'gen9championsbssregmc';
const label = (b, side, c) => {
  const [kind, n] = c.split(' ');
  if (kind === 'switch') return 'switch:' + side.pokemon[Number(n) - 1].species.id;
  return 'move:' + side.active[0].moveSlots[Number(n) - 1].id;
};
""" + code
    out = subprocess.run(["node", "-e", js, str(ROOT / "services/sim-tools/lookahead.js"), str(SHOWDOWN_DIR),
                          json.dumps(sample)], capture_output=True, text=True, check=True, timeout=60)
    return json.loads(out.stdout)


def side(team, mons=None):
    return {"team": team, "mons": mons or [{"hp": 1.0} for _ in team]}


def choose(policy: str, ours: dict, theirs: dict) -> str:
    sample = {"p1": ours, "p2": theirs, "field": {}}
    return run(f"const b = setUp(sample, format, 1); console.log(JSON.stringify(label(b, b.p1, {policy}(b, b.p1))));",
               sample)


def test_damage_queries_leave_the_prng_and_the_active_move_alone():
    sample = {"p1": side([CHOMP]), "p2": side([HIPPO]), "field": {}}
    out = run("const b = setUp(sample, format, 1); const before = b.prng.getSeed();"
              "const d = [1, 2, 3].map(() => expectedDamage(b, b.p1.active[0], b.p2.active[0], 'earthquake'));"
              "console.log(JSON.stringify({same: before === b.prng.getSeed(), d, move: b.activeMove}));", sample)
    assert out["same"] and out["move"] is None
    assert out["d"][0] > 0 and len(set(out["d"])) == 1  # deterministic: the mean of the highest and lowest roll


def test_damage_queries_see_unaware():
    boosted = {"p1": side([CHOMP], [{"hp": 1.0, "boosts": {"atk": 2}}]), "p2": side([DIRGE]), "field": {}}
    plain = {"p1": side([CHOMP]), "p2": side([DIRGE]), "field": {}}
    code = ("const b = setUp(sample, format, 1);"
            "console.log(JSON.stringify(damageRoll(b, b.p1.active[0], b.p2.active[0], 'earthquake', 0)));")
    assert run(code, boosted) == run(code, plain)  # Unaware ignores the +2


def test_principle_1_stacks_iron_defense_against_a_physical_foe():
    ours = side([PASSER, GHOLDENGO], [{"hp": 1.0, "boosts": {"def": 2}}, {"hp": 1.0}])
    theirs = side([CHOMP])
    assert choose("tactical", ours, theirs) == "move:batonpass"  # tactical passes the first +2 away
    assert choose("expert", ours, theirs) == "move:irondefense"
    maxed = side([PASSER, GHOLDENGO], [{"hp": 1.0, "boosts": {"def": 6, "spe": 1}}, {"hp": 1.0}])
    assert choose("expert", maxed, theirs) == "move:substitute"  # at +6 Earthquake no longer breaks it
    covered = side([PASSER, GHOLDENGO], [{"hp": 0.75, "boosts": {"def": 6, "spe": 1}, "volatiles": ["substitute"]},
                                         {"hp": 1.0}])
    assert choose("expert", covered, theirs) == "move:batonpass"  # the boosts and the Substitute go on


def test_principle_5_substitute_when_the_foe_cannot_break_it():
    ours = side([PASSER, FLOETTE])  # Floette is slower than this Blissey: principle 8 does not apply
    assert choose("expert", ours, side([WEAK])) == "move:substitute"  # a resisted Dazzling Gleam
    assert choose("expert", ours, side([CHOMP])) != "move:substitute"  # Earthquake breaks it


def test_principle_6_stalls_for_speed_against_a_special_foe_then_passes():
    floette = dict(FLOETTE, evs=BULK)
    ours = side([PASSER, floette], [{"hp": 1.0, "volatiles": ["substitute"], "boosts": {"def": 1}}, {"hp": 1.0}])
    assert choose("expert", ours, side([SLOW_SPECIAL])) in ("move:protect", "move:batonpass")
    fast = side([PASSER, floette], [{"hp": 1.0, "volatiles": ["substitute"], "boosts": {"def": 1, "spe": 4}},
                                    {"hp": 1.0}])
    assert choose("expert", fast, side([SLOW_SPECIAL])) == "move:batonpass"


def test_principles_2_3_7_pick_the_receiver_by_the_threat():
    code = ("const b = setUp(sample, format, 1);"
            "console.log(JSON.stringify(label(b, b.p1, receiver(b, b.p1, true))));")
    mons = [{"hp": 1.0, "boosts": {"def": 2}}, {"hp": 1.0}, {"hp": 1.0}]

    def pick(team, foe, boosts=None):
        return run(code, {"p1": side(team, mons), "p2": side([foe], [{"hp": 1.0, "boosts": boosts or {}}]),
                          "field": {}})

    assert pick([PASSER, FLOETTE, GHOLDENGO], HIPPO) == "switch:gholdengo"  # Yawn
    assert pick([PASSER, GHOLDENGO, FLOETTE], TAILER) == "switch:floetteeternal"  # Dragon Tail
    assert pick([PASSER, FLOETTE, DIRGE], CHOMP, {"atk": 2}) == "switch:skeledirge"  # a Swords Dance


def test_principle_8_no_set_up_when_the_attackers_already_outspeed():
    slow = dict(HIPPO, moves=["Earthquake", "Slack Off", "Stealth Rock", "Rock Slide"])
    fast_attacker = dict(GHOLDENGO, evs=FAST, nature="Timid")
    ours = side([PASSER, fast_attacker])
    assert choose("expert", ours, side([slow])) == choose("tactical", ours, side([slow]))


def test_disrupt_yawns_a_set_up_passer():
    ours = side([HIPPO])
    theirs = side([PASSER, GHOLDENGO], [{"hp": 1.0, "boosts": {"def": 2, "spe": 1}}, {"hp": 1.0}])
    assert choose("disrupt", ours, theirs) == "move:yawn"
    assert choose("tactical", ours, theirs) == "move:earthquake"


def test_pairs_and_strict_terminal_scoring():
    win = {"p1": side([CHOMP, PASSER], [{"hp": 1.0, "boosts": {"atk": 2}}, {"hp": 1.0}]),
           "p2": side([HIPPO, PASSER], [{"hp": 0.05}, {"hp": 0}]), "field": {}}
    out = run("console.log(JSON.stringify(evaluate(sample, format, 1, 2, 0, "
              "{scoring: 'terminal', tiebreak: 0, continuations: ['expert/disrupt']})));", win)
    assert list(out["byContinuation"]) == ["expert/disrupt"]
    row = out["payoff"][out["p1"].index("move:earthquake")]
    assert all(v == 1 for v in row)  # a win is +1 however long it takes


def plan_search(request: dict) -> dict:
    out = subprocess.run(["node", str(ROOT / "services/sim-tools/lookahead.js"), str(SHOWDOWN_DIR)],
                         input=json.dumps(request) + "\n", capture_output=True, text=True, check=True, timeout=120)
    return json.loads(out.stdout)


def test_a_plan_stops_stacking_at_its_cap():
    theirs = side([CHOMP])
    for cap, boosts, expect in ((2, {"def": 2}, "move:batonpass"), (4, {"def": 2}, "move:irondefense"),
                                (4, {"def": 4}, "move:batonpass")):
        ours = side([PASSER, GHOLDENGO], [{"hp": 1.0, "boosts": dict(boosts, spe=1)}, {"hp": 1.0}])
        sample = {"p1": ours, "p2": theirs, "field": {}}
        code = "const b = setUp(sample, format, 1); console.log(JSON.stringify(label(b, b.p1, plan(b, b.p1))));"
        assert run(f"const plan = policy('expert@{cap}');" + code, sample) == expect


def test_plan_search_plays_every_plan_to_the_end_on_the_same_seeds():
    sample = {"p1": side([PASSER, GHOLDENGO, FLOETTE]), "p2": side([CHOMP, HIPPO, TAILER]), "field": {}}
    req = {"format": "gen9championsbssregmc", "samples": [sample, sample],
           "plans": {"candidates": ["expert", "expert@0", "expert@6h33"], "opponents": ["tactical", "disrupt"],
                     "seeds": 2, "maxTurns": 60}}
    out = plan_search(req)
    assert out["cycles"] == [2, 2]
    for name, r in out["plans"].items():
        assert r["win"] + r["loss"] + r["draw"] + r["unfinished"] + r["failed"] == 2 * 2 * 2  # seeds x samples x opp
        assert r["failed"] == 0
        assert r["perSample"][0] == r["perSample"][1]  # the same position and seeds: the same games
    again = plan_search(req)
    assert again["plans"] == out["plans"]  # reproducible


def test_unknown_plans_are_rejected():
    req = {"format": "gen9championsbssregmc", "samples": [], "plans": {"candidates": ["expert@9"]}}
    assert "unknown continuation" in plan_search(req)["error"]


def test_with_plan_replaces_only_our_expert_continuation():
    from pokesim.live.advisor import with_plan

    settings = {"continuations": ["expert/tactical", "expert/disrupt", "tactical", "greedy/expert"], "seeds": 1}
    out = with_plan(settings, "expert@4h33")
    assert out["continuations"] == ["expert@4h33/tactical", "expert@4h33/disrupt", "tactical", "greedy/expert"]
    assert settings["continuations"][0] == "expert/tactical"  # not changed in place


def test_no_passing_back_and_forth_or_passing_a_single_boost():
    umbreon = {"species": "Umbreon", "item": "Leftovers", "ability": "Inner Focus", "nature": "Bold",
               "moves": ["Snarl", "Baton Pass", "Calm Mind", "Moonlight"], "evs": BULK, "level": 50}
    code = ("const b = setUp(sample, format, 1);"
            "console.log(JSON.stringify(label(b, b.p1, receiver(b, b.p1, true))));")
    mons = [{"hp": 1.0, "boosts": {"spa": 2, "spd": 2}}, {"hp": 1.0}, {"hp": 1.0}]
    sample = {"p1": side([umbreon, PASSER, FLOETTE], mons), "p2": side([SLOW_SPECIAL]), "field": {}}
    assert run(code, sample) == "switch:floetteeternal"  # not the other passer, which cannot hurt the foe
    one = side([PASSER, GHOLDENGO], [{"hp": 1.0, "boosts": {"spe": 1}}, {"hp": 1.0}])
    assert choose("expert", one, side([WEAK])) != "move:batonpass"  # +1 Speed alone is not worth a turn yet


def test_a_passer_cannot_outspeed_a_lethal_priority_move():
    ours = side([PASSER, GHOLDENGO], [{'hp': .05, 'boosts': {'spe': 6}}, {'hp': 1}])
    assert choose('expert', ours, side([dict(CHOMP, moves=['Extreme Speed'])])) == 'switch:gholdengo'


def test_can_set_up_sees_boosting_moves_and_baton_pass():
    from poke_env.battle import Pokemon

    from pokesim.live.advisor import can_set_up

    def mon(*moves):
        m = Pokemon(gen=9, species="gholdengo")
        for move in moves:
            m._add_move(move)
        return m

    assert can_set_up(mon("shadowball", "nastyplot"))
    assert can_set_up(mon("snarl", "batonpass"))
    assert not can_set_up(mon("shadowball", "flashcannon", "recover"))


TR_SETTER = {"species": "Hatterene", "item": "Leftovers", "ability": "Magic Bounce", "nature": "Quiet",
             "moves": ["Trick Room", "Psychic", "Dazzling Gleam", "Mystical Fire"], "evs": SPEC, "level": 50}


def test_trick_room_inverts_speed_for_bench_and_active_alike():
    code = ("const b = setUp(sample, format, 1); const s = b.p1.pokemon;"
            "console.log(JSON.stringify([speedOf(s[0]) > speedOf(b.p2.active[0]), speedOf(s[1]) > speedOf(s[2])]));")
    sample = {"p1": side([GHOLDENGO, GHOLDENGO, DIRGE]), "p2": side([SLOW_SPECIAL]), "field": {}}
    fast = dict(GHOLDENGO, evs=FAST)
    sample["p1"]["team"] = [fast, fast, DIRGE]
    assert run(code, sample) == [True, True]  # Gholdengo outspeeds Hatterene and Skeledirge
    sample["field"] = {"trickroom": 3}
    assert run(code, sample) == [False, False]  # under Trick Room both comparisons flip


def test_disrupt_sets_trick_room_against_a_faster_foe():
    ours = side([TR_SETTER])
    theirs = side([dict(GHOLDENGO, evs=FAST)])  # faster, and Hatterene cannot knock it out
    assert choose("disrupt", ours, theirs) == "move:trickroom"
    sample = {"p1": ours, "p2": theirs, "field": {"trickroom": 3}}
    out = run("const b = setUp(sample, format, 1); console.log(JSON.stringify(label(b, b.p1, disrupt(b, b.p1))));",
              sample)
    assert out != "move:trickroom"  # not again while it is up (that would end it)


def test_bench_speed_includes_tailwind_items_status_and_weather_after_the_pass():
    code = ("const b = setUp(sample, format, 1);"
            "console.log(JSON.stringify(b.p1.pokemon.map(m=>speedOf(m))));")
    scarf = dict(GHOLDENGO, item='Choice Scarf')
    ours = side([scarf, scarf], [{'hp': 1, 'status': 'par'}, {'hp': 1, 'status': 'par'}])
    ours['conditions'] = {'tailwind': 3}
    speed = run(code, {'p1': ours, 'p2': side([SLOW_SPECIAL]), 'field': {}})
    assert speed[0] == speed[1]
    swimmer = dict(GHOLDENGO, ability='Swift Swim')
    speed = run(code, {'p1': side([swimmer, swimmer]), 'p2': side([SLOW_SPECIAL]),
                       'field': {'weather': 'raindance'}})
    assert speed[0] == speed[1]
