// Damage rolls and speeds for candidate opponent sets, for inferring the opponent's sets from what the battle showed
// (src/pokesim/belief). Long-running: reads one JSON request per stdin line, writes one JSON response per line.
//
// Usage: node belief.js <showdown_dir>
// Request:  {"format", "jobs": [{"sample": {"p1": SIDE, "p2": SIDE, "field": FIELD}, "candidates": [set, ...],
//            "attacker": "p1" | "p2" (optional), "move": id, "crit": bool}]}
//   SIDE and FIELD as in lookahead.js. Each candidate replaces p2's active set (team[0]) in turn.
// Response: {"results": [[{"rolls": [16 damages, high roll first] | null, "hp": defender HP, "maxhp": defender max HP,
//              "speeds": [p1 action speed, p2 action speed]} | null (the position could not be built), ...], ...]}
//   rolls are the simulator's damage formula with each of the 16 random factors, the critical hit fixed to "crit";
//   null when the move does not damage (immune, status move, fixed failure). Action speeds include boosts, items,
//   paralysis, Tailwind and Trick Room (inverted), as the simulator orders moves of equal priority.
const path = require('path');
const readline = require('readline');
const dir = process.argv[2];
const {setUp} = require(path.resolve(__dirname, 'state_codec.js')).createCodec(dir);

function evaluate(job, candidate, format) {
  const sample = JSON.parse(JSON.stringify(job.sample));
  sample.p2.team = sample.p2.team.map((s, i) => (i === 0 || !s) ? candidate : s);  // padding repeats it
  const battle = setUp(sample, format, 1);
  const p1 = battle.p1.active[0], p2 = battle.p2.active[0];
  const out = {speeds: [p1.getActionSpeed(), p2.getActionSpeed()], rolls: null};
  if (job.attacker) {
    const [source, target] = job.attacker === 'p1' ? [p1, p2] : [p2, p1];
    out.hp = target.hp;
    out.maxhp = target.maxhp;
    const tr = battle.trunc;
    const rolls = [];
    for (let r = 0; r < 16; r++) {
      battle.randomizer = d => tr(tr(d * (100 - r)) / 100);
      const move = battle.dex.getActiveMove(job.move);
      move.willCrit = !!job.crit;
      // as during a move: abilities such as Unaware check the active attacker / target / move
      battle.activeMove = move; battle.activePokemon = source; battle.activeTarget = target;
      const dmg = battle.actions.getDamage(source, target, move, true);
      if (typeof dmg !== 'number') { rolls.length = 0; break; }
      rolls.push(dmg);
    }
    if (rolls.length === 16) out.rolls = rolls;
  }
  return out;
}

const rl = readline.createInterface({input: process.stdin});
rl.on('line', line => {
  let reply;
  try {
    const req = JSON.parse(line);
    reply = {results: req.jobs.map(job => job.candidates.map(c => {
      try {
        return evaluate(job, c, req.format);
      } catch (e) {
        return null;
      }
    }))};
  } catch (e) {
    reply = {error: String(e && e.stack || e)};
  }
  process.stdout.write(JSON.stringify(reply) + '\n');
});
