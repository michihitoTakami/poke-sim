// One-turn lookahead on the Showdown sim: rebuild a battle position, try every pair of choices (ours x theirs),
// then play a few more turns with a greedy damage policy for both sides and score the result.
// Long-running: reads one JSON request per stdin line, writes one JSON response per line.
//
// Usage: node lookahead.js <showdown_dir>
// Request:  {"format", "seeds": n, "rollout": turns, "samples": [{"p1": SIDE, "p2": SIDE, "field": FIELD}, ...]}
//   SIDE  = {"team": [packed-able sets, active first], "mons": [{"hp": 0..1, "status", "boosts", "volatiles", "stall", "perish", "encore", "forme",
//            "mega": bool}], "megaUsed": bool, "conditions": {"stealthrock": 1, "reflect": turns, ...}}
//   FIELD = {"weather", "weatherTurns", "terrain", "terrainTurns", "trickroom": turns}
// Response: {"results": [{"p1": [labels], "p2": [labels], "payoff": [[...]]}, ...]} (payoff from p1's view)
const path = require('path');
const readline = require('readline');
const dir = process.argv[2];
const { Battle, Teams } = require(path.resolve(dir, 'dist/sim'));
const { State } = require(path.resolve(dir, 'dist/sim/state'));

const WIN_BONUS = 3;
const WIN_DECAY = 0.05;  // per turn: a quicker win scores higher

function setUp(sample, format, seed) {
  const battle = new Battle({ formatid: format, seed: [seed, seed + 1, seed + 2, seed + 3] });
  battle.setPlayer('p1', { name: 'p1', team: Teams.pack(sample.p1.team) });
  battle.setPlayer('p2', { name: 'p2', team: Teams.pack(sample.p2.team) });
  battle.choose('p1', 'team 123');
  battle.choose('p2', 'team 123');
  for (const sideId of ['p1', 'p2']) {
    const side = battle[sideId], spec = sample[sideId];
    side.pokemon.forEach((mon, i) => {
      const s = spec.mons[i] || {};
      if (i === 0 && s.mega && mon.canMegaEvo) battle.actions.runMegaEvo(mon);
      if (spec.megaUsed) mon.canMegaEvo = null;
      if (s.forme) mon.formeChange(s.forme, null, true);  // battle-only forme it is in (Mimikyu-Busted, ...)
      if (s.hp !== undefined && s.hp <= 0) {
        mon.hp = 0; mon.fainted = true; mon.status = 'fnt'; side.pokemonLeft--;
        return;
      }
      if (s.hp !== undefined) mon.hp = Math.max(1, Math.round(s.hp * mon.maxhp));
      mon.clearStatus();
      if (s.status) mon.setStatus(s.status, null, null, true);
      if (i === 0) {
        mon.clearBoosts();
        Object.assign(mon.boosts, s.boosts || {});
        for (const v of s.volatiles || []) mon.addVolatile(v);
        if (s.perish !== undefined) {
          mon.addVolatile('perishsong');
          if (mon.volatiles['perishsong']) mon.volatiles['perishsong'].duration = Math.max(1, s.perish);
        }
        if (s.encore && mon.hasMove(s.encore)) {  // Encore locks the last move used
          mon.lastMove = battle.dex.moves.get(s.encore);
          mon.addVolatile('encore');
        }
        if (s.stall) {  // Protect used s.stall turns in a row: success chance 1 / 3^stall
          mon.addVolatile('stall');
          if (mon.volatiles['stall']) mon.volatiles['stall'].counter = Math.min(729, 3 ** s.stall);
        }
      }
    });
    for (const id of Object.keys(side.sideConditions)) side.removeSideCondition(id);
    for (const [id, n] of Object.entries(spec.conditions || {})) {
      const layers = ['spikes', 'toxicspikes'].includes(id) ? n : 1;
      for (let k = 0; k < layers; k++) side.addSideCondition(id, 'debug');
      if (!['spikes', 'toxicspikes', 'stealthrock', 'stickyweb'].includes(id) && side.sideConditions[id]) {
        side.sideConditions[id].duration = n;
      }
    }
  }
  const f = sample.field || {};
  battle.field.clearWeather();
  if (f.weather) {
    battle.field.setWeather(f.weather, 'debug');
    if (f.weatherTurns && battle.field.weatherState) battle.field.weatherState.duration = f.weatherTurns;
  }
  battle.field.clearTerrain();
  if (f.terrain) {
    battle.field.setTerrain(f.terrain, 'debug');
    if (f.terrainTurns && battle.field.terrainState) battle.field.terrainState.duration = f.terrainTurns;
  }
  for (const id of Object.keys(battle.field.pseudoWeather)) battle.field.removePseudoWeather(id);
  if (f.trickroom) {
    battle.field.addPseudoWeather('trickroom', 'debug');
    battle.field.pseudoWeather.trickroom.duration = f.trickroom;
  }
  battle.makeRequest('move');
  return battle;
}

// Legal choices for one side as [choice, label]; labels name the move or the switch target.
function choices(battle, side) {
  const req = side.activeRequest;
  if (!req || req.wait) return [['pass', 'pass']];
  const out = [];
  if (req.forceSwitch) {
    side.pokemon.forEach((mon, i) => {
      if (i > 0 && !mon.fainted) out.push([`switch ${i + 1}`, `switch:${mon.species.id}`]);
    });
    return out.length ? out : [['pass', 'pass']];
  }
  const active = req.active[0];
  active.moves.forEach((m, i) => {
    if (m.disabled) return;
    out.push([`move ${i + 1}`, `move:${m.id}`]);
    if (active.canMegaEvo) out.push([`move ${i + 1} mega`, `move:${m.id}:mega`]);
  });
  if (!active.trapped) {
    side.pokemon.forEach((mon, i) => {
      if (i > 0 && !mon.fainted) out.push([`switch ${i + 1}`, `switch:${mon.species.id}`]);
    });
  }
  return out.length ? out : [['move 1', 'move:struggle']];
}

function expectedDamage(battle, source, target, moveId) {
  const move = battle.dex.getActiveMove(moveId);
  if (move.category === 'Status') return 0;
  const dmg = battle.actions.getDamage(source, target, move, true);
  return typeof dmg === 'number' ? dmg * (move.accuracy === true ? 1 : move.accuracy / 100) : 0;
}

// Greedy policy for rollouts: the move with the highest expected damage, or the bench mon that hits hardest.
function greedy(battle, side) {
  const opts = choices(battle, side);
  if (opts.length === 1) return opts[0][0];
  const foe = side.foe.active[0];
  if (!foe || foe.fainted) return opts[0][0];
  const me = side.active[0];
  let best = opts[0][0], bestScore = -1;
  for (const [choice, label] of opts) {
    let score;
    if (label.startsWith('move:')) {
      if (label.endsWith(':mega')) continue;
      score = expectedDamage(battle, me, foe, label.split(':')[1]);
    } else if (side.activeRequest.forceSwitch) {
      const mon = side.pokemon[Number(choice.split(' ')[1]) - 1];
      score = Math.max(0, ...mon.moveSlots.map(s => expectedDamage(battle, mon, foe, s.id)));
    } else {
      continue;  // no voluntary switching in rollouts
    }
    if (score > bestScore) { best = choice; bestScore = score; }
  }
  return best;
}

function material(side) {
  return side.pokemon.reduce((acc, mon) => acc + (mon.fainted ? 0 : mon.hp / mon.maxhp), 0);
}

function score(battle, startTurn) {
  let v = material(battle.p1) - material(battle.p2);
  if (battle.ended) {
    const bonus = WIN_BONUS * Math.max(0.5, 1 - WIN_DECAY * (battle.turn - startTurn));
    v += battle.winner === 'p1' ? bonus : battle.winner === 'p2' ? -bonus : 0;
  }
  return v;
}

function step(battle, c1, c2) {
  for (const [side, c] of [[battle.p1, c1], [battle.p2, c2]]) {
    if (side.activeRequest && !side.activeRequest.wait && c !== 'pass') battle.choose(side.id, c);
  }
}

function evaluate(sample, format, seeds, rollout) {
  const base = setUp(sample, format, 1);
  const p1 = choices(base, base.p1), p2 = choices(base, base.p2);
  const snapshot = JSON.stringify(State.serializeBattle(base));
  const payoff = p1.map(() => p2.map(() => 0));
  for (let s = 0; s < seeds; s++) {
    for (let i = 0; i < p1.length; i++) {
      for (let j = 0; j < p2.length; j++) {
        const battle = State.deserializeBattle(snapshot);
        battle.resetRNG([s * 4 + 11, s * 4 + 12, s * 4 + 13, s * 4 + 14]);
        try {
          step(battle, p1[i][0], p2[j][0]);
          for (let turns = 0, guard = 0; !battle.ended && turns < rollout && guard < rollout * 3; guard++) {
            if (battle.requestState === 'move') turns++;
            step(battle, greedy(battle, battle.p1), greedy(battle, battle.p2));
          }
        } catch (e) {
          // an illegal pairing (e.g. a sampled move the request rejects) scores as no progress
        }
        payoff[i][j] += score(battle, base.turn) / seeds;
      }
    }
  }
  return { p1: p1.map(x => x[1]), p2: p2.map(x => x[1]), payoff };
}

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', line => {
  let reply;
  try {
    const req = JSON.parse(line);
    reply = { results: req.samples.map(s => evaluate(s, req.format, req.seeds ?? 2, req.rollout ?? 3)) };
  } catch (e) {
    reply = { error: String(e && e.stack || e) };
  }
  process.stdout.write(JSON.stringify(reply) + '\n');
});
