// One-turn lookahead on the Showdown sim: rebuild a battle position, try every pair of choices (ours x theirs),
// then play a few more turns with continuation policies for both sides and score the result.
// Long-running: reads one JSON request per stdin line, writes one JSON response per line.
//
// Usage: node lookahead.js <showdown_dir>
// Request:  {"format", "seeds": n, "rollout": turns, "winDecay": per turn (default 0.05; 0 = a win is a win whenever it
//            comes), "continuations": ["greedy", "tactical"] (default ["greedy"]; each branch is played on with each),
//            "scoring": "legacy" | "terminal", "lostPathPenalty": (default 0), "samples": [{"p1": SIDE, "p2": SIDE,
//            "field": FIELD}, ...]}
//   continuations: greedy = the highest expected damage, no Mega, no voluntary switch, status moves never chosen;
//     tactical = also takes a KO, Mega Evolves, recovers, sets up while it is safe (up to +2), passes boosts or
//     switches when the active cannot damage the foe.
//   scoring: legacy = HP balance (each Pokemon 0..1) + 3 for a win (less the slower it comes);
//     terminal = a real end is +-1 (0 for a tie), less 0.005 per turn it took (a tie-break among wins, at least
//     0.95); a position still going is the HP balance scaled into (-0.5, 0.5) plus the lost-path term, kept
//     within +-0.9, so it never outranks a real win. It is a stand-in, not a win probability (see "resolved").
//   lostPathPenalty: subtracted when we have no damage source left (no remaining Pokemon with a damaging move the
//     foe's remaining Pokemon are not immune to by type) while the foe still has one; added in the mirror case.
//     Not a loss: Struggle, residual damage, Perish Song or a timer can still decide it.
//   SIDE  = {"team": [packed-able sets, active first], "mons": [{"hp": 0..1, "status", "boosts", "volatiles", "stall", "perish", "encore", "forme",
//            "mega": bool}], "megaUsed": bool, "conditions": {"stealthrock": 1, "reflect": turns, ...}}
//   FIELD = {"weather", "weatherTurns", "terrain", "terrainTurns", "trickroom": turns}
//   SIDE may include incoming (zero-based bench index) and switchMove (Baton Pass): execute its entry after restoring.
// Response: {"results": [{"p1": [labels], "p2": [labels], "payoff": [[...]], "invalid": [[...]],
//              "resolved": [[...]], "byContinuation": {name: [[...]]}}, ...]}
//   payoff from p1's view, averaged over the (seed, continuation) runs that ran; null when none did. invalid: per
//   pair, how many runs failed (an illegal pairing, e.g. a sampled move the request rejects): a failure is not
//   scored as a result. resolved: share of the runs that reached a real end of the battle. byContinuation: the
//   payoff per continuation policy (how much they disagree). passes: [done, planned] (see deadlineMs).
// "deadlineMs" (optional): no new pass (one run per pair, see prepare) starts after this budget; the first always runs.
const path = require('path');
const readline = require('readline');
const dir = process.argv[2];
const { Battle, Teams } = require(path.resolve(dir, 'dist/sim'));
const { State } = require(path.resolve(dir, 'dist/sim/state'));

const WIN_BONUS = 3;
const WIN_DECAY = 0.05;  // per turn: a quicker win scores higher
// terminal scoring: a win is 1 less TERMINAL_TIEBREAK per turn it takes (at least TERMINAL_FLOOR); a position still
// going stays within +-TERMINAL_OPEN, below any win and above any loss
const TERMINAL_TIEBREAK = 0.005, TERMINAL_FLOOR = 0.95, TERMINAL_OPEN = 0.9;

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
      if (s.mega) {
        // A previously evolved bench member remains Mega when it comes back in, even after the right was used.
        mon.canMegaEvo = battle.actions.canMegaEvo(mon);
        if (mon.canMegaEvo) battle.actions.runMegaEvo(mon);
      }
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
  for (const sideId of ['p1', 'p2']) {
    const spec = sample[sideId];
    if (spec.incoming === undefined) continue;
    const incoming = battle[sideId].pokemon[spec.incoming];
    const source = spec.switchMove ? battle.dex.moves.get(spec.switchMove) : null;
    if (!battle.actions.switchIn(incoming, 0, source)) throw new Error('Incoming switch rejected');
    // Execute the simulator's entry events (hazards, abilities, items and Baton Pass), without playing a turn.
    while (!battle.ended && battle.queue.peek()?.choice === 'runSwitch') {
      battle.runAction(battle.queue.shift());
    }
  }
  if (!battle.ended) {
    battle.checkFainted();
    battle.makeRequest(battle.sides.some(side => side.active.some(mon => mon.switchFlag)) ? 'switch' : 'move');
  }
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

function bestDamage(battle, source, target) {
  return Math.max(0, ...source.moveSlots.filter(s => s.pp > 0).map(s => expectedDamage(battle, source, target, s.id)));
}

function selfBoosts(move) {
  if (move.target === 'self' && move.boosts) return move.boosts;
  return null;
}

// Tactical policy for rollouts: what a reasonable player does beyond "hit hardest" (see the header).
function tactical(battle, side) {
  const opts = choices(battle, side);
  if (opts.length === 1) return opts[0][0];
  const foe = side.foe.active[0];
  if (!foe || foe.fainted) return opts[0][0];
  const switchTo = choice => side.pokemon[Number(choice.split(' ')[1]) - 1];
  const switches = opts.filter(([, label]) => label.startsWith('switch:'));
  const bestSwitch = () => {
    let best = null, bestScore = 0;
    for (const [choice] of switches) {
      const mon = switchTo(choice), score = bestDamage(battle, mon, foe) + mon.hp / mon.maxhp;
      if (bestDamage(battle, mon, foe) > 0 && score > bestScore) { best = choice; bestScore = score; }
    }
    return best;
  };
  if (side.activeRequest.forceSwitch) return bestSwitch() || greedy(battle, side);
  const me = side.active[0];
  const moves = opts.filter(([, label]) => label.startsWith('move:') && !label.endsWith(':mega'))
    .map(([choice, label]) => {
      const move = battle.dex.moves.get(label.split(':')[1]);
      return {choice, label, move, damage: expectedDamage(battle, me, foe, move.id)};
    });
  const mega = choice => {
    const label = opts.find(([c]) => c === choice)[1];
    const evolved = opts.find(([, l]) => l === `${label}:mega`);
    return evolved ? evolved[0] : choice;
  };
  if (!moves.length) return bestSwitch() || opts[0][0];
  const best = moves.reduce((a, b) => (b.damage > a.damage ? b : a));
  const incoming = bestDamage(battle, foe, me);
  const kos = moves.filter(m => m.damage >= foe.hp);
  if (kos.length) return mega(kos.reduce((a, b) => (b.move.priority > a.move.priority ? b : a)).choice);
  const heal = moves.find(m => (m.move.heal || m.move.flags.heal) && m.move.target === 'self');
  if (heal && me.hp < me.maxhp / 2 && incoming < me.maxhp / 2) return mega(heal.choice);
  if (best.damage <= 0) {  // nothing here hurts the foe: pass the boosts on or bring in an attacker
    const pass = moves.find(m => m.move.id === 'batonpass');
    const target = bestSwitch();
    if (pass && target && Object.values(me.boosts).some(v => v > 0)) return mega(pass.choice);
    if (target) return target;
  }
  const setup = moves.find(m => {
    const boosts = selfBoosts(m.move);
    return boosts && Object.entries(boosts).some(([stat, n]) => n > 0 && me.boosts[stat] < 2);
  });
  const safe = incoming * 2 < me.hp;
  if (setup && (best.damage <= 0 || (safe && best.damage * 2 < foe.hp))) return mega(setup.choice);
  if (best.damage <= 0) return mega((setup || moves[0]).choice);
  return mega(best.choice);
}

const CONTINUATIONS = {greedy, tactical};

// A remaining Pokemon with a damaging move (PP left) that some remaining foe is not immune to by type.
function hasDamageSource(battle, side) {
  const foes = side.foe.pokemon.filter(mon => !mon.fainted);
  return side.pokemon.some(mon => !mon.fainted && mon.moveSlots.some(slot => {
    if (slot.pp <= 0) return false;
    const move = battle.dex.moves.get(slot.id);
    return move.category !== 'Status' && foes.some(foe => battle.dex.getImmunity(move.type, foe.getTypes()));
  }));
}

function lostPath(battle) {
  if (battle.ended) return 0;
  const ours = hasDamageSource(battle, battle.p1), theirs = hasDamageSource(battle, battle.p2);
  return ours === theirs ? 0 : ours ? 1 : -1;
}

function material(side) {
  return side.pokemon.reduce((acc, mon) => acc + (mon.fainted ? 0 : mon.hp / mon.maxhp), 0);
}

function score(battle, startTurn, winDecay = WIN_DECAY, scoring = 'legacy', lostPathPenalty = 0) {
  const balance = material(battle.p1) - material(battle.p2);
  if (scoring === 'terminal') {
    if (battle.ended) {  // a quicker end breaks ties between wins (or losses) but never reaches a position's score
      const sooner = Math.max(TERMINAL_FLOOR, 1 - TERMINAL_TIEBREAK * (battle.turn - startTurn));
      return battle.winner === 'p1' ? sooner : battle.winner === 'p2' ? -sooner : 0;
    }
    const v = Math.max(-0.5, Math.min(0.5, balance / 6)) + lostPathPenalty * lostPath(battle);
    return Math.max(-TERMINAL_OPEN, Math.min(TERMINAL_OPEN, v));
  }
  let v = balance + lostPathPenalty * lostPath(battle);
  if (battle.ended) {
    const bonus = WIN_BONUS * Math.max(0.5, 1 - winDecay * (battle.turn - startTurn));
    v += battle.winner === 'p1' ? bonus : battle.winner === 'p2' ? -bonus : 0;
  }
  return v;
}

function step(battle, c1, c2) {
  if (battle.ended) return;
  const before = [battle.turn, battle.requestState, battle.log.length, battle.queue.list.length];
  let submitted = false;
  const pending = [[battle.p1, c1], [battle.p2, c2]].filter(([side]) => side.activeRequest && !side.activeRequest.wait);
  for (const [side, c] of pending) {
    if (c === 'pass' || !battle.choose(side.id, c)) {
      throw new Error(`Choice rejected: ${side.id} ${c}: ${side.choice.error}`);
    }
    submitted = true;
  }
  const after = [battle.turn, battle.requestState, battle.log.length, battle.queue.list.length];
  if (!submitted || before.every((v, i) => v === after[i])) throw new Error('Branch made no progress');
}

// A position's evaluation in passes: pass k plays every pair once with seed floor(k / continuations) and the k-th
// continuation policy, so stopping between passes keeps every pair on the same runs.
function prepare(sample, format, seeds, rollout, winDecay, options = {}) {
  const scoring = options.scoring || 'legacy', penalty = options.lostPathPenalty || 0;
  const names = options.continuations || ['greedy'];
  for (const name of names) if (!CONTINUATIONS[name]) throw new Error(`unknown continuation ${name}`);
  const base = setUp(sample, format, 1);
  const st = {base, names, seeds, rollout, winDecay, scoring, penalty, passes: seeds * names.length, done: 0};
  if (base.ended) return st;
  st.p1 = choices(base, base.p1);
  st.p2 = choices(base, base.p2);
  st.snapshot = JSON.stringify(State.serializeBattle(base));
  const grid = () => st.p1.map(() => st.p2.map(() => 0));
  st.total = grid(); st.invalid = grid(); st.ended = grid();
  st.per = Object.fromEntries(names.map(n => [n, {total: grid(), runs: grid()}]));
  return st;
}

function runPass(st) {
  if (st.base.ended || st.done >= st.passes) return;
  const k = st.done, s = Math.floor(k / st.names.length), name = st.names[k % st.names.length];
  const policy = CONTINUATIONS[name];
  for (let i = 0; i < st.p1.length; i++) {
    for (let j = 0; j < st.p2.length; j++) {
      const battle = State.deserializeBattle(st.snapshot);
      battle.resetRNG([s * 4 + 11, s * 4 + 12, s * 4 + 13, s * 4 + 14]);
      try {
        step(battle, st.p1[i][0], st.p2[j][0]);
        let turns = 0;
        for (let guard = 0; !battle.ended && turns < st.rollout && guard < st.rollout * 3; guard++) {
          if (battle.requestState === 'move') turns++;
          step(battle, policy(battle, battle.p1), policy(battle, battle.p2));
        }
        if (!battle.ended && turns < st.rollout) throw new Error('Rollout did not finish');
      } catch (e) {
        st.invalid[i][j]++;  // an illegal pairing (e.g. a sampled move the request rejects): no result
        continue;
      }
      const v = score(battle, st.base.turn, st.winDecay, st.scoring, st.penalty);
      st.total[i][j] += v;
      st.ended[i][j] += battle.ended ? 1 : 0;
      st.per[name].total[i][j] += v;
      st.per[name].runs[i][j]++;
    }
  }
  st.done++;
}

function finish(st) {
  if (st.base.ended) {
    const v = score(st.base, st.base.turn, st.winDecay, st.scoring, st.penalty);
    return {p1: ['pass'], p2: ['pass'], payoff: [[v]], invalid: [[0]], resolved: [[1]],
            byContinuation: Object.fromEntries(st.names.map(n => [n, [[v]]])), passes: [1, 1]};
  }
  const mean = (t, n) => t.map((row, i) => row.map((v, j) => n(i, j) > 0 ? v / n(i, j) : null));
  const ran = (i, j) => st.done - st.invalid[i][j];
  const used = st.names.slice(0, Math.min(st.done, st.names.length));
  return {
    p1: st.p1.map(x => x[1]), p2: st.p2.map(x => x[1]), payoff: mean(st.total, ran), invalid: st.invalid,
    resolved: mean(st.ended, ran),
    byContinuation: Object.fromEntries(used.map(n => [n, mean(st.per[n].total, (i, j) => st.per[n].runs[i][j])])),
    passes: [st.done, st.passes],
  };
}

function evaluate(sample, format, seeds, rollout, winDecay, options = {}) {
  const st = prepare(sample, format, seeds, rollout, winDecay, options);
  while (!st.base.ended && st.done < st.passes) runPass(st);
  return finish(st);
}

// Every sample's first pass always runs; later passes start only before the deadline (deadlineMs after the request
// arrived), all samples together, so a cut keeps the samples comparable.
function evaluateAll(req) {
  const options = {continuations: req.continuations, scoring: req.scoring, lostPathPenalty: req.lostPathPenalty};
  const start = Date.now();
  const states = req.samples.map(s => prepare(s, req.format, req.seeds ?? 2, req.rollout ?? 3,
                                              req.winDecay ?? WIN_DECAY, options));
  const passes = Math.max(0, ...states.map(st => st.passes));
  for (let k = 0; k < passes; k++) {
    if (k > 0 && req.deadlineMs && Date.now() - start > req.deadlineMs) break;
    for (const st of states) runPass(st);
  }
  return {results: states.map(finish), elapsedMs: Date.now() - start};
}

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', line => {
  let reply;
  try {
    const req = JSON.parse(line);
    reply = evaluateAll(req);
  } catch (e) {
    reply = { error: String(e && e.stack || e) };
  }
  process.stdout.write(JSON.stringify(reply) + '\n');
});
