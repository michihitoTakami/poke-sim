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
//     switches when the active cannot damage the foe; disrupt = tactical plus haze / phazing / Encore / Taunt /
//     Toxic / Yawn / Will-O-Wisp against set-up Pokemon; expert = tactical plus the multi-turn Baton Pass and set-up
//     plans of a strong player (see expert()). "a/b" plays p1 (us) on with a and p2 with b; "a" is "a/a".
//   scoring: legacy = HP balance (each Pokemon 0..1) + 3 for a win (less the slower it comes);
//     terminal = a real end is +-1 (0 for a tie), less 0.005 per turn it took (a tie-break among wins, at least
//     0.95; "tiebreak": 0 scores every win +1 and every loss -1); a position still going is the HP balance scaled into (-0.5, 0.5) plus the lost-path term, kept
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
// "deadlineMs" (optional): no new cycle (all continuation policies, one run per pair each) starts after this
// soft budget; the first complete cycle always runs so a timeout cannot silently remove the tactical policy.
const path = require('path');
const readline = require('readline');
const dir = process.argv[2];
const { Battle, Teams } = require(path.resolve(dir, 'dist/sim'));
const { State } = require(path.resolve(dir, 'dist/sim/state'));
const {extractChannelMessages} = require(path.resolve(dir, 'dist/sim/battle'));

const WIN_BONUS = 3;
const WIN_DECAY = 0.05;  // per turn: a quicker win scores higher
// terminal scoring: a win is 1 less TERMINAL_TIEBREAK per turn it takes (at least TERMINAL_FLOOR); a position still
// going stays within +-TERMINAL_OPEN, below any win and above any loss
const TERMINAL_TIEBREAK = 0.005, TERMINAL_FLOOR = 0.95, TERMINAL_OPEN = 0.9;

const {setUp} = require(path.resolve(__dirname, 'state_codec.js')).createCodec(dir);

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

// The simulator's damage for one hit of moveId, without a critical hit, as a move is used (the active attacker /
// target / move set, so abilities such as Unaware apply). roll: 0 the highest of the 16 random factors .. 15 the
// lowest; null the mean of the highest and the lowest. A query must not change the battle: getDamage draws the damage
// roll (and a critical hit) from the battle's PRNG, so the PRNG and the active move are saved and restored, and a
// policy thinking about its options does not change the random numbers the rest of the branch plays with.
function damageRoll(battle, source, target, moveId, roll = null) {
  const move = battle.dex.getActiveMove(moveId);
  if (move.category === 'Status') return 0;
  const seed = battle.prng.getSeed();
  const saved = [battle.randomizer, battle.activeMove, battle.activePokemon, battle.activeTarget];
  const tr = battle.trunc;
  const at = r => d => tr(tr(d * (100 - r)) / 100);
  try {
    battle.activeMove = move; battle.activePokemon = source; battle.activeTarget = target;
    move.willCrit = false;
    const hit = r => {
      battle.randomizer = at(r);
      const dmg = battle.actions.getDamage(source, target, move, true);
      return typeof dmg === 'number' ? dmg : 0;
    };
    return roll === null ? (hit(0) + hit(15)) / 2 : hit(roll);
  } catch (e) {
    return 0;
  } finally {
    [battle.randomizer, battle.activeMove, battle.activePokemon, battle.activeTarget] = saved;
    battle.prng.setSeed(seed);
  }
}

function expectedDamage(battle, source, target, moveId) {
  const move = battle.dex.moves.get(moveId);
  if (move.category === 'Status') return 0;
  const hits = Array.isArray(move.multihit) ? (move.multihit[0] === 2 && move.multihit[1] === 5 ? 3.1 : move.multihit[0])
    : (move.multihit || 1);
  return damageRoll(battle, source, target, moveId) * hits * (move.accuracy === true ? 1 : move.accuracy / 100);
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
function tactical(battle, side, healBelow = 0.5) {
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
  if (heal && me.hp < me.maxhp * healBelow && incoming < me.maxhp / 2) return mega(heal.choice);
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


// --- expert / disrupt: multi-turn plans a strong player follows, as continuation policies -----------------------
// Both read the sampled world (the foe's sampled set included), like greedy / tactical: across worlds this models
// "the foe may have Yawn" by the share of worlds where it has it.

const STATUS_MOVES = {toxic: 'tox', willowisp: 'brn', thunderwave: 'par', glare: 'par', nuzzle: 'par',
                      spore: 'slp', sleeppowder: 'slp', hypnosis: 'slp'};
const PHAZE = new Set(['roar', 'whirlwind', 'dragontail', 'circlethrow']);
const RESET = new Set(['haze', 'clearsmog']);
const BLOCKS_STATUS_MOVES = new Set(['goodasgold', 'magicbounce']);

const knows = (mon, ids) => mon.moveSlots.some(s => s.pp > 0 && ids.has(s.id));
const boostSum = mon => Object.values(mon.boosts).reduce((a, v) => a + Math.max(0, v), 0);
const stage = s => (s >= 0 ? (2 + s) / 2 : 2 / (2 - s));

// Speed as it orders moves: higher moves first. Under Trick Room it is inverted for every Pokemon alike (as
// getActionSpeed does for the active), so bench and active Pokemon compare correctly either way.
function rawSpeed(mon, speBoost = mon.boosts ? mon.boosts.spe : 0) {
  if (mon.isActive && speBoost === mon.boosts.spe) return mon.getStat('spe', false, false);  // items, paralysis, Tailwind
  let v = mon.storedStats.spe * stage(speBoost);
  if (mon.item === 'choicescarf') v *= 1.5;
  if (mon.status === 'par') v *= 0.5;
  return v;
}

function speedOf(mon, speBoost = mon.boosts ? mon.boosts.spe : 0) {
  const v = rawSpeed(mon, speBoost);
  return mon.battle.field.getPseudoWeather('trickroom') ? 1e4 - v : v;
}

// Does a status move have its effect on the target now (not the accuracy roll)?
function statusWorks(battle, source, target, move) {
  const ability = target.isActive && target.ignoringAbility() ? '' : target.ability;
  if (move.target !== 'self' && BLOCKS_STATUS_MOVES.has(ability)) return false;
  if (target.volatiles.substitute && !move.flags.bypasssub && move.target !== 'self') return false;
  const status = STATUS_MOVES[move.id];
  if (status) {
    if (target.status) return false;
    if (move.id === 'thunderwave' && !battle.dex.getImmunity('Electric', target)) return false;
    if (status === 'tox' && source.ability === 'corrosion') return !target.side.sideConditions.safeguard;
    return target.runStatusImmunity(status === 'tox' ? 'psn' : status);
  }
  if (move.id === 'yawn') {
    return !target.status && !target.volatiles.yawn && target.runStatusImmunity('slp') &&
      !['insomnia', 'vitalspirit', 'sweetveil', 'comatose', 'purifyingsalt'].includes(ability);
  }
  if (move.id === 'encore') {
    const last = target.lastMove;
    return !!last && !target.volatiles.encore && last.id !== 'struggle' && ability !== 'aromaveil';
  }
  if (move.id === 'taunt') return !target.volatiles.taunt && ability !== 'oblivious' && ability !== 'aromaveil';
  if (move.id === 'roar' || move.id === 'whirlwind') {
    return target.side.pokemon.some(m => !m.fainted && !m.isActive) && ability !== 'suctioncups' &&
      !(move.id === 'roar' && ability === 'soundproof') && !target.volatiles.ingrain;
  }
  return true;
}

// A foe that sets up or passes boosts on: the targets of disruption.
function isSetupThreat(mon) {
  return boostSum(mon) >= 2 || mon.moveSlots.some(s => {
    const move = mon.battle.dex.moves.get(s.id);
    return move.id === 'batonpass' || (move.target === 'self' && move.boosts &&
      Object.values(move.boosts).some(n => n > 0));
  });
}

// Disrupt: tactical, but uses the status and anti-setup moves it has: Trick Room against a faster foe, haze / phaze
// a boosted foe, Encore a foe that just used a status move, Taunt a set-up Pokemon, then Toxic / Will-O-Wisp / Yawn /
// paralysis when its attacks do little. Stands for the opponents that beat a setup / Baton Pass team.
function disrupt(battle, side) {
  const opts = choices(battle, side);
  if (opts.length === 1 || side.activeRequest.forceSwitch) return tactical(battle, side);
  const foe = side.foe.active[0], me = side.active[0];
  if (!foe || foe.fainted) return tactical(battle, side);
  const moves = opts.filter(([, l]) => l.startsWith('move:') && !l.endsWith(':mega'))
    .map(([choice, label]) => ({choice, move: battle.dex.moves.get(label.split(':')[1])}));
  const pick = pred => moves.find(m => pred(m.move) && statusWorks(battle, me, foe, m.move));
  const best = Math.max(0, ...moves.map(m => expectedDamage(battle, me, foe, m.move.id)));
  if (best >= foe.hp) return tactical(battle, side);  // a KO comes first
  const threat = isSetupThreat(foe), boosted = boostSum(foe) >= 2;
  let m;
  // Trick Room when the foe outspeeds us (it also turns a Speed-boosted Baton Pass against the receiver)
  if (!battle.field.getPseudoWeather('trickroom') && rawSpeed(foe) > rawSpeed(me) &&
      (m = moves.find(x => x.move.id === 'trickroom'))) return m.choice;
  if (boosted && (m = pick(mv => RESET.has(mv.id) || PHAZE.has(mv.id)))) return m.choice;
  if (threat && foe.lastMove && foe.lastMove.category === 'Status' && (m = pick(mv => mv.id === 'encore'))) {
    return m.choice;
  }
  if (threat && (m = pick(mv => mv.id === 'taunt'))) return m.choice;
  const weak = best * 3 < foe.hp;
  if ((threat || weak) && (m = pick(mv => mv.id === 'yawn' || STATUS_MOVES[mv.id]))) {
    if (m.move.id !== 'willowisp' || physicalShare(battle, foe, me) > 0.5) return m.choice;
  }
  return tactical(battle, side);
}

// How much of the foe's damage to us is physical (by the best physical vs the best special move).
function physicalShare(battle, foe, me) {
  let phys = 0, spec = 0;
  for (const s of foe.moveSlots) {
    const move = battle.dex.moves.get(s.id);
    if (move.category === 'Status' || s.pp <= 0) continue;
    const d = expectedDamage(battle, foe, me, move.id);
    if (move.category === 'Physical') phys = Math.max(phys, d); else spec = Math.max(spec, d);
  }
  return phys + spec > 0 ? phys / (phys + spec) : 0.5;
}

// Who receives a Baton Pass (or a switch while boosts are not being passed): the user's principles first
// (2: Yawn / Roar / Whirlwind -> a Good as Gold Pokemon; 3: Dragon Tail -> a Fairy; 7: a Swords Dance / attack
// boosts -> Unaware), then the bench Pokemon that hits the foe hardest and moves first after the pass.
function receiver(battle, side, passing) {
  const foe = side.foe.active[0];
  const bench = side.pokemon.filter(m => !m.fainted && !m.isActive);
  if (!bench.length) return null;
  const me = side.active[0];
  const spe = passing && me ? me.boosts.spe : 0;
  const foeSpeed = foe ? speedOf(foe) : 0;
  // a Pokemon that cannot damage the foe only receives when nothing else can (no passing back and forth)
  const hurts = foe && !foe.fainted ? bench.filter(mon => bestDamage(battle, mon, foe) > 0) : [];
  const scored = (hurts.length ? hurts : bench).map(mon => {
    let v = mon.hp / mon.maxhp;
    if (foe && !foe.fainted) {
      const dmg = bestDamage(battle, mon, foe);
      v += 2 * Math.min(1, dmg / Math.max(1, foe.hp)) + (dmg > 0 ? 0.5 : 0);
      if (speedOf(mon, spe) > foeSpeed) v += 0.5;
      if (knows(foe, new Set(['yawn', 'roar', 'whirlwind'])) && mon.ability === 'goodasgold') v += 2;
      if (knows(foe, new Set(['dragontail'])) && mon.hasType('Fairy')) v += 2;
      if (mon.ability === 'unaware') {  // a foe already boosted is the case the principle is for
        if (foe.boosts.atk > 0 || foe.boosts.spa > 0) v += 3.5;
        else if (knows(foe, new Set(['swordsdance', 'dragondance', 'nastyplot', 'bulkup', 'calmmind', 'quiverdance',
          'shellsmash']))) v += 2;
      }
    }
    return {mon, v};
  });
  const top = scored.reduce((a, b) => (b.v > a.v ? b : a));
  return `switch ${side.pokemon.indexOf(top.mon) + 1}`;
}

// Expert: tactical plus the plans of the team's player (docs/winning-ai-roadmap.md, 8.2).
//  A passer (knows Baton Pass) builds and passes boosts: Substitute when the foe cannot break it (5); against a
//  physical foe it stacks the defence boost to +6 (1); against a special foe it stalls with Protect / Substitute
//  until the receiver outspeeds, then passes (6); Encore / phazing / Yawn on the foe cut the plan short at +2 (2-4);
//  it passes before the foe's attack would knock it out. Against a foe slower than every bench attacker it does
//  not stay to pass (8).
//  Others set up further than tactical (to +6 while safe, not into Encore) and the receiver is chosen by threat.
// A plan (see policy()): cap, the boost to stack to (instead of +6, or +2 when the foe can cut it short);
// heal, the share of HP below which it uses a recovery move (tactical's 1/2 by default).
function makeExpert({cap: planCap = null, heal = 0.5} = {}) {
  const tact = (battle, side) => tactical(battle, side, heal);
  return function expert(battle, side) {
    const opts = choices(battle, side);
    const foe = side.foe.active[0], me = side.active[0];
    if (side.activeRequest.forceSwitch) {
      if (!foe || foe.fainted) return tact(battle, side);
      const passing = me && !me.fainted && me.lastMove && me.lastMove.id === 'batonpass';
      return receiver(battle, side, passing) || tact(battle, side);
    }
    if (opts.length === 1 || !foe || foe.fainted || !me) return tact(battle, side);
    const moves = opts.filter(([, l]) => l.startsWith('move:') && !l.endsWith(':mega'))
      .map(([choice, label]) => ({choice, move: battle.dex.moves.get(label.split(':')[1])}));
    const byId = id => moves.find(m => m.move.id === id);
    const best = Math.max(0, ...moves.map(m => expectedDamage(battle, me, foe, m.move.id)));
    if (best >= foe.hp) return tact(battle, side);
    const incoming = bestDamage(battle, foe, me);
    const worstHit = Math.max(0, ...foe.moveSlots.filter(s => s.pp > 0)
      .map(s => damageRoll(battle, foe, me, s.id, 0) * (battle.dex.moves.get(s.id).multihit ? 2 : 1)));
    const encore = knows(foe, new Set(['encore'])), stops = knows(foe, new Set(['yawn', 'roar', 'whirlwind',
      'dragontail', 'haze', 'clearsmog', 'taunt', 'encore']));
    const lastProtect = me.volatiles.stall && me.lastMove && ['protect', 'detect'].includes(me.lastMove.id);
    const subHp = Math.floor(me.maxhp / 4);
    const setups = moves.filter(m => m.move.target === 'self' && m.move.boosts &&
      Object.entries(m.move.boosts).some(([st, n]) => n > 0 && me.boosts[st] < 6));
    const pass = byId('batonpass');
    if (pass) {
      const bench = side.pokemon.filter(m => !m.fainted && !m.isActive);
      const attackers = bench.filter(m => bestDamage(battle, m, foe) > 0);
      const foeSpeed = speedOf(foe);
      const passTo = receiver(battle, side, true);
      const gained = boostSum(me);
      // passing is worth a turn only with something to hand over (+2 in all, or a Substitute) to an attacker
      const receiverHurts = passTo && bestDamage(battle, side.pokemon[Number(passTo.split(' ')[1]) - 1], foe) > 0;
      const canPass = passTo && receiverHurts && (gained >= 2 || (gained > 0 && !!me.volatiles.substitute));
      // 8: the foe is slower than our attackers already: no need to build up
      const slowest = Math.min(...attackers.map(m => speedOf(m)));
      if (planCap === null && !gained && attackers.length &&  // a plan with its own cap is chosen by the search
          foe.side.pokemon.every(m => m.fainted || speedOf(m) < slowest)) {
        return tact(battle, side);
      }
      if (passTo && gained > 0 && incoming >= me.hp && !me.volatiles.substitute) {
        return pass.choice;  // about to fall: whatever it has
      }
      const sub = byId('substitute');
      if (sub && !me.volatiles.substitute && me.hp > subHp && worstHit < subHp && incoming < me.hp - subHp) {
        return sub.choice;  // 5
      }
      const physical = physicalShare(battle, foe, me) > 0.5;
      const stat = physical ? 'def' : 'spd';
      const wall = setups.find(m => (m.move.boosts[stat] || 0) > 0);
      // 2-4: a plan the foe can cut short is passed early; a plan's own cap replaces that judgement
      const cap = planCap ?? (encore || stops ? 2 : 6);
      if (wall && me.boosts[stat] < cap && incoming * 2 < me.hp) return wall.choice;  // 1, 4
      if (!physical) {  // 6: stall for Speed until the receiver moves first
        const target = passTo ? side.pokemon[Number(passTo.split(' ')[1]) - 1] : null;
        const fastEnough = target && speedOf(target, me.boosts.spe) > foeSpeed;
        if (!fastEnough && me.boosts.spe < 6) {
          const protect = byId('protect') || byId('detect');
          if (protect && !lastProtect) return protect.choice;
        }
      }
      if (canPass) return pass.choice;
      if (gained > 0 && gained < 2) {  // not yet worth passing: one more turn behind Protect (Speed Boost)
        const protect = byId('protect') || byId('detect');
        if (protect && !lastProtect) return protect.choice;
      }
      return tact(battle, side);
    }
    // not a passer: set up further than tactical while it is safe and the foe cannot Encore it
    const setup = setups.find(m => m.move.id !== 'substitute');
    const capped = planCap !== null && Object.entries(setup ? setup.move.boosts : {}).every(([st, n]) => n <= 0 ||
      me.boosts[st] >= planCap);
    if (setup && !capped && !encore && incoming * 3 < me.hp && best * 2 < foe.hp) return setup.choice;
    return tact(battle, side);
  };
}

const expert = makeExpert();

const CONTINUATIONS = {greedy, tactical, disrupt, expert};

// A continuation by name: those above, or an expert plan "expert@<cap>" / "expert@<cap>h<heal %>" / "experth<heal %>"
// (e.g. expert@4h33: stack to +4, recover below a third of its HP).
function policy(name) {
  if (CONTINUATIONS[name]) return CONTINUATIONS[name];
  const m = /^expert(?:@([0-6]))?(?:h(\d{1,2}))?$/.exec(name);
  if (!m) throw new Error(`unknown continuation ${name}`);
  CONTINUATIONS[name] = makeExpert({cap: m[1] === undefined ? null : Number(m[1]),
                                    heal: m[2] === undefined ? 0.5 : Number(m[2]) / 100});
  return CONTINUATIONS[name];
}

// Damage-source possibility, not a KO prediction. Include type-changing and immunity abilities in sampled worlds.
function canHurt(battle, source, target, moveId) {
  const move = battle.dex.getActiveMove(moveId);
  if (move.category === 'Status') return false;
  // Bench capabilities describe the Pokemon after entering, rather than its currently inactive ability.
  const ability = mon => mon.isActive && mon.ignoringAbility() ? '' : mon.ability;
  const attackAbility = ability(source), defendAbility = ability(target);
  const changedType = {aerilate: 'Flying', pixilate: 'Fairy', refrigerate: 'Ice', galvanize: 'Electric'};
  if (move.type === 'Normal' && changedType[attackAbility]) move.type = changedType[attackAbility];
  if (attackAbility === 'normalize') move.type = 'Normal';
  if (attackAbility === 'liquidvoice' && move.flags.sound) move.type = 'Water';
  let types = target.getTypes();
  if (['Normal', 'Fighting'].includes(move.type) && ['scrappy', 'mindseye'].includes(attackAbility)) {
    types = types.filter(t => t !== 'Ghost');
  }
  if (!battle.dex.getImmunity(move.type, types)) return false;
  if (move.type === 'Ground' && (target.isActive ? target.hasItem('airballoon') : target.item === 'airballoon')) return false;
  if (['moldbreaker', 'teravolt', 'turboblaze'].includes(attackAbility)) return true;
  const immune = {Ground: ['levitate', 'eartheater'], Fire: ['flashfire', 'wellbakedbody'],
    Water: ['waterabsorb', 'stormdrain', 'dryskin'], Electric: ['voltabsorb', 'lightningrod', 'motordrive'],
    Grass: ['sapsipper']};
  if ((immune[move.type] || []).includes(defendAbility)) return false;
  if ((move.flags.sound && defendAbility === 'soundproof') ||
      (move.flags.bullet && defendAbility === 'bulletproof') ||
      (move.flags.wind && defendAbility === 'windrider')) return false;
  if (defendAbility === 'wonderguard' && battle.dex.getEffectiveness(move.type, types) <= 0) return false;
  return true;
}

function hasDamageSource(battle, side) {
  const foes = side.foe.pokemon.filter(mon => !mon.fainted);
  return side.pokemon.some(mon => !mon.fainted && mon.moveSlots.some(slot => {
    if (slot.pp <= 0) return false;
    return foes.some(foe => canHurt(battle, mon, foe, slot.id));
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

function score(battle, startTurn, winDecay = WIN_DECAY, scoring = 'legacy', lostPathPenalty = 0,
               tiebreak = TERMINAL_TIEBREAK) {
  const balance = material(battle.p1) - material(battle.p2);
  if (scoring === 'terminal') {
    if (battle.ended) {  // a quicker end breaks ties between wins (or losses) but never reaches a position's score
      const sooner = Math.max(TERMINAL_FLOOR, 1 - tiebreak * (battle.turn - startTurn));
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
  const tiebreak = options.tiebreak ?? TERMINAL_TIEBREAK;
  const names = options.continuations || ['greedy'];
  for (const name of names) for (const part of name.split('/')) {
    policy(part);
  }
  const base = setUp(sample, format, 1);
  const st = {base, names, seeds, rollout, winDecay, scoring, penalty, tiebreak, passes: seeds * names.length,
              done: 0};
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
  const [mine, theirs = mine] = name.split('/');  // "a/b": p1 (us) plays on with a, p2 with b
  const p1Policy = policy(mine), p2Policy = policy(theirs);
  for (let i = 0; i < st.p1.length; i++) {
    for (let j = 0; j < st.p2.length; j++) {
      const battle = State.deserializeBattle(st.snapshot);
      battle.resetRNG([s * 4 + 11, s * 4 + 12, s * 4 + 13, s * 4 + 14]);
      try {
        step(battle, st.p1[i][0], st.p2[j][0]);
        let turns = 0;
        for (let guard = 0; !battle.ended && turns < st.rollout && guard < st.rollout * 3; guard++) {
          if (battle.requestState === 'move') turns++;
          step(battle, p1Policy(battle, battle.p1), p2Policy(battle, battle.p2));
        }
        if (!battle.ended && turns < st.rollout) throw new Error('Rollout did not finish');
      } catch (e) {
        st.invalid[i][j]++;  // an illegal pairing (e.g. a sampled move the request rejects): no result
        continue;
      }
      const v = score(battle, st.base.turn, st.winDecay, st.scoring, st.penalty, st.tiebreak);
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
    const v = score(st.base, st.base.turn, st.winDecay, st.scoring, st.penalty, st.tiebreak);
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

// A cycle includes all policies and samples. Stop only between complete cycles so the estimator stays balanced.
function evaluateAll(req) {
  const options = {continuations: req.continuations, scoring: req.scoring, lostPathPenalty: req.lostPathPenalty,
                   tiebreak: req.tiebreak};
  const start = Date.now();
  const states = req.samples.map(s => prepare(s, req.format, req.seeds ?? 2, req.rollout ?? 3,
                                              req.winDecay ?? WIN_DECAY, options));
  const cycles = Math.max(0, ...states.map(st => st.seeds));
  for (let k = 0; k < cycles; k++) {
    if (k > 0 && req.deadlineMs && Date.now() - start > req.deadlineMs) break;
    for (let policy = 0; policy < (options.continuations || ['greedy']).length; policy++) {
      for (const st of states) runPass(st);
    }
  }
  return {results: states.map(finish), elapsedMs: Date.now() - start};
}

// Plan search: which of our plans (expert continuations, e.g. how far to stack) wins most from this position.
// Each plan is played to the real end of the battle (or maxTurns) in every sample, against each opponent policy,
// with the same seeds for every plan (common random numbers). A cycle is one seed over every sample, opponent and
// plan; no new cycle starts after deadlineMs (the first always runs). A game still going at maxTurns, or one that
// failed, is counted as such: score = (wins - losses) / (games that ran), a game still going counting 0.
//   request: {"format", "samples": [...], "plans": {"candidates": [names], "opponents": [names] (default
//             ["tactical", "disrupt"]), "seeds": n (default 2), "maxTurns": (default 60), "deadlineMs"}}
//   reply:   {"plans": {name: {"win", "loss", "draw", "unfinished", "failed", "score", "perSample": [score|null]}},
//             "cycles": [done, planned], "elapsedMs"}
function playOut(snapshot, p1Policy, p2Policy, seed, maxTurns) {
  const battle = State.deserializeBattle(snapshot);
  battle.resetRNG([seed * 4 + 11, seed * 4 + 12, seed * 4 + 13, seed * 4 + 14]);
  const start = battle.turn;
  for (let guard = 0; !battle.ended && battle.turn - start < maxTurns && guard < maxTurns * 4; guard++) {
    step(battle, p1Policy(battle, battle.p1), p2Policy(battle, battle.p2));
  }
  if (!battle.ended) return 'unfinished';
  return battle.winner === 'p1' ? 'win' : battle.winner === 'p2' ? 'loss' : 'draw';
}

function evaluatePlans(req) {
  const cfg = req.plans, start = Date.now();
  const names = cfg.candidates, opponents = cfg.opponents || ['tactical', 'disrupt'];
  const seeds = cfg.seeds ?? 2, maxTurns = cfg.maxTurns ?? 60;
  for (const n of [...names, ...opponents]) policy(n);
  const snapshots = req.samples.map(sample => {
    const base = setUp(sample, req.format, 1);
    return base.ended ? null : JSON.stringify(State.serializeBattle(base));
  });
  const blank = () => ({win: 0, loss: 0, draw: 0, unfinished: 0, failed: 0});
  const tally = Object.fromEntries(names.map(n => [n, {all: blank(), per: snapshots.map(blank)}]));
  let done = 0;
  for (let s = 0; s < seeds; s++) {
    if (s > 0 && cfg.deadlineMs && Date.now() - start > cfg.deadlineMs) break;
    snapshots.forEach((snap, k) => {
      if (!snap) return;
      for (const opp of opponents) {
        for (const name of names) {
          let r;
          try {
            r = playOut(snap, policy(name), policy(opp), s, maxTurns);
          } catch (e) {
            r = 'failed';
          }
          tally[name].all[r]++;
          tally[name].per[k][r]++;
        }
      }
    });
    done++;
  }
  const scoreOf = t => {  // a game still going counts 0 (neither side won yet), not left out
    const n = t.win + t.loss + t.draw + t.unfinished;
    return n ? (t.win - t.loss) / n : null;
  };
  const plans = Object.fromEntries(names.map(n => [n, {...tally[n].all, score: scoreOf(tally[n].all),
                                                         perSample: tally[n].per.map(scoreOf)}]));
  return {plans, cycles: [done, seeds], elapsedMs: Date.now() - start};
}

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', line => {
  let reply;
  try {
    const req = JSON.parse(line);
    reply = req.terminalSearch ? require('./terminal_search').terminalSearch(req,
      {setUp, choices, step, State, extractChannelMessages}) :
      req.valueLeaves ? require('./value_search').evaluateValue(req,
      {setUp, choices, step, score, State, extractChannelMessages}) :
      req.branches ? require('./branch_matches').evaluateBranches(req,
      {setUp, choices, step, policy, State, extractChannelMessages}) :
      req.plans ? evaluatePlans(req) : evaluateAll(req);
  } catch (e) {
    reply = { error: String(e && e.stack || e) };
  }
  process.stdout.write(JSON.stringify(reply) + '\n');
});
