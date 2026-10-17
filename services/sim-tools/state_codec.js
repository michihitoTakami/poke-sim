// Simulator reconstruction boundary. Explicit fields carry known/sample state; unknown counters stay sampled.
const path = require('path');

function createCodec(dir) {
  const { Battle, Teams } = require(path.resolve(dir, 'dist/sim'));
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
        if (s.status) {
          if (mon.isActive) mon.setStatus(s.status, null, null, true);
          else {
            // setStatus deliberately rejects inactive targets. This is restoration of an existing status,
            // not a new infliction: keep it on the bench, including its normal sleep/toxic initial state.
            const status = battle.dex.conditions.get(s.status);
            mon.status = status.id;
            mon.statusState = battle.initEffectState({id: status.id, target: mon});
            battle.singleEvent('Start', status, mon.statusState, mon, mon);
          }
        }
        // PP persists on the bench. Zero is meaningful; never restore it to a fresh full-PP set.
        for (const slot of mon.moveSlots) {
          if (s.ppUsed && s.ppUsed[slot.id] !== undefined) {
            const used = s.ppUsed[slot.id];
            if (!Number.isInteger(used) || used < 0) throw new Error('Invalid PP consumption');
            slot.pp = Math.max(0, slot.maxpp - used);
          }
          if (s.pp && s.pp[slot.id] !== undefined) {
            const pp = s.pp[slot.id];
            if (!Number.isInteger(pp) || pp < 0 || pp > slot.maxpp) throw new Error('Invalid PP');
            slot.pp = pp;
          }
        }
        if (s.status === 'tox' && s.toxicStage !== undefined) {
          mon.statusState.stage = Math.max(0, Math.min(15, s.toxicStage));
        }
        if (s.status === 'slp' && s.sleepRemaining !== undefined) {
          if (!Number.isInteger(s.sleepRemaining) || s.sleepRemaining < 1) throw new Error('Invalid sleep counter');
          mon.statusState.time = s.sleepRemaining;
        }
        if (i === 0) {
          // An ActiveMove, as the simulator itself records: the search snapshots worlds with State.serializeBattle,
          // which turns a plain dex Move into a '[DataMove:id]' string that the restore leaves as is, and any
          // later Encore / Disable / cantusetwice check on that lastMove throws or misreads.
          if (s.lastMove) mon.lastMove = battle.dex.getActiveMove(s.lastMove);
          mon.clearBoosts();
          Object.assign(mon.boosts, s.boosts || {});
          for (const v of s.volatiles || []) mon.addVolatile(v);
          if (s.perish !== undefined) {
            mon.addVolatile('perishsong');
            if (mon.volatiles['perishsong']) mon.volatiles['perishsong'].duration = Math.max(1, s.perish);
          }
          if (s.encore && mon.hasMove(s.encore)) {  // Encore locks the last move used
            const last = mon.lastMove;
            mon.lastMove = battle.dex.getActiveMove(s.encore);
            mon.addVolatile('encore');
            if (s.lastMove) mon.lastMove = last;
            if (s.encoreTurns !== undefined && mon.volatiles.encore) mon.volatiles.encore.duration = s.encoreTurns;
          }
          if (s.disabledMove && mon.hasMove(s.disabledMove)) {
            const last = mon.lastMove;
            mon.lastMove = battle.dex.getActiveMove(s.disabledMove);
            mon.addVolatile('disable');
            mon.lastMove = last;
            if (s.disableTurns !== undefined && mon.volatiles.disable) mon.volatiles.disable.duration = s.disableTurns;
          }
          if (s.choiceMove && mon.hasMove(s.choiceMove) && mon.getItem().isChoice) {
            // onStart needs activeMove; preserve simulator validation and avoid a hand-built condition object.
            const previous = battle.activeMove;
            battle.activeMove = battle.dex.getActiveMove(s.choiceMove);
            mon.addVolatile('choicelock');
            battle.activeMove = previous;
          }
          if (s.substituteHp !== undefined && mon.volatiles.substitute) {
            if (!Number.isInteger(s.substituteHp) || s.substituteHp < 1 || s.substituteHp > mon.maxhp) {
              throw new Error('Invalid substitute HP');
            }
            mon.volatiles.substitute.hp = s.substituteHp;
          }
          if (s.stall) {  // Protect used s.stall turns in a row: success chance 1 / 3^stall
            mon.addVolatile('stall');
            if (mon.volatiles['stall']) {
              mon.volatiles['stall'].counter = Math.min(729, 3 ** s.stall);
              mon.volatiles['stall'].duration = 1; // previous turn's residual already ran
            }
          }
          for (const [id, turns] of Object.entries(s.volatileTurns || {})) {
            if (!Number.isInteger(turns) || turns < 1) throw new Error('Invalid volatile duration');
            if (mon.volatiles[id]) mon.volatiles[id].duration = turns;
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
    const entryLogPosition = battle.log.length;
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
      // makeRequest reads cached disable flags; restoration does not pass through nextTurn's refresh.
      for (const side of battle.sides) for (const mon of side.active) {
        if (!mon || mon.fainted) continue;
        for (const slot of mon.moveSlots) { slot.disabled = false; slot.disabledSource = ''; }
        battle.runEvent('DisableMove', mon);
        for (const slot of mon.moveSlots) {
          const move = battle.dex.getActiveMove(slot.id);
          battle.singleEvent('DisableMove', move, null, mon);
          if (move.flags.cantusetwice && mon.lastMove?.id === slot.id) mon.disableMove(slot.id);
        }
      }
      battle.makeRequest(battle.sides.some(side => side.active.some(mon => mon.switchFlag)) ? 'switch' : 'move');
    }
    // Only real incoming-switch events may extend a public observer; the reconstructed setup log is private.
    battle.pokesimEntryLog = battle.log.slice(entryLogPosition);
    return battle;
  }

  return {setUp};
}
module.exports = {createCodec};
