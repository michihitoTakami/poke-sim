// Conservative root pruning across every sampled world and legal reply. Never change engine legality.
// A full-HP heal is useful if damage can arrive before it; a capped boost can matter after Haze/Intimidate.
const HEALS = new Set(['recover', 'roost', 'synthesis', 'moonlight', 'morningsun', 'milkdrink',
  'softboiled', 'shoreup', 'slackoff', 'healorder']);
const RESETS = new Set(['haze', 'clearsmog', 'spectralthief', 'topsyturvy', 'heartswap', 'powerswap', 'guardswap']);
const TYPE_CHANGES = new Set(['soak', 'magicpowder', 'reflecttype', 'transform', 'conversion', 'conversion2',
  'camouflage', 'burnup', 'doubleshock', 'roost', 'forestscurse', 'trickortreat']);
const ABILITY_CHANGES = new Set(['skillswap', 'gastroacid', 'entrainment', 'simplebeam', 'worryseed', 'roleplay', 'doodle']);
const CURES = new Set(['refresh', 'healbell', 'aromatherapy', 'junglehealing', 'lunarblessing', 'rest']);
const TYPE_ABILITIES = new Set(['aerilate', 'pixilate', 'refrigerate', 'galvanize', 'normalize', 'liquidvoice']);

function priority(b, source, target, move) {
  const active = b.dex.getActiveMove(move.id);
  const p = b.singleEvent('ModifyPriority', active, null, source, target, null, active.priority);
  return [b.runEvent('ModifyPriority', source, target, active, p),
    b.runEvent('FractionalPriority', source, null, active, 0)];
}

function canPrecede(b, ours, theirs, move, reply, mega = false) {
  const a = priority(b, ours, theirs, move), c = priority(b, theirs, ours, reply);
  if (a[0] !== c[0]) return c[0] > a[0];
  if (mega || theirs.hasItem('quickclaw') || theirs.hasItem('custapberry') || theirs.hasAbility('quickdraw')) return true;
  if (a[1] !== c[1]) return c[1] > a[1];
  return theirs.getActionSpeed() >= ours.getActionSpeed(); // a speed tie can go either way
}

function immune(b, source, target, move) {
  const bypass = ['moldbreaker', 'turboblaze', 'teravolt'].includes(source.ability) && !target.hasItem('abilityshield');
  if (move.category === 'Status') {
    if (!bypass && target.hasAbility('goodasgold')) return true;
    if (TYPE_ABILITIES.has(source.ability) || move.onModifyType) return false;
    const types = target.getTypes();
    if (move.status && target.status) return true;
    const status = move.status === 'tox' ? 'psn' : move.status;
    if (status && !b.dex.getImmunity(status, types) &&
        !(status === 'psn' && source.hasAbility('corrosion'))) return true;
    if (move.ignoreImmunity === false && !b.dex.getImmunity(move.type, types)) return true;
    if (move.flags.powder && (!b.dex.getImmunity('powder', types) || target.hasItem('safetygoggles') ||
        (!bypass && target.hasAbility('overcoat')))) return true;
    if (source.hasAbility('prankster') && types.includes('Dark')) return true;
    return false;
  }
  if (move.ignoreImmunity || TYPE_ABILITIES.has(source.ability) || move.onModifyType) return false;
  const types = target.getTypes();
  const scrappy = ['scrappy', 'mindseye'].includes(source.ability) || target.volatiles.foresight;
  if (!target.hasItem('ringtarget') && !b.dex.getImmunity(move.type, types)) {
    if (!(scrappy && ['Normal', 'Fighting'].includes(move.type)) &&
        !(move.type === 'Ground' && target.isGrounded())) return true;
  }
  if (bypass) return false;
  const abilities = {Fire: ['flashfire', 'wellbakedbody'], Water: ['waterabsorb', 'stormdrain', 'dryskin'],
    Electric: ['voltabsorb', 'lightningrod', 'motordrive'], Grass: ['sapsipper'], Ground: ['levitate']};
  if ((abilities[move.type] || []).some(a => target.hasAbility(a))) {
    if (move.type !== 'Ground' || !target.isGrounded()) return true;
  }
  return move.type === 'Ground' && target.hasItem('airballoon') && !target.isGrounded();
}

function reducedStats(move, stats) {
  if (RESETS.has(move.id)) return true;
  const effects = [move, move.secondary, ...(move.secondaries || [])].filter(Boolean);
  return effects.some(e => Object.entries(e.boosts || {}).some(([s, n]) => n < 0 && stats.includes(s)));
}

function reason(b, side, option, replies) {
  const [choice, label] = option;
  if (!label.startsWith('move:') || label.endsWith(':mega') || side.activeRequest?.forceSwitch) return null;
  const me = side.active[0], foe = side.foe.active[0], move = b.dex.moves.get(label.split(':')[1]);
  if (!me || !foe || me.fainted || foe.fainted) return null;
  if (move.id === 'substitute' && me.hp <= Math.floor(me.maxhp / 4)) {
    // Public HP proves failure unless a preceding reply can heal us. Damage-triggered berries and
    // abilities, Pain Split/Heal Pulse and custom onHit effects make this proof inconclusive.
    for (const [, otherLabel] of replies) {
      if (!otherLabel.startsWith('move:')) continue;
      const response = b.dex.moves.get(otherLabel.split(':')[1]);
      if (canPrecede(b, me, foe, move, response, otherLabel.endsWith(':mega')) &&
          (response.onHit || me.getItem().isBerry ||
           ['waterabsorb', 'dryskin', 'voltabsorb', 'eartheater'].includes(me.ability))) return null;
    }
    return 'insufficient-hp-for-substitute';
  }
  // These can create useful effects even when the primary move has no effect.
  // Roost's grounding effect is not applied if its heal fails at full HP (verified on the deployed engine).
  const fullRoost = move.id === 'roost' && me.hp === me.maxhp;
  if ((move.flags.sound && me.hasItem('throatspray')) || move.selfSwitch || move.selfdestruct ||
      move.flags.charge || (me.status === 'frz' && move.flags.defrost) || (move.self && !fullRoost)) return null;
  const heal = HEALS.has(move.id) && me.hp === me.maxhp;
  const gains = move.target === 'self' && move.boosts && Object.values(move.boosts).every(n => n > 0)
    && !move.volatileStatus && !move.onHit && !move.onTry && !move.onTryHit && !move.heal ? Object.keys(move.boosts) : [];
  const direction = me.hasAbility('contrary') ? -1 : 1;
  const capped = gains.length && gains.every(s => direction * me.boosts[s] >= 6);
  const attack = move.category !== 'Status' && !move.onHit && !move.onTryHit;
  const hostileStatus = move.category === 'Status' && ['normal', 'adjacentFoe', 'any'].includes(move.target);
  const blocked = attack || hostileStatus;
  if (!heal && !capped && !blocked) return null;
  for (const [other, otherLabel] of replies) {
    if (other.startsWith('switch ')) {
      const incoming = side.foe.pokemon[Number(other.split(' ')[1]) - 1];
      if (capped && gains.includes('atk') && incoming.ability === 'intimidate') return null;
      if (blocked && !immune(b, me, incoming, move)) return null;
      continue;
    }
    if (!otherLabel.startsWith('move:')) continue;
    const response = b.dex.moves.get(otherLabel.split(':')[1]), mega = otherLabel.endsWith(':mega');
    if (blocked && (mega || TYPE_CHANGES.has(response.id) || ABILITY_CHANGES.has(response.id) || CURES.has(response.id) ||
        !immune(b, me, foe, move))) return null;
    if (canPrecede(b, me, foe, move, response, mega)) {
      if (heal && ((response.category !== 'Status' && !immune(b, foe, me, response)) || response.onHit)) return null;
      if (capped && (reducedStats(response, gains) || response.onHit ||
          (mega && gains.includes('atk')))) return null;
    }
  }
  if (heal) return 'full-hp-heal-before-any-damage';
  if (capped) return 'boosts-already-capped';
  if (blocked) return 'immune-to-every-sampled-reply';
  return null;
}

function pruneRoot(states, deps) {
  const reasons = states[0].p1.map(() => []);
  let residualFinish = false;
  let suckerPunch = false;
  for (const st of states) {
    const b = deps.State.deserializeBattle(st.snapshot);
    try {
      st.p1.forEach((option, i) => reasons[i].push(reason(b, b.p1, option, st.p2)));
      const foe = b.p2.active[0];
      const residual = foe?.status === 'brn' ? foe.maxhp / 16 :
        foe?.status === 'psn' ? foe.maxhp / 8 :
          foe?.status === 'tox' ? foe.maxhp * ((foe.statusState.stage || 0) + 1) / 16 : 0;
      residualFinish ||= !!foe && (residual >= foe.hp || foe.volatiles.perishsong?.duration <= 1);
      suckerPunch ||= st.p2.some(([, label]) => label.split(':')[1] === 'suckerpunch');
    }
    finally { b.destroy(); }
  }
  const keep = reasons.map(row => !row.every(Boolean));
  // Waiting for a known residual KO can be useful, e.g. avoiding recoil/contact. Retain one passive option.
  let passiveWait = null;
  if (residualFinish || suckerPunch) {
    const i = reasons.findIndex(row => row.every(r => r && r !== 'immune-to-every-sampled-reply'));
    if (i >= 0) { keep[i] = true; passiveWait = states[0].p1[i][1]; }
  }
  // A legal wait may be the only way to win via residual effects. Never delete the entire legal action set.
  const fallback = !keep.some(Boolean);
  const excluded = [];
  if (!fallback) for (let i = 0; i < keep.length; i++) {
    if (!keep[i]) excluded.push({label: states[0].p1[i][1], reasons: [...new Set(reasons[i])]});
  }
  if (!fallback) for (const st of states) st.p1 = st.p1.filter((_, i) => keep[i]);
  return {excluded, all_no_effect: fallback, passive_wait: passiveWait,
    scope: 'all sampled worlds and legal root replies'};
}

module.exports = {pruneRoot, reason, canPrecede, immune};
