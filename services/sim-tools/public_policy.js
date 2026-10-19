// A versioned, deliberately simple reference policy over client observations. No sampled world or engine objects.
const {health, id, legalChoices} = require('./public_observation');
const {estimateDamage, megaSource} = require('./public_damage');
const VERSION = 'public-reference-v2';
// Keep the training reference frozen. The live tree uses the revised policy explicitly;
// its V_ref remains a v2 bootstrap estimate, not a value trained for this new continuation.
const BATON_VERSION = 'public-baton-v4';
const stage = s => s >= 0 ? (2 + s) / 2 : 2 / (2 - s);

function choosePublic(view, name, dex, baton = false, mechanics = baton) {
  if (!['greedy', 'tactical', 'expert', 'disrupt'].includes(name)) throw new Error(`unknown public policy ${name}`);
  const opts = legalChoices(view);
  if (!opts.length) throw new Error('public request has no legal actions');
  if (opts.length === 1) return opts[0];
  const side = view.side, foeSide = side === 'p1' ? 'p2' : 'p1';
  const own = view.request.side.pokemon;
  const me = own.find(m => m.active);
  const foe = view.public.mons[foeSide][view.public.active[foeSide]];
  if (!foe) return opts[0];
  const spec = m => dex.species.get(m.details ? m.details.split(',')[0] : m.species);
  const publicMon = m => m.active ? view.public.mons[side][view.public.active[side]] : Object.values(view.public.mons[side]).find(p => spec(p).baseSpecies === spec(m).baseSpecies);
  const boosts = m => baton && m !== foe && !m.active ? {} : (m === foe ? foe : publicMon(m))?.boosts || {};
  const stat = (m, s, projected) => {
    const base = spec(m).baseStats[s];
    // Opponent EVs/nature/item modifiers remain unknown: this is an explicit neutral, half-invested estimate.
    let value = m.stats?.[s] ?? Math.floor((2 * base + 31 + 16) / 2) + 5;
    value *= stage((projected || boosts(m))[s] || 0);
    const h = health(m.condition);
    if (s === 'spe' && h.status === 'par') value *= .5;
    if (baton && s === 'spe' && m.item === 'choicescarf') value *= 1.5;
    if (baton && s === 'spe') {
      const team = m === foe ? foeSide : side;
      if (Object.keys(view.public.conditions?.[team] || {}).some(c => /tailwind/i.test(c))) value *= 2;
      const weather = id(view.public.weather);
      if ((['raindance', 'primordialsea'].includes(weather) && m.ability === 'swiftswim') ||
          (['sunnyday', 'desolateland'].includes(weather) && m.ability === 'chlorophyll') ||
          (weather === 'sandstorm' && m.ability === 'sandrush') ||
          (['snow', 'snowscape', 'hail'].includes(weather) && m.ability === 'slushrush')) value *= 2;
    }
    return value;
  };
  const maxhp = m => m === foe ? Math.floor((2 * spec(m).baseStats.hp + 31 + 16) / 2) + 60 : health(m.condition).maxhp;
  const hp = m => m === foe ? health(m.condition).fraction * maxhp(m) : health(m.condition).hp;
  const damage = (source, target, move, sourceBoosts, targetBoosts) => {
    if (mechanics) return estimateDamage(view, dex, source === me ? megaSource(view, dex, source) : source,
      target === me ? megaSource(view, dex, target) : target, move, sourceBoosts, targetBoosts);
    if (!move || move.category === 'Status') return 0;
    const types = spec(target).types;
    if (!dex.getImmunity(move.type, types)) return 0;
    const ability = target.ability || '';
    if ((move.type === 'Ground' && (ability === 'levitate' || target.item === 'airballoon')) ||
        (move.type === 'Water' && ['waterabsorb', 'stormdrain'].includes(ability)) ||
        (move.type === 'Fire' && ability === 'flashfire')) return 0;
    const physical = move.category === 'Physical';
    let power = move.basePower || (move.id === 'bodypress' ? 80 : 60);
    if (Array.isArray(move.multihit)) power *= 3; else power *= move.multihit || 1;
    if (baton && target.ability === 'unaware') sourceBoosts = {};
    if (baton && source.ability === 'unaware') targetBoosts = {};
    const focus = baton && source === foe && (foe.volatiles.includes('focusenergy') || move.willCrit);
    if (focus) {
      sourceBoosts = Object.fromEntries(Object.entries(sourceBoosts || boosts(source)).map(([s, n]) => [s, Math.max(0, n)]));
      targetBoosts = Object.fromEntries(Object.entries(targetBoosts || boosts(target)).map(([s, n]) => [s, Math.min(0, n)]));
    }
    let result = ((22 * power * stat(source, physical ? 'atk' : 'spa', sourceBoosts) /
      Math.max(1, stat(target, physical ? 'def' : 'spd', targetBoosts))) / 50 + 2) * .925;
    if (focus) result *= 1.5;
    if (spec(source).types.includes(move.type)) result *= 1.5;
    result *= 2 ** dex.getEffectiveness(move.type, types);
    if (physical && health(source.condition).status === 'brn') result *= .5;
    if (typeof move.accuracy === 'number') result *= move.accuracy / 100;
    return result;
  };
  const knownThreats = foe.moves.map(m => dex.moves.get(m)).filter(m => m.category !== 'Status');
  // An unrevealed slot is not assumed harmless: use a generic STAB attack from public species statistics.
  const generic = spec(foe).types.map(type => ({type, basePower: 90, accuracy: 100,
    category: spec(foe).baseStats.atk >= spec(foe).baseStats.spa ? 'Physical' : 'Special'}));
  const threats = [...knownThreats, ...(!baton || foe.moves.length < 4 ? generic : [])];
  const incoming = (target, projected) => Math.max(0, ...threats.map(m => damage(foe, target, m, undefined, projected)));
  const bestDamage = (source, projected) => Math.max(0,
    ...source.moves.map(m => damage(source, foe, dex.moves.get(m), projected)));
  const knownMe = publicMon(me) || {boosts: {}, volatiles: []};
  let trickRoom = false;
  if (baton) for (const line of view.history || []) {
    if (/^\|-fieldstart\|(?:move: )?Trick Room/.test(line)) trickRoom = true;
    if (/^\|-fieldend\|(?:move: )?Trick Room/.test(line)) trickRoom = false;
  }
  const faster = (mon, projected) => trickRoom ? stat(mon, 'spe', projected) < stat(foe, 'spe') :
    stat(mon, 'spe', projected) > stat(foe, 'spe');
  const scoreSwitch = ([choice, label], passing = false) => {
    const mon = own[Number(choice.split(' ')[1]) - 1];
    const projected = passing ? knownMe.boosts : undefined;
    let score = bestDamage(mon, projected) / Math.max(1, hp(foe)) -
      incoming(mon, projected) / Math.max(1, hp(mon));
    if (baton && faster(mon, projected)) score += .5;
    if (foe.moves.some(m => ['yawn', 'roar', 'whirlwind'].includes(m)) && mon.ability === 'goodasgold') score += 2;
    if (foe.moves.includes('dragontail') && spec(mon).types.includes('Fairy')) score += 2;
    if (mon.ability === 'unaware' && ((foe.boosts.atk || 0) > 0 || (foe.boosts.spa || 0) > 0)) score += 2;
    return {choice, label, mon, score};
  };
  const switchOpts = opts.filter(([, l]) => l.startsWith('switch:'));
  const passing = baton && view.phase === 'switch' && hp(me) > 0 && knownMe.lastMove === 'batonpass';
  const switches = switchOpts.map(o => scoreSwitch(o, passing)).sort((a, b) => b.score - a.score);
  // Baton Pass can leave trapping, so its recipients are not limited to ordinary legal switches.
  let passTargets = baton ? own.flatMap((m, i) => !m.active && hp(m) > 0 ?
    [scoreSwitch([`switch ${i + 1}`, `switch:${id(m.details.split(',')[0])}`], true)] : []) : [];
  if (baton) {
    // A passive receiver must still add a missing, useful boost for a living attacker.
    // Otherwise the expert can bounce a completed payload between Scolipede and Umbreon forever.
    passTargets = passTargets.filter(target => {
      const relay = target.mon, payload = knownMe.boosts;
      if (bestDamage(relay, payload) > 0) return true;
      const endpoints = own.filter(m => m !== relay && hp(m) > 0 && bestDamage(m, payload) > 0);
      const addSpeed = relay.ability === 'speedboost' && (payload.spe || 0) < 6 &&
        endpoints.some(m => !faster(m, payload));
      const addDefence = relay.moves.includes('irondefense') && (payload.def || 0) < 4 &&
        threats.some(m => m.category === 'Physical');
      const addSpecial = relay.moves.includes('calmmind') && (payload.spa || 0) < 2 &&
        endpoints.some(m => m.moves.some(mid => dex.moves.get(mid).category === 'Special'));
      return endpoints.length && (addSpeed || addDefence || addSpecial) &&
        incoming(relay, payload) * 3 < hp(relay) &&
        !foe.moves.some(m => ['haze', 'roar', 'whirlwind', 'encore', 'taunt'].includes(m));
    });
    // A relay is useful only while it can safely add special boosts for a remaining special attacker.
    for (const target of passTargets) {
      const relay = target.mon;
      const endpoint = passTargets.some(s => s !== target && s.mon.moves.some(mid => {
        const m = dex.moves.get(mid);
        return m.category === 'Special' && m.basePower >= 70 && damage(s.mon, foe, m, knownMe.boosts) > 0;
      }));
      if (!me.moves.includes('calmmind') && relay.moves.includes('batonpass') && relay.moves.includes('calmmind') &&
          (knownMe.boosts.spa || 0) < 2 && (knownMe.boosts.def || 0) >= 2 && endpoint &&
          incoming(relay, knownMe.boosts) * 3 < hp(relay) &&
          !foe.moves.some(m => ['haze', 'roar', 'whirlwind', 'encore', 'taunt'].includes(m))) target.score += 2;
    }
    passTargets.sort((a, b) => b.score - a.score);
    if (passing && passTargets.length) return [passTargets[0].choice, passTargets[0].label];
  }
  if (view.phase === 'switch') return switches.length ? [switches[0].choice, switches[0].label] : opts[0];
  const moves = opts.filter(([, l]) => l.startsWith('move:') && !l.endsWith(':mega')).map(([choice, label]) => {
    const move = dex.moves.get(label.split(':')[1]);
    return {choice, label, move, damage: damage(me, foe, move)};
  }).filter(x => !mechanics || x.move.id !== 'substitute' ||
    (hp(me) > Math.floor(maxhp(me) / 4) && !knownMe.volatiles.includes('substitute')));
  if (!moves.length) return opts[0];
  const finish = candidate => opts.find(([, l]) => l === candidate.label + ':mega') || [candidate.choice, candidate.label];
  const pick = move => moves.find(x => x.move.id === move);
  const best = moves.reduce((a, b) => b.damage > a.damage ? b : a);
  if (name === 'greedy') return finish(best);
  const hit = incoming(me), currentHp = hp(me), maximum = maxhp(me);
  if (best.damage >= hp(foe) && (!baton || faster(me) || hit < currentHp || best.move.priority > 0)) return finish(best);
  if (baton && name === 'expert' && pick('batonpass') && passTargets.length) {
    const pass = pick('batonpass'), sub = pick('substitute'), protect = pick('protect') || pick('detect');
    const gained = Object.values(knownMe.boosts).reduce((s, b) => s + Math.max(0, b), 0);
    const covered = knownMe.volatiles.includes('substitute');
    // Safety uses a hit that lands, rather than accuracy-weighted expected damage.
    const worst = Math.max(0, ...threats.map(m => damage(foe, me, {...m, accuracy: true}) / .925));
    const priority = threats.some(m => m.priority > 0 && damage(foe, me, m) >= currentHp);
    const sheltered = covered && foe.ability !== 'infiltrator' &&
      !threats.some(m => m.flags?.sound || m.flags?.bypasssub || m.multihit);
    const canAct = !priority && (faster(me) || worst < currentHp || sheltered);
    const stops = foe.moves.some(m => ['encore', 'taunt', 'haze', 'clearsmog', 'roar', 'whirlwind', 'yawn'].includes(m));
    const carry = gained > 0 || covered;
    const status = health(me.condition).status;
    let residual = status === 'brn' ? maximum / 16 : ['tox', 'psn'].includes(status) ? maximum / 8 : 0;
    if (id(view.public.weather) === 'sandstorm' && !spec(me).types.some(t => ['Rock', 'Ground', 'Steel'].includes(t)) &&
        !['magicguard', 'overcoat', 'sandveil', 'sandrush', 'sandforce'].includes(me.ability) && me.item !== 'safetygoggles')
      residual += maximum / 16;
    if (me.ability === 'magicguard') residual = 0;
    // An unseen Garchomp can be special. Scout its first attack while obtaining the first Speed Boost,
    // instead of committing the unprotected passer to Iron Defense from base stats alone.
    if (protect && me.ability === 'speedboost' && !carry && !knownMe.lastMove && foe.moves.length === 0 &&
        residual < currentHp) return finish(protect);
    const special = Math.max(0, ...threats.filter(m => m.category === 'Special').map(m =>
      damage(foe, me, {...m, accuracy: true}) / .925));
    const physical = Math.max(0, ...threats.filter(m => m.category === 'Physical').map(m =>
      damage(foe, me, {...m, accuracy: true}) / .925));
    const speedNeeded = me.ability === 'speedboost' && (knownMe.boosts.spe || 0) < 6 &&
      !faster(passTargets[0].mon, knownMe.boosts);
    const unsafeEntry = faster(me) && !sheltered &&
      incoming(passTargets[0].mon, knownMe.boosts) >= hp(passTargets[0].mon);
    const interrupted = knownMe.volatiles.includes('yawn') ||
      foe.moves.some(m => ['encore', 'taunt', 'haze', 'clearsmog', 'roar', 'whirlwind'].includes(m));
    // Against special pressure a breakable shield can still buy a Speed Boost. Alternate the two
    // protections until the RECEIVER wins the speed race, rather than spending HP on successive shields.
    if ((speedNeeded || unsafeEntry) && special > physical && !interrupted && !priority) {
      const lastProtect = ['protect', 'detect'].includes(knownMe.lastMove);
      if (protect && !lastProtect && residual < currentHp) return finish(protect);
      const priorityDamage = Math.max(0, ...threats.filter(m => m.priority > 0).map(m =>
        damage(foe, me, {...m, accuracy: true}) / .925));
      if (sub && !covered && canAct && currentHp - Math.floor(maximum / 4) > residual + priorityDamage &&
          foe.ability !== 'infiltrator' && !threats.some(m => m.flags?.sound || m.flags?.bypasssub || m.multihit))
        return finish(sub);
    }
    if (carry && canAct && (2 * worst + residual >= currentHp || stops || knownMe.volatiles.includes('yawn')))
      return finish(pass);
    const heal = moves.find(x => x.move.heal && x.move.target === 'self');
    if (heal && currentHp < maximum / 2 && worst + residual < maximum / 2 && canAct) return finish(heal);
    const setup = moves.find(x => x.move.target === 'self' && x.move.boosts &&
      Object.entries(x.move.boosts).some(([s, n]) => n > 0 && (knownMe.boosts[s] || 0) <
        (x.move.id === 'calmmind' ? 2 : 4)));
    const safe = !stops && worst * 2 + residual < currentHp;
    if (setup && safe) return finish(setup);
    if (sub && !covered && safe && worst < maximum / 4 && currentHp > maximum / 2 &&
        knownMe.substituteBrokenTurn !== view.turn - 1) return finish(sub);
    const speedBoost = me.ability === 'speedboost';
    if (protect && speedBoost && knownMe.lastMove !== 'protect' && knownMe.lastMove !== 'detect' &&
        residual < currentHp && (!carry || (!faster(passTargets[0].mon, knownMe.boosts) && (knownMe.boosts.spe || 0) < 6)))
      return finish(protect);
    if (carry && canAct) return finish(pass);
    // No useful payload, or a faster lethal hit: let the fallback take a legal hard switch.
    const escape = switches.find(s => incoming(s.mon) < hp(s.mon));
    if (escape) return [escape.choice, escape.label];
    // With no escape, still try to transfer what we have rather than repeat Protect or spend HP on Substitute.
    if (carry) return finish(pass);
  }
  const safeSwitch = switches.find(s => incoming(s.mon) / Math.max(1, hp(s.mon)) < hit / currentHp && bestDamage(s.mon) > 0);
  if (safeSwitch && (best.damage <= 0 || hit >= currentHp || (hit * 2 >= currentHp && best.damage * 2 < hp(foe)))) {
    return [safeSwitch.choice, safeSwitch.label];
  }
  if (name === 'disrupt' && (foe.lastMove || Object.values(foe.boosts).some(v => v > 0))) {
    const disruption = moves.find(x => ['haze', 'roar', 'whirlwind', 'encore', 'taunt', 'toxic', 'willowisp', 'yawn']
      .includes(x.move.id) && (x.move.id !== 'encore' || dex.moves.get(foe.lastMove).category === 'Status'));
    if (disruption && foe.ability !== 'goodasgold' && !foe.volatiles.includes('substitute')) return finish(disruption);
  }
  const heal = moves.find(x => x.move.heal && x.move.target === 'self');
  const healingHit = heal && mechanics && heal.move.id === 'roost' && faster(me) ? Math.max(0,
    ...threats.map(m => {
      const target = megaSource(view, dex, me);
      const types = spec(target).types.filter(t => t !== 'Flying');
      return damage(foe, (m.priority || 0) <= 0 ? {...target, types: types.length ? types : ['Normal']} : me, m);
    })) : hit;
  if (heal && currentHp < maximum / 2 && healingHit < maximum / 2) return finish(heal);
  const pass = pick('batonpass');
  const gained = Object.values(knownMe.boosts).reduce((s, b) => s + Math.max(0, b), 0);
  const sub = pick('substitute');
  if (sub && !knownMe.volatiles.includes('substitute') && hit < maximum / 4 && currentHp > maximum / 2 &&
      knownMe.substituteBrokenTurn !== view.turn - 1) return finish(sub);
  const setup = moves.find(x => x.move.target === 'self' && x.move.boosts && Object.entries(x.move.boosts)
    .some(([s, n]) => n > 0 && (knownMe.boosts[s] || 0) < (name === 'expert' ? 6 : 2)));
  const encoreRisk = foe.moves.includes('encore') || foe.volatiles.includes('encore');
  if (pass && gained >= 2 && switches.some(s => bestDamage(s.mon) > 0) &&
      (hit * 2 >= currentHp || encoreRisk || (knownMe.boosts.spe || 0) >= 2)) return finish(pass);
  const criticalDefence = baton && foe.volatiles.includes('focusenergy') && setup?.move.boosts?.spd;
  if (setup && !criticalDefence && !encoreRisk && hit * 3 < currentHp && best.damage * 2 < hp(foe)) return finish(setup);
  if (pass && gained >= 2 && (baton ? passTargets.length : switches.length)) return finish(pass);
  const protect = pick('protect');
  if (pass && protect && (knownMe.boosts.spe || 0) < 2 && knownMe.lastMove !== 'protect') return finish(protect);
  return finish(best);
}

function observedStep(battle, ours, theirs, step, rechoose, allowed = ['p1', 'p2']) {
  const actions = [ours, theirs];
  for (let attempt = 0; attempt < 3; attempt++) {
    try { return step(battle, actions[0], actions[1]); }
    catch (e) {
      // The official engine sends this private request update only after the attempted switch is rejected.
      // Retry from that observed request; never inspect the opponent's hidden trapping ability.
      const match = String(e.message || e).match(/^Choice rejected: (p[12]) switch \d+: Can't switch: The active Pokémon is trapped/);
      if (!match || !allowed.includes(match[1]) || !battle[match[1]].activeRequest?.active?.[0]?.trapped) throw e;
      actions[match[1] === 'p1' ? 0 : 1] = rechoose(battle[match[1]]);
    }
  }
  throw new Error('observed request retries exhausted');
}

const chooseBatonPublic = (view, name, dex) => {
  if (name === 'expert') {
    const strategy = require('./relay_strategy').relayStrategy(view, dex);
    if (strategy) return strategy.choice;
  }
  return choosePublic(view, name, dex, name === 'expert', true);
};
module.exports = {choosePublic, chooseBatonPublic, observedStep, VERSION, BATON_VERSION};
