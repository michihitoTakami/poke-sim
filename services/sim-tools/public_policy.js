// A versioned, deliberately simple reference policy over client observations. No sampled world or engine objects.
const {health, id, legalChoices} = require('./public_observation');
const VERSION = 'public-reference-v1';
const stage = s => s >= 0 ? (2 + s) / 2 : 2 / (2 - s);

function choosePublic(view, name, dex) {
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
  const publicMon = m => Object.values(view.public.mons[side]).find(p => spec(p).baseSpecies === spec(m).baseSpecies);
  const boosts = m => (m === foe ? foe : publicMon(m))?.boosts || {};
  const stat = (m, s) => {
    const base = spec(m).baseStats[s];
    // Opponent EVs/nature/item modifiers remain unknown: this is an explicit neutral, half-invested estimate.
    let value = m.stats?.[s] ?? Math.floor((2 * base + 31 + 16) / 2) + 5;
    value *= stage(boosts(m)[s] || 0);
    const h = health(m.condition);
    if (s === 'spe' && h.status === 'par') value *= .5;
    return value;
  };
  const maxhp = m => m === foe ? Math.floor((2 * spec(m).baseStats.hp + 31 + 16) / 2) + 60 : health(m.condition).maxhp;
  const hp = m => m === foe ? health(m.condition).fraction * maxhp(m) : health(m.condition).hp;
  const damage = (source, target, move) => {
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
    let result = ((22 * power * stat(source, physical ? 'atk' : 'spa') /
      Math.max(1, stat(target, physical ? 'def' : 'spd'))) / 50 + 2) * .925;
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
  const threats = [...knownThreats, ...generic];
  const incoming = target => Math.max(0, ...threats.map(m => damage(foe, target, m)));
  const bestDamage = source => Math.max(0, ...source.moves.map(m => damage(source, foe, dex.moves.get(m))));
  const switches = opts.filter(([, l]) => l.startsWith('switch:')).map(([choice, label]) => {
    const mon = own[Number(choice.split(' ')[1]) - 1];
    let score = bestDamage(mon) / Math.max(1, hp(foe)) - incoming(mon) / Math.max(1, hp(mon));
    if (foe.moves.some(m => ['yawn', 'roar', 'whirlwind'].includes(m)) && mon.ability === 'goodasgold') score += 2;
    if (foe.moves.includes('dragontail') && spec(mon).types.includes('Fairy')) score += 2;
    if (mon.ability === 'unaware' && ((foe.boosts.atk || 0) > 0 || (foe.boosts.spa || 0) > 0)) score += 2;
    return {choice, label, mon, score};
  }).sort((a, b) => b.score - a.score);
  if (view.phase === 'switch') return switches.length ? [switches[0].choice, switches[0].label] : opts[0];
  const moves = opts.filter(([, l]) => l.startsWith('move:') && !l.endsWith(':mega')).map(([choice, label]) => {
    const move = dex.moves.get(label.split(':')[1]);
    return {choice, label, move, damage: damage(me, foe, move)};
  });
  if (!moves.length) return opts[0];
  const finish = candidate => opts.find(([, l]) => l === candidate.label + ':mega') || [candidate.choice, candidate.label];
  const pick = move => moves.find(x => x.move.id === move);
  const best = moves.reduce((a, b) => b.damage > a.damage ? b : a);
  if (name === 'greedy') return finish(best);
  const knownMe = publicMon(me) || {boosts: {}, volatiles: []};
  const hit = incoming(me), currentHp = hp(me), maximum = maxhp(me);
  if (best.damage >= hp(foe)) return finish(best);
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
  if (heal && currentHp < maximum / 2 && hit < maximum / 2) return finish(heal);
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
  if (setup && !encoreRisk && hit * 3 < currentHp && best.damage * 2 < hp(foe)) return finish(setup);
  if (pass && gained >= 2 && switches.length) return finish(pass);
  const protect = pick('protect');
  if (pass && protect && (knownMe.boosts.spe || 0) < 2 && knownMe.lastMove !== 'protect') return finish(protect);
  return finish(best);
}

module.exports = {choosePublic, VERSION};
