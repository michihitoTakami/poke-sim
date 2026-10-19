// Approximate damage for live continuations, using client-visible facts only.
// The frozen public-reference-v2 policy deliberately does not use this estimator.
const {health, id} = require('./public_observation');
const stage = n => n >= 0 ? (2 + n) / 2 : 2 / (2 - n);

function estimateDamage(view, dex, source, target, raw, sourceBoosts, targetBoosts) {
  if (!raw || raw.category === 'Status') return 0;
  const species = m => dex.species.get(m.details?.split(',')[0] || m.species);
  const observed = m => m.details ? (m.active ? view.public.mons[view.side][view.public.active[view.side]] :
    Object.values(view.public.mons[view.side]).find(p => species(p).baseSpecies === species(m).baseSpecies)) : m;
  const boosts = m => m.details && !m.active ? {} : observed(m)?.boosts || {};
  const typesOf = m => m.types || observed(m)?.types || species(m).types;
  const stat = (m, s, projected) => (m.stats?.[s] ??
    Math.floor((2 * species(m).baseStats[s] + 47) / 2) + 5) * stage((projected || boosts(m))[s] || 0);
  const own = view.request.side.pokemon.find(m => m.active);
  const foeSide = view.side === 'p1' ? 'p2' : 'p1';
  const foe = view.public.mons[foeSide][view.public.active[foeSide]];
  const suppressed = [own, foe].some(m => ['airlock', 'cloudnine'].includes(m?.ability));
  const weather = suppressed ? '' : id(view.public.weather);
  let move = {...raw}, power = raw.basePower || 60;
  if (move.id === 'weatherball') {
    const type = {sunnyday: 'Fire', desolateland: 'Fire', raindance: 'Water', primordialsea: 'Water',
      sandstorm: 'Rock', snowscape: 'Ice', hail: 'Ice'}[weather];
    move.type = type || 'Normal'; power = type ? 100 : 50;
  }
  if (move.id === 'lastrespects') {
    const side = source.details ? view.side : foeSide;
    // Cumulative public faints include revived teammates; current fainted membership is insufficient.
    power = 50 + 50 * (view.public.totalFainted?.[side] || 0);
  }
  if (['storedpower', 'powertrip'].includes(move.id))
    power = 20 + 20 * Object.values(sourceBoosts || boosts(source)).reduce((s, n) => s + Math.max(0, n), 0);
  const ability = source.ability || observed(source)?.ability || '';
  const targetAbility = target.ability || observed(target)?.ability || '';
  const convert = {aerilate: 'Flying', pixilate: 'Fairy', refrigerate: 'Ice', galvanize: 'Electric'};
  if (move.type === 'Normal' && convert[ability]) { move.type = convert[ability]; power *= 1.2; }
  if (ability === 'normalize') { move.type = 'Normal'; power *= 1.2; }
  if (ability === 'liquidvoice' && move.flags?.sound) move.type = 'Water';
  const types = typesOf(target);
  if (!dex.getImmunity(move.type, types)) return 0;
  const bypass = ['moldbreaker', 'teravolt', 'turboblaze'].includes(ability) && target.item !== 'abilityshield';
  if (move.type === 'Ground' && target.item === 'airballoon') return 0;
  if (!bypass && ((move.type === 'Ground' && targetAbility === 'levitate') ||
      (move.type === 'Water' && ['waterabsorb', 'stormdrain', 'dryskin'].includes(targetAbility)) ||
      (move.type === 'Fire' && ['flashfire', 'wellbakedbody'].includes(targetAbility)) ||
      (move.type === 'Electric' && ['voltabsorb', 'lightningrod', 'motordrive'].includes(targetAbility)) ||
      (move.type === 'Grass' && targetAbility === 'sapsipper'))) return 0;
  const physical = move.category === 'Physical';
  let attackBoosts = sourceBoosts || boosts(source), defenceBoosts = targetBoosts || boosts(target);
  if (!bypass && targetAbility === 'unaware') attackBoosts = {};
  if (ability === 'unaware') defenceBoosts = {};
  const critical = observed(source)?.volatiles?.includes('focusenergy') || move.willCrit;
  if (critical) {
    attackBoosts = Object.fromEntries(Object.entries(attackBoosts).map(([s, n]) => [s, Math.max(0, n)]));
    defenceBoosts = Object.fromEntries(Object.entries(defenceBoosts).map(([s, n]) => [s, Math.min(0, n)]));
  }
  const attackStat = move.overrideOffensiveStat || (physical ? 'atk' : 'spa');
  const defenceStat = move.overrideDefensiveStat || (physical ? 'def' : 'spd');
  const attacker = move.overrideOffensivePokemon === 'target' ? target : source;
  let attack = stat(attacker, attackStat, attacker === source ? attackBoosts : defenceBoosts);
  if (attackStat === 'atk' && ['hugepower', 'purepower'].includes(ability)) attack *= 2;
  if (attackStat === 'atk' && source.item === 'choiceband') attack *= 1.5;
  if (attackStat === 'spa' && source.item === 'choicespecs') attack *= 1.5;
  if (Array.isArray(move.multihit)) power *= 3; else power *= move.multihit || 1;
  let damage = ((22 * power * attack / Math.max(1, stat(target, defenceStat, defenceBoosts))) / 50 + 2) * .925;
  if (critical) damage *= 1.5;
  if (typesOf(source).includes(move.type)) damage *= ability === 'adaptability' ? 2 : 1.5;
  damage *= 2 ** dex.getEffectiveness(move.type, types);
  if (['sunnyday', 'desolateland'].includes(weather)) damage *= move.type === 'Fire' ? 1.5 : move.type === 'Water' ? .5 : 1;
  if (['raindance', 'primordialsea'].includes(weather)) damage *= move.type === 'Water' ? 1.5 : move.type === 'Fire' ? .5 : 1;
  if (move.type === 'Fairy' && [own, foe, source, target].some(m => m?.ability === 'fairyaura')) damage *= 4 / 3;
  if (move.type === 'Dark' && [own, foe, source, target].some(m => m?.ability === 'darkaura')) damage *= 4 / 3;
  if (physical && health(source.condition).status === 'brn' && ability !== 'guts') damage *= .5;
  if (typeof move.accuracy === 'number') damage *= move.accuracy / 100;
  return damage;
}

// The actor knows its own stone. A policy that always selects the Mega variant must evaluate that variant.
function megaSource(view, dex, mon) {
  if (!mon.active || !view.request.active?.[0]?.canMegaEvo) return mon;
  const from = dex.species.get(mon.details.split(',')[0]);
  const stone = dex.items.get(mon.item).megaStone;
  const name = typeof stone === 'string' ? stone : stone?.[from.name];
  if (!name) return mon;
  const to = dex.species.get(name);
  if (!to.exists) return mon;
  const stats = mon.stats ? Object.fromEntries(Object.entries(mon.stats).map(([s, n]) =>
    [s, n + (to.baseStats[s] || 0) - (from.baseStats[s] || 0)])) : undefined;
  return {...mon, details: name, ability: id(to.abilities['0']), stats, types: to.types};
}

module.exports = {estimateDamage, megaSource};
