// An additional preview stress opponent. Only the actor's request and public observations are inputs.
const {legalChoices, health} = require('./public_observation');
const {chooseBatonPublic} = require('./public_policy');
const {estimateDamage, megaSource} = require('./public_damage');
const VERSION = 'public-counterplay-v2';

function chooseCounterplay(view, dex) {
  const opts = legalChoices(view), own = view.request.side.pokemon.find(m => m.active);
  const enemy = view.side === 'p1' ? 'p2' : 'p1';
  const foe = view.public.mons[enemy][view.public.active[enemy]];
  if (view.phase !== 'move' || !foe || !own) return chooseBatonPublic(view, 'expert', dex);
  const move = mid => opts.find(([, label]) => label === `move:${mid}:mega`) ||
    opts.find(([, label]) => label === `move:${mid}`);
  const hasBoost = Object.values(foe.boosts).some(n => n > 0);
  if (hasBoost && move('haze')) return move('haze');
  const blocked = foe.volatiles.includes('substitute') || ['magicbounce', 'goodasgold'].includes(foe.ability);
  const statusPlay = !foe.moves.length || foe.moves.some(mid => {
    const m = dex.moves.get(mid); return m.category === 'Status' && (m.boosts || m.heal || m.selfSwitch ||
      ['protect', 'substitute'].includes(mid));
  });
  if (!blocked && statusPlay && !foe.volatiles.includes('taunt') && move('taunt')) return move('taunt');
  if (!blocked && hasBoost) {
    for (const mid of ['whirlwind', 'roar', 'encore']) if (move(mid) &&
        (mid !== 'encore' || dex.moves.get(foe.lastMove).category === 'Status')) return move(mid);
  }
  const me = view.public.mons[view.side][view.public.active[view.side]];
  const spec = m => dex.species.get(m.details?.split(',')[0] || m.species);
  const types = foe.types || spec(foe).types;
  const knownAbility = foe.ability || (new Set(Object.values(spec(foe).abilities)).size === 1 ?
    String(spec(foe).abilities['0']).toLowerCase().replace(/[^a-z0-9]/g, '') : '');
  const immuneStatus = blocked || ['magicbounce', 'goodasgold'].includes(knownAbility);
  const best = chooseBatonPublic(view, 'greedy', dex);
  const attack = dex.moves.get(best[1].split(':')[1]);
  const damage = best[1].startsWith('move:') && attack.category !== 'Status' ?
    estimateDamage(view, dex, best[1].endsWith(':mega') ? megaSource(view, dex, own) : own, foe, attack) : 0;
  const finish = damage * .85 / .925 >= health(foe.condition).fraction * (spec(foe).baseStats.hp + 107);
  if (!finish && !immuneStatus) {
    const poison = move('toxic');
    if (poison && !health(foe.condition).status && !types.some(t => ['Steel', 'Poison'].includes(t))) return poison;
    // Soak removes poison immunity and can expose an Electric/Grass weakness; it is not a damage move.
    if (move('soak') && !(types.length === 1 && types[0] === 'Water') &&
        ((poison && !health(foe.condition).status) ||
          own.moves.some(mid => ['Electric', 'Grass'].includes(dex.moves.get(mid).type) &&
            dex.moves.get(mid).category !== 'Status'))) return move('soak');
  }
  // Punish free Protect/Baton/recovery turns with special setup as well as Taunt/phazing.
  // Known attacks plus ordinary STAB bound the risk; no sampled opposing moves enter the actor.
  if (!finish && statusPlay && me) {
    const threats = foe.moves.map(mid => dex.moves.get(mid)).filter(m => m.category !== 'Status');
    if (foe.moves.length < 4) for (const type of spec(foe).types) threats.push({type, basePower: 90,
      category: spec(foe).baseStats.atk >= spec(foe).baseStats.spa ? 'Physical' : 'Special'});
    const source = megaSource(view, dex, own);
    const setup = ['calmmind', 'nastyplot'].find(mid => move(mid));
    const incoming = Math.max(0, ...threats.map(m => estimateDamage(view, dex, foe, source,
      {...m, accuracy: true}) / .925));
    if (setup && (me.boosts.spa || 0) < 2 && damage > 0 &&
        !['tox', 'slp', 'frz'].includes(health(own.condition).status) &&
        !me.volatiles.some(v => ['taunt', 'yawn', 'encore', 'curse'].includes(v)) &&
        incoming < health(own.condition).hp * .4) return move(setup);
  }
  // Attack the exposed recipient rather than repeatedly selecting a neutral generic status policy.
  if (best[1].startsWith('move:') && attack.category !== 'Status' &&
      estimateDamage(view, dex, best[1].endsWith(':mega') ? megaSource(view, dex, own) : own, foe, attack) > 0)
    return best;
  return chooseBatonPublic(view, 'expert', dex);
}
module.exports = {chooseCounterplay, VERSION};
