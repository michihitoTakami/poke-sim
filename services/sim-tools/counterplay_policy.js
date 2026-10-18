// An additional preview stress opponent. Only the actor's request and public observations are inputs.
const {legalChoices} = require('./public_observation');
const {chooseBatonPublic} = require('./public_policy');
const {estimateDamage, megaSource} = require('./public_damage');
const VERSION = 'public-counterplay-v1';

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
  // Attack the exposed recipient rather than repeatedly selecting a neutral generic status policy.
  const best = chooseBatonPublic(view, 'greedy', dex);
  const attack = dex.moves.get(best[1].split(':')[1]);
  if (best[1].startsWith('move:') && attack.category !== 'Status' &&
      estimateDamage(view, dex, best[1].endsWith(':mega') ? megaSource(view, dex, own) : own, foe, attack) > 0)
    return best;
  return chooseBatonPublic(view, 'expert', dex);
}
module.exports = {chooseCounterplay, VERSION};
