// Static preview estimates from hypothetical sets. No battle result or later reveal is an input.
function previewMatchups(req, deps) {
  const started = Date.now(), rows = [];
  for (const opponent of req.previewMatchups.opponents) {
    for (let i = 0; i < req.previewMatchups.ours.length; i++) {
      for (const mega of [false, true]) {
        const estimates = [];
        for (const set of opponent.sets) {
          for (const weather of req.previewMatchups.weather || ['']) {
            const side = (team, evolved) => ({team: [team], mons: [{hp: 1, mega: evolved}]});
            const b = deps.setUp({p1: side(req.previewMatchups.ours[i], mega), p2: side(set, true),
              field: weather ? {weather, weatherTurns: 5} : {}}, req.format, 1);
            try {
              const me = b.p1.active[0], foe = b.p2.active[0];
              if (mega && !me.species.isMega) break;
              const best = (source, target) => Math.max(0, ...source.moveSlots.map(m =>
                deps.expectedDamage(b, source, target, m.id))) / target.maxhp;
              estimates.push({outgoing: best(me, foe), incoming: best(foe, me),
                faster: me.getActionSpeed() > foe.getActionSpeed()});
            } finally { b.destroy(); }
          }
        }
        if (estimates.length) rows.push({ours: i, foe: opponent.species, mega,
          outgoing: estimates.reduce((s, x) => s + x.outgoing, 0) / estimates.length,
          incoming: estimates.reduce((s, x) => s + x.incoming, 0) / estimates.length,
          faster: estimates.reduce((s, x) => s + x.faster, 0) / estimates.length});
      }
    }
  }
  return {matchups: rows, elapsedMs: Date.now() - started};
}

module.exports = {previewMatchups};
