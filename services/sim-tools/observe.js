// Actor views of sampled worlds without any simulation: what a policy would receive at this decision.
// Each sample is set up like a search world; the observed history is replayed into a PublicObservation and the
// requested side's view (its own request plus public state) and legal choices are returned. The other side's
// private request never leaves this function.
const {PublicObservation, legalChoices} = require('./public_observation');

function observe(req, deps) {
  const sides = req.sides || [req.side || 'p1'];
  for (const side of sides) if (!['p1', 'p2'].includes(side)) throw Error(`unknown side ${side}`);
  const out = [];
  for (const sample of req.samples) {
    const b = deps.setUp(sample, req.format, 1);
    try {
      const obs = new PublicObservation(req.observationLog, b.dex, deps.extractChannelMessages,
        b.dex.currentMod.startsWith('champions'));
      obs.append(b.pokesimEntryLog || []);
      for (const s of b.sides) {
        obs.bindActive(s.id, s.active[0]?.name); obs.bindActive(s.id, s.active[0]?.illusion?.name);
      }
      const views = {};
      for (const side of sides) {
        const view = obs.view(b[side]);
        views[side] = {view, legal: legalChoices(view), ended: b.ended};
      }
      out.push(views);
    } finally { b.destroy(); }
  }
  return {observations: out};
}

module.exports = {observe};
