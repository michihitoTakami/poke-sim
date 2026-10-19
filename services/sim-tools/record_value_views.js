// Saved requests + public prefixes only. No reconstruction from future opponent sets or battle outcomes.
const path = require('path'), readline = require('readline');
const {Dex} = require(path.join(process.argv[2], 'dist/sim'));
const {extractChannelMessages} = require(path.join(process.argv[2], 'dist/sim/battle'));
const {PublicObservation} = require('./public_observation');
const dex = Dex.mod('champions');

function views(record) {
  const turns = new Map(), rows = [], skipped = [], seen = new Set();
  record.log.forEach((line, i) => { if (/^\|turn\|\d+$/.test(line)) turns.set(Number(line.split('|')[2]), i); });
  for (const advice of record.advice || []) {
    if (advice.request?.forceSwitch || !advice.request?.active || !turns.has(advice.turn) || seen.has(advice.turn)) continue;
    try {
      const observer = new PublicObservation(record.log.slice(0, turns.get(advice.turn) + 1), dex, extractChannelMessages);
      const request = JSON.parse(JSON.stringify(advice.request));
      // Saved ladder logs already place us at p1, even when the original wire request called us p2.
      request.side.id = 'p1';
      const view = observer.view({id: 'p1', activeRequest: request, getRequestData: () => request.side});
      const own = view.request.side.pokemon.find(m => m.active);
      const active = view.public.mons.p1[view.public.active.p1];
      if (!own || !active || dex.species.get(own.details.split(',')[0]).baseSpecies !==
          dex.species.get(active.species).baseSpecies) throw Error('request actor does not match turn-start prefix');
      rows.push({turn: advice.turn, view});
      seen.add(advice.turn);
    } catch (e) { skipped.push({turn: advice.turn, error: String(e.message || e)}); }
  }
  return {rows, skipped};
}

readline.createInterface({input: process.stdin}).on('line', line => {
  try { console.log(JSON.stringify(views(JSON.parse(line)))); }
  catch (e) { console.log(JSON.stringify({error: String(e.message || e)})); }
});
