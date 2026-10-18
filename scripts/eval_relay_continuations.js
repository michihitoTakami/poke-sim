// Offline cross-play: the same hypothetical sets, chance streams and frozen opponent for both actors.
// Usage: node scripts/eval_relay_continuations.js ROOT BASELINE SHOWDOWN CASES OUT
const fs = require('fs'), path = require('path'), vm = require('vm');
const [root, before, showdown, casesFile, out] = process.argv.slice(2);
const {PublicObservation} = require(path.join(root, 'services/sim-tools/public_observation'));
const {observedStep} = require(path.join(root, 'services/sim-tools/public_policy'));
const opponent = require(path.join(before, 'services/sim-tools/public_policy')).choosePublic;
const actors = Object.fromEntries([['before', before], ['after', root]].map(([name, dir]) =>
  [name, require(path.join(dir, 'services/sim-tools/public_plans')).PlanPolicy]));
const worker = path.join(root, 'services/sim-tools/lookahead.js');
const sandbox = {__dirname: path.dirname(worker),
  require: n => n === 'readline' ? {createInterface: () => ({on: () => {}})} : require(n),
  process: {argv: ['node', 'offline', showdown]}, module: {exports: {}}, console};
vm.runInNewContext(fs.readFileSync(worker, 'utf8') + '\nmodule.exports={step,startBattle};', sandbox);
const {step, startBattle} = sandbox.module.exports;
const {extractChannelMessages} = require(path.join(showdown, 'dist/sim/battle'));
const trials = [];
const input = JSON.parse(fs.readFileSync(casesFile, 'utf8'));
const cases = Array.isArray(input) ? input : input.continuation_matches.cases;
for (const item of cases) {
  for (const seed of [11, 43]) for (const style of ['tactical', 'disrupt', 'expert']) {
    for (const [name, Actor] of Object.entries(actors)) {
      const b = startBattle(item.sample, 'gen9championsbssregmc', seed);
      const row = {record: item.record, seed, opponent: style, actor: name, outcome: null, turns: 0, error: null};
      try {
        const history = b.log.filter(l => !l.startsWith('|poke|'));
        const at = history.findIndex(l => l.startsWith('|teampreview'));
        history.splice(at, 0, ...['p1', 'p2'].flatMap(side => item.preview[side].map(s => `|poke|${side}|${s}, L50|`)));
        const obs = new PublicObservation(history, b.dex, extractChannelMessages, true), actor = new Actor();
        let pos = b.log.length, guard = 0;
        const choose = side => side.activeRequest?.wait || !side.activeRequest ? 'pass' :
          (side.id === 'p1' ? actor.choose(obs.view(side), b.dex) : opponent(obs.view(side), style, b.dex))[0];
        while (!b.ended && b.turn <= 160) {
          if (++guard > 1400) throw Error('request guard exhausted');
          observedStep(b, choose(b.p1), choose(b.p2), step, choose);
          obs.append(b.log.slice(pos), true); pos = b.log.length;
        }
        row.outcome = !b.ended ? 'unfinished' : b.winner === 'p1' ? 'win' : b.winner === 'p2' ? 'loss' : 'draw';
      } catch (e) { row.outcome = 'failed'; row.error = String(e.stack || e); }
      finally { row.turns = b.turn; b.destroy(); }
      trials.push(row);
    }
  }
  fs.writeFileSync(out, JSON.stringify({scope: 'hypothetical sets; public actors; frozen public-reference-v2 opponent',
    trials}, null, 1) + '\n');
  console.log(item.record, 'done');
}
