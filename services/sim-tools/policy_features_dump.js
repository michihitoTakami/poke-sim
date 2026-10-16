// Dump state/action vectors for views given as JSON lines ({view, legal}) on stdin: the parity check's JS side.
//   node services/sim-tools/policy_features_dump.js <showdown_dir> <format> < views.jsonl
const path = require('path');
const [dir, format] = process.argv.slice(2);
const {Dex} = require(path.resolve(dir, 'dist/sim'));
const dex = Dex.forFormat(Dex.formats.get(format, true));
const pf = require('./policy_features');
const schema = require('./policy_features.schema.json');
const lines = require('fs').readFileSync(0, 'utf8').split('\n').filter(Boolean);
for (const line of lines) {
  const {view, legal} = JSON.parse(line);
  const state = Array.from(pf.vector(pf.stateFeatures(view, dex), schema.state));
  const actions = legal.map(([choice, label]) => Array.from(pf.vector(pf.actionFeatures(view, dex, choice, label), schema.action)));
  process.stdout.write(JSON.stringify({state, actions}) + '\n');
}
