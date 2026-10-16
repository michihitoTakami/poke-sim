// Probabilities of an exported policy net for views given as JSON lines ({view, legal}) on stdin.
//   node services/sim-tools/policy_net_dump.js <showdown_dir> <format> <net.json> < views.jsonl
const path = require('path');
const [dir, format, net] = process.argv.slice(2);
const {Dex} = require(path.resolve(dir, 'dist/sim'));
const dex = Dex.forFormat(Dex.formats.get(format, true));
const {loadPolicyNet} = require('./policy_net');
const model = loadPolicyNet(net);
for (const line of require('fs').readFileSync(0, 'utf8').split('\n').filter(Boolean)) {
  const {view, legal} = JSON.parse(line);
  process.stdout.write(JSON.stringify(model.prior(view, dex, legal)) + '\n');
}
