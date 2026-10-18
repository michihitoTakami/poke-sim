// Offline-only deterministic driver: bootstrap every cell, then complete a fixed number of trials.
const fs = require('fs'), path = require('path'), vm = require('vm'), readline = require('readline');
const [showdown, root] = process.argv.slice(2);
const file = path.join(root, 'services/sim-tools/lookahead.js');
const sandbox = {__dirname: path.dirname(file),
  require: n => n === 'readline' ? {createInterface: () => ({on: () => {}})} : require(n),
  process: {argv: ['node', 'offline', showdown]}, module: {exports: {}}, console};
vm.runInNewContext(fs.readFileSync(file, 'utf8') + '\nmodule.exports={setUp,step,choices,State,extractChannelMessages};', sandbox);
const {Search} = require(path.join(root, 'services/sim-tools/terminal_search'));
const {observedStep} = require(path.join(root, 'services/sim-tools/public_policy'));
let search;
readline.createInterface({input: process.stdin}).on('line', line => {
  try {
    const req = JSON.parse(line), pending = [];
    const complete = (job, initial) => { for (const _ of search.trial(job, initial, pending)) {} };
    if (req.op === 'start') {
      search = new Search(req.request, {...sandbox.module.exports, observedStep});
      for (const job of search.jobs) complete(job, true);
    } else {
      search.feedback(req.feedback);
      for (let i = 0; i < (req.trials || 0); i++) complete(search.pickJob(), false);
    }
    console.log(JSON.stringify(search.reply(pending)));
  } catch (e) { console.log(JSON.stringify({error: String(e.stack || e)})); }
});
