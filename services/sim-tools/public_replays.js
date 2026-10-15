// Public replay states for auxiliary value pretraining. No set reconstruction, requests, future moves or winner input.
const path = require('path');
const readline = require('readline');
const {Dex} = require(path.resolve(process.argv[2], 'dist/sim'));
const {extractChannelMessages} = require(path.resolve(process.argv[2], 'dist/sim/battle'));
const {PublicObservation} = require('./public_observation');

function snapshots(req) {
  const dex = Dex.forFormat(req.format);
  const lines = extractChannelMessages(req.log, [0])[0];
  const at = lines.findIndex(line => /^\|turn\|\d+$/.test(line));
  if (at < 0) throw new Error('no turn boundary');
  const observer = new PublicObservation(lines.slice(0, at + 1), dex, extractChannelMessages,
    dex.currentMod.startsWith('champions'));
  const states = [JSON.parse(JSON.stringify(observer.state))];
  let chunk = [];
  for (const line of lines.slice(at + 1)) {
    if (/^\|(win|tie)(\||$)/.test(line)) break;
    chunk.push(line);
    if (/^\|turn\|\d+$/.test(line)) {
      observer.append(chunk);
      states.push(JSON.parse(JSON.stringify(observer.state)));
      // The output is structured public state only; no chat, usernames or growing history is exported.
      observer.history = [];
      chunk = [];
    }
  }
  return states;
}

const rl = readline.createInterface({input: process.stdin});
rl.on('line', line => {
  try { process.stdout.write(JSON.stringify({states: snapshots(JSON.parse(line))}) + '\n'); }
  catch (e) { process.stdout.write(JSON.stringify({error: String(e.message || e)}) + '\n'); }
});
