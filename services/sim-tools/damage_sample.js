// Sample real damage from the Showdown sim: p1 uses `move` on turn 1, p2 uses a status move.
// Usage: node damage_sample.js <showdown_dir> <format> '<attacker set json>' '<defender set json>' <move> <n>
const path = require('path');
const [dir, format, atkJson, defJson, move, n] = process.argv.slice(2);
const { BattleStream, getPlayerStreams, Teams } = require(path.resolve(dir, 'dist/sim'));

async function once(seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const out = [];
  (async () => { for await (const chunk of streams.omniscient) out.push(chunk); })();
  const atk = JSON.parse(atkJson), def = JSON.parse(defJson);
  streams.omniscient.write(`>start {"formatid":"${format}","seed":[${seed},${seed + 1},${seed + 2},${seed + 3}]}`);
  streams.omniscient.write(`>player p1 {"name":"a","team":"${Teams.pack([atk, atk, atk])}"}`);
  streams.omniscient.write(`>player p2 {"name":"b","team":"${Teams.pack([def, def, def])}"}`);
  streams.omniscient.write(`>p1 team 123`);
  streams.omniscient.write(`>p2 team 123`);
  streams.omniscient.write(`>p1 move ${move}`);
  streams.omniscient.write(`>p2 move ${def.moves[0]}`);
  await new Promise(r => setTimeout(r, 30));
  if (process.env.DEBUG) console.error(out.join("\n").slice(-1500));
  const log = out.join('\n');
  const m = log.match(/\|-damage\|p2a: [^|]+\|(\d+)\/(\d+)/);
  return m ? 1 - Number(m[1]) / Number(m[2]) : null;
}
(async () => {
  const xs = [];
  for (let i = 0; i < Number(n); i++) { const d = await once(i * 7 + 1); if (d !== null) xs.push(d); }
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  console.log(JSON.stringify({ mean, min: Math.min(...xs), n: xs.length }));
})();
