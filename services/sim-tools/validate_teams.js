// Validate teams (JSON array of PokemonSet arrays on stdin) for a format; prints [{ok, problems, packed}] as JSON.
// Usage: node services/sim-tools/validate_teams.js <showdown_dir> <format_id> < teams.json
const path = require('path');
const [showdownDir, formatId] = process.argv.slice(2);
const { TeamValidator, Teams } = require(path.resolve(showdownDir, 'dist/sim'));

let input = '';
process.stdin.on('data', chunk => { input += chunk; });
process.stdin.on('end', () => {
  const validator = TeamValidator.get(formatId);
  const results = JSON.parse(input).map(team => {
    const problems = validator.validateTeam(team) || [];
    return { ok: problems.length === 0, problems, packed: Teams.pack(team) };
  });
  process.stdout.write(JSON.stringify(results));
});
