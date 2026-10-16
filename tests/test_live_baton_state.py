import asyncio
import json

import pytest
from poke_env.battle import Effect
from poke_env.data import GenData
from test_worker_transitions import CHOMP, worker

from pokesim.config import FORMAT, ROOT
from pokesim.evaluation.shadow import position
from pokesim.live.session import BattleSession
from pokesim.rl.myteam import FixedTeam, to_id
from pokesim.search.lookahead import Lookahead
from pokesim.state.player import TrackedPlayer
from pokesim.state.tracker import TrackedBattle

SETS = json.loads((ROOT / 'data/my_teams/baton.json').read_text())
CASES = json.loads((ROOT / 'tests/fixtures/baton-received.json').read_text())


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['source'])
def test_real_baton_log_survives_the_next_request_and_search_reconstruction(case):
    session, battle = position(case['record'], SETS, case['turn'])
    active = battle.active_pokemon
    assert {s: n for s, n in active.boosts.items() if n} == case['boosts']
    assert (Effect.SUBSTITUTE in active.effects) == case['substitute']
    # The next actual decision, rather than just a simulated future pass, must retain the payload.
    battle = session.decision()
    engine = object.__new__(Lookahead)
    def base_id(sid):
        entry = GenData.from_gen(9).pokedex[sid]
        return to_id(entry.get('baseSpecies') if entry.get('forme', '').startswith('Mega') else sid)
    engine.sampler = type('Sampler', (), {'base_id': staticmethod(base_id)})()
    own = engine.own_side(battle, {to_id(s['species']): s for s in SETS})
    result = worker("""
const b=setUp(sample,format,1), m=b.p1.active[0];
console.log(JSON.stringify({boosts:m.boosts,substitute:!!m.volatiles.substitute,speed:m.getActionSpeed()}));
b.destroy();
""", {'p1': own, 'p2': {'team': [CHOMP], 'mons': [{}]}})
    assert {s: n for s, n in result['boosts'].items() if n} == case['boosts']
    assert result['substitute'] == case['substitute']
    assert result['speed'] == case['speed']


@pytest.mark.parametrize('side', ['p1', 'p2'])
def test_baton_copies_negative_stages_and_copyable_effects_but_not_move_locks(side):
    battle = TrackedBattle('protocol-test', 'me', None, gen=9)
    lines = ['|player|p1|me|', '|player|p2|foe|',
             f'|switch|{side}a: Scolipede|Scolipede, L50|100/100', '|turn|1',
             f'|-boost|{side}a: Scolipede|def|6', f'|-unboost|{side}a: Scolipede|atk|2',
             f'|-unboost|{side}a: Scolipede|accuracy|1', f'|-start|{side}a: Scolipede|Substitute',
             f'|-start|{side}a: Scolipede|confusion', f'|-start|{side}a: Scolipede|perish2',
             f'|move|{side}a: Scolipede|Baton Pass|{side}a: Scolipede',
             f'|-start|{side}a: Scolipede|Encore', f'|-start|{side}a: Scolipede|Disable|Baton Pass',
             f'|switch|{side}a: Gholdengo|Gholdengo, L50|100/100|[from] Baton Pass', '|turn|2']
    for line in lines:
        battle.parse_message(line.split('|'))
    active = battle.active_pokemon if side == 'p1' else battle.opponent_active_pokemon
    assert {s: n for s, n in active.boosts.items() if n} == {'def': 6, 'atk': -2, 'accuracy': -1}
    assert set(active.effects) == {Effect.SUBSTITUTE, Effect.CONFUSION, Effect.PERISH2}
    assert battle.state_tracker.locks(side, 'Gholdengo') == {}
    battle.parse_message(f'|switch|{side}a: Scolipede|Scolipede, L50|100/100'.split('|'))
    active = battle.active_pokemon if side == 'p1' else battle.opponent_active_pokemon
    assert not any(active.boosts.values()) and not active.effects


@pytest.mark.parametrize('ending', ['fail', 'cant', 'faint', 'turn', 'drag'])
def test_failed_baton_or_forced_switch_never_transfers_a_payload(ending):
    session = BattleSession(SETS)
    session.apply(['|switch|p1a: Scolipede|Scolipede, L50|100/165', '|turn|1',
                   '|-boost|p1a: Scolipede|spe|4', '|-start|p1a: Scolipede|Substitute',
                   '|move|p1a: Scolipede|Baton Pass|p1a: Scolipede'])
    event = {'fail': '|-fail|p1a: Scolipede', 'cant': '|cant|p1a: Scolipede|par',
             'faint': '|faint|p1a: Scolipede', 'turn': '|turn|2', 'drag': '|upkeep'}[ending]
    kind = 'drag' if ending == 'drag' else 'switch'
    session.apply([event, f'|{kind}|p1a: Gholdengo|Gholdengo, L50|175/175'])
    assert not any(session.battle.active_pokemon.boosts.values())
    assert not session.battle.active_pokemon.effects


def test_online_player_creates_the_same_tracked_battle_as_the_replay_session():
    class TestPlayer(TrackedPlayer):
        def choose_move(self, battle):
            return self.choose_default_move()

    player = TestPlayer(team=FixedTeam(SETS).yield_team(), battle_format=FORMAT, start_listening=False)
    async def create():
        battle = await player._create_battle(['>battle', FORMAT, 'adapter-test'])
        assert isinstance(battle, TrackedBattle)
        assert len(battle.teampreview_team) == len(SETS)
        assert await player._create_battle(['>battle', FORMAT, 'adapter-test']) is battle
    asyncio.run(create())


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['source'])
def test_online_protocol_and_next_real_request_retain_the_baton_payload(case):
    session, _ = position(case['record'], SETS, case['turn'])
    class Probe(TrackedPlayer):
        def choose_move(self, battle):
            raise AssertionError('No network move should be required in a protocol replay')

        async def _handle_battle_request(self, battle, maybe_default_order=False):
            self.captured = dict(battle.active_pokemon.boosts), set(battle.active_pokemon.effects)

    player = Probe(battle_format=FORMAT, start_listening=False)
    tag = f'>battle-{FORMAT}-baton-replay'
    lines = ['|init|battle', f'|player|p1|{player.username}|', '|player|p2|foe|',
             *case['record']['log'], '|request|' + json.dumps(session.request())]
    asyncio.run(player._handle_battle_message([[tag], *(line.split('|') for line in lines)]))
    boosts, effects = player.captured
    assert {s: n for s, n in boosts.items() if n} == case['boosts']
    assert (Effect.SUBSTITUTE in effects) == case['substitute']


def test_good_as_gold_encore_immunity_does_not_create_an_encore_lock():
    session = BattleSession(SETS)
    session.apply(['|switch|p1a: Gholdengo|Gholdengo, L50|175/175',
                   '|switch|p2a: Primarina|Primarina, L50|100/100', '|turn|1',
                   '|move|p1a: Gholdengo|Recover|p1a: Gholdengo',
                   '|move|p2a: Primarina|Encore|p1a: Gholdengo',
                   '|-immune|p1a: Gholdengo|[from] ability: Good as Gold', '|turn|2'])
    battle = session.decision()
    assert Effect.ENCORE not in battle.active_pokemon.effects
    assert session.battle.state_tracker.locks('p1', 'Gholdengo') == {}
    assert len(battle.available_moves) == 4
