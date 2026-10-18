import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from test_expert_continuation import GHOLDENGO

from pokesim.config import FORMAT
from pokesim.live.advisor import Suggestion
from pokesim.live.session import BattleSession
from scripts.ladder import AdvisorPlayer


@pytest.fixture
def state(tmp_path):
    session = BattleSession([GHOLDENGO])
    session.preview(['Primarina'])
    session.select([0])
    session.apply(['|switch|p1a: Gholdengo|Gholdengo, L50|175/175',
                   '|switch|p2a: Primarina|Primarina, L50|100/100', '|turn|1'])
    battle = session.decision()
    player = object.__new__(AdvisorPlayer)
    player.out = tmp_path
    player.advice, player.logs = {}, {battle.battle_tag: list(session.log)}
    player.sent_requests, player.retried_requests = {}, {}
    player._battles = {battle.battle_tag: battle}
    player._format = FORMAT
    player._strict_battle_tracking = False
    player._accept_open_team_sheet = False
    player._waiting = asyncio.Event()
    player.value_logged, player.quality_issues = set(), []
    player.sample = False
    player.activate_policy = lambda path: None
    player.policy_for = lambda battle: ''
    player.advisor = SimpleNamespace(
        engine=SimpleNamespace(value_model=object()),
        advise=Mock(return_value=Suggestion([('move:shadowball', 1.0)], elapsed=.01, status={})))
    player.ps_client = SimpleNamespace(send_message=AsyncMock())
    return player, session, battle


def diagnostics(player):
    return [json.loads(line) for line in (player.out / 'diagnostics.jsonl').read_text().splitlines()]


def test_finished_battle_does_not_recreate_advice_or_run_search(state):
    player, _, battle = state
    battle.won_by('me')
    player.logs.clear()  # the end callback has already finalized this record
    assert asyncio.run(player.choose_move(battle)) is None
    asyncio.run(player._handle_battle_request(battle, maybe_default_order=True))
    player.advisor.advise.assert_not_called()
    player.ps_client.send_message.assert_not_called()
    assert not player.advice and not player.quality_issues
    assert diagnostics(player)[0]['kind'] == 'discarded-request'


@pytest.mark.parametrize('change', ['finished', 'request', 'wait'])
def test_stale_search_result_cannot_send_or_trigger_quality_stop(state, change):
    player, session, battle = state
    def advise(*args):
        if change == 'finished':
            battle.won_by('me')
        elif change == 'wait':
            battle._wait = True
        else:
            battle.parse_request(session.request())
        return Suggestion([('move:shadowball', 1)], elapsed=11, status={'fallback': 'stale failure'})
    player.advisor.advise.side_effect = advise
    asyncio.run(player._handle_battle_request(battle))
    player.ps_client.send_message.assert_not_called()
    assert not player.quality_issues
    assert any(d['kind'] == 'discarded-result' for d in diagnostics(player))


def test_a_current_choice_includes_rqid_and_duplicates_are_ignored(state):
    player, _, battle = state
    asyncio.run(player._handle_battle_request(battle))
    asyncio.run(player._handle_battle_request(battle))
    key = battle.last_request['rqid']
    player.ps_client.send_message.assert_awaited_once_with(f'/choose move shadowball|{key}', battle.battle_tag)
    entry = player.advice[battle.battle_tag][0]
    assert entry['chosen'] == 'move:shadowball'
    assert entry['order'] == '/choose move shadowball'
    assert entry['sent'] == f'/choose move shadowball|{key}'
    assert entry['active']['species'] == 'gholdengo'


def test_old_request_is_ignored_before_it_can_revert_the_battle_state(state):
    player, session, battle = state
    old = session.request()
    battle.parse_request(session.request())
    current = battle.last_request
    old['side']['pokemon'][0]['condition'] = '1/175'
    packet = [['>' + battle.battle_tag], ['', 'request', json.dumps(old)]]
    asyncio.run(player._handle_battle_message(packet))
    assert battle.last_request is current and battle.active_pokemon.current_hp == 175
    player.advisor.advise.assert_not_called()


def test_same_request_id_does_not_send_twice_even_if_a_turn_marker_has_changed(state):
    player, _, battle = state
    asyncio.run(player._handle_battle_request(battle))
    battle.parse_message(['', 'turn', '2'])
    asyncio.run(player._handle_battle_request(battle))
    assert player.ps_client.send_message.await_count == 1


def test_rejected_current_choice_has_one_recorded_default_retry(state):
    player, _, battle = state
    async def request():
        await player._handle_battle_request(battle)
        await player._handle_battle_request(battle, maybe_default_order=True)
        await player._handle_battle_request(battle, maybe_default_order=True)
    asyncio.run(request())
    assert player.ps_client.send_message.await_count == 2
    assert player.quality_issues == ['choice rejected']
    assert player.advice[battle.battle_tag][-1]['retry'] is True
    assert any(d['kind'] == 'quality-failure' for d in diagnostics(player))


@pytest.mark.parametrize('message', ["[Invalid choice] There's nothing to choose",
    '[Invalid choice] Sorry, too late to make a different move; the next turn has already started'])
def test_late_choice_error_does_not_start_another_search(state, message):
    player, _, battle = state
    packet = [['>' + battle.battle_tag], ['', 'error', message]]
    asyncio.run(player._handle_battle_message(packet))
    player.advisor.advise.assert_not_called()
    player.ps_client.send_message.assert_not_called()
    assert not player.quality_issues


def test_post_finish_packet_does_not_recreate_a_finalized_log(state):
    player, _, battle = state
    battle.won_by('me')
    player.logs.clear()
    packet = [['>' + battle.battle_tag], ['', 'request', json.dumps(battle.last_request)],
              ['', 'error', "[Invalid choice] There's nothing to choose"]]
    asyncio.run(player._handle_battle_message(packet))
    assert not player.logs and not player.advice and not player.quality_issues
    player.advisor.advise.assert_not_called()


def test_advisor_receives_only_the_processed_prefix_of_a_protocol_packet(state):
    player, session, battle = state
    before = list(player.logs[battle.battle_tag])
    packet = [['>' + battle.battle_tag], ['', 'request', json.dumps(session.request())],
              ['', 'move', 'p2a: Primarina', 'Calm Mind', 'p2a: Primarina']]
    asyncio.run(player._handle_battle_message(packet))
    assert player.advisor.advise.call_args.args[2] == before
    assert player.logs[battle.battle_tag][-1] == '|move|p2a: Primarina|Calm Mind|p2a: Primarina'


def test_active_game_fallback_still_stops_collection_and_persists_its_full_reason(state):
    player, _, battle = state
    player.advisor.advise.return_value.status = {'fallback': 'worker failed while active'}
    asyncio.run(player._handle_battle_request(battle))
    assert player.quality_issues == ['search fallback']
    failure = next(d for d in diagnostics(player) if d['kind'] == 'quality-failure')
    assert failure['battle'] == battle.battle_tag
    assert failure['advice']['status']['fallback'] == 'worker failed while active'
    assert failure['rqid'] == battle.last_request['rqid']


def test_delayed_move_error_cannot_reject_the_current_switch_order(state):
    player, _, battle = state
    rqid = battle.last_request['rqid']
    player.advice[battle.battle_tag] = [{'rqid': rqid, 'sent': f'/choose switch Skeledirge|{rqid}'}]
    packet = [['>' + battle.battle_tag], ['', 'error',
               "[Invalid choice] Can't move: You need a switch response"]]
    asyncio.run(player._handle_battle_message(packet))
    player.ps_client.send_message.assert_not_called()
    assert not player.quality_issues
    assert diagnostics(player)[-1]['reason'] == 'move-error-after-current-switch-command'
