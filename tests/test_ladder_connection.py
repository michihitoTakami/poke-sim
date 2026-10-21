import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pokesim.live.connection import LadderConnection


def connection():
    now = [0.]
    client = SimpleNamespace(logged_in=asyncio.Event(), _handle_message=AsyncMock(),
                             _listening_coroutine=SimpleNamespace(done=lambda: False))
    player = SimpleNamespace(battles={}, last_activity=0.)
    health = LadderConnection(client, 'gen9championsbssregmc', clock=lambda: now[0])
    return now, client, player, health


def test_failed_authentication_is_detected_without_waiting_ten_minutes():
    now, client, player, health = connection()
    asyncio.run(client._handle_message('|challstr|4|not-a-real-challenge'))
    assert health.state(player) == 'authenticating'
    now[0] = 61
    assert health.failure(player) == 'login-timeout'


def test_confirmed_matchmaking_stays_queued_even_without_battle_messages():
    now, client, player, health = connection()
    client.logged_in.set()
    health.observe('|updatesearch|{"searching":["gen9championsbssregmc"],"games":{}}')
    now[0] = 3600
    assert health.state(player) == 'searching'
    assert health.failure(player) is None
    health.observe('|updatesearch|{"searching":[],"games":{}}')
    assert health.failure(player) == 'search-not-confirmed'


def test_server_restart_is_retried_after_finishing_the_active_battle():
    now, client, player, health = connection()
    client.logged_in.set()
    battle = SimpleNamespace(finished=False)
    player.battles['room'] = battle
    health.observe('|popup|The server is restarting. Battles will be available again in a few minutes.')
    assert health.failure(player) is None
    battle.finished = True
    assert health.failure(player) == 'server-restarting'


def test_closed_listener_and_unconfirmed_search_are_distinct():
    now, client, player, health = connection()
    client.logged_in.set()
    now[0] = 91
    assert health.failure(player) == 'search-not-confirmed'
    client._listening_coroutine.done = lambda: True
    assert health.failure(player) == 'connection-closed'


def test_battle_timeout_uses_activity_instead_of_total_game_length():
    now, client, player, health = connection()
    player.battles['room'] = SimpleNamespace(finished=False)
    now[0] = 3600
    player.last_activity = 3590
    assert health.failure(player) is None
    player.last_activity = 2900
    assert health.failure(player) == 'battle-inactive'


def test_connection_diagnostics_never_store_challenges_or_private_packets(tmp_path):
    _, client, player, health = connection()
    health.observe('|challstr|4|secret-challenge')
    health.observe('>battle-room\n|request|{"private":"secret-team"}')
    path = tmp_path / 'connection.json'
    health.publish(player, path)
    saved = path.read_text()
    assert 'secret' not in saved and json.loads(saved)['phase'] == 'authenticating'


def test_bundled_global_messages_are_all_dispatched_but_battle_batches_stay_atomic():
    callback = AsyncMock()
    other = SimpleNamespace(logged_in=asyncio.Event(), _handle_message=callback)
    LadderConnection(other, 'gen9championsbssregmc')
    asyncio.run(other._handle_message('|updatesearch|{"searching":[]}\n|popup|Your team was rejected.'))
    assert [call.args[0] for call in callback.await_args_list] == [
        '|updatesearch|{"searching":[]}', '|popup|Your team was rejected.']
    callback.reset_mock()
    packet = '>battle-room\n|turn|1\n|request|{}'
    asyncio.run(other._handle_message(packet))
    callback.assert_awaited_once_with(packet)


def test_informational_popup_does_not_abort_matchmaking():
    _, client, player, health = connection()
    client.logged_in.set()
    health.observe('|popup|Welcome back!')
    assert health.failure(player) is None
