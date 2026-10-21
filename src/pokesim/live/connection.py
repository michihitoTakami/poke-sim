"""Observe matchmaking separately from authentication and an active battle, without logging credentials."""
import json
import time
from pathlib import Path


class LadderConnection:
    def __init__(self, client, format_id, *, clock=time.monotonic):
        self.client, self.format_id, self.clock = client, format_id, clock
        self.started = self.search_started = clock()
        self.challenge_seen = False
        self.search_confirmed = False
        self.blocked = None
        original = client._handle_message

        async def handle(message):
            self.observe(message)
            if message.startswith('>'):
                await original(message)
            else:
                # poke-env dispatches global packets by their first line. A bundled authentication reply,
                # search rejection or restart notice must not disappear behind another global message.
                for line in message.splitlines():
                    if line.startswith('|'):
                        await original(line)

        client._handle_message = handle

    def observe(self, message):
        # Never retain raw packets: challstr, login assertions and private requests are not diagnostics.
        if message.startswith('>'):
            return
        for line in message.splitlines():
            parts = line.split('|', 2)
            if len(parts) < 3:
                continue
            kind, payload = parts[1:]
            if kind == 'challstr':
                self.challenge_seen = True
            elif kind == 'updatesearch':
                try:
                    self.search_confirmed = self.format_id in json.loads(payload).get('searching', [])
                except (ValueError, TypeError, AttributeError):
                    pass
            elif kind == 'popup':
                text = payload.lower()
                if 'restart' in text:
                    self.blocked = 'server-restarting'
                elif any(word in text for word in ('rejected', 'not found', 'not allowed', 'banned', 'invalid')):
                    self.blocked = 'server-search-rejected'
            elif kind in {'nametaken', 'error'}:
                self.blocked = 'authentication-or-search-rejected'

    def begin_search(self):
        self.search_started = self.clock()
        self.search_confirmed = False

    def state(self, player):
        if any(not battle.finished for battle in player.battles.values()):
            return 'battling'
        if self.blocked:
            return self.blocked
        if not self.client.logged_in.is_set():
            return 'authenticating' if self.challenge_seen else 'connecting'
        return 'searching' if self.search_confirmed else 'awaiting-search-confirmation'

    def failure(self, player, *, login_timeout=60., search_timeout=90., battle_timeout=600.):
        state, now = self.state(player), self.clock()
        if state == 'battling':
            return 'battle-inactive' if now - player.last_activity > battle_timeout else None
        listener = getattr(self.client, '_listening_coroutine', None)
        if listener is not None and listener.done():
            return 'connection-closed'
        if self.blocked:
            return self.blocked
        if not self.client.logged_in.is_set():
            return 'login-timeout' if now - self.started > login_timeout else None
        if not self.search_confirmed and now - self.search_started > search_timeout:
            return 'search-not-confirmed'
        # Confirmed matchmaking on a live connection can legitimately wait for an opponent. Cancelling it
        # every ten minutes followed by a thirty-minute sleep prevents the collector from staying queued.
        return None

    def publish(self, player, path: Path | None):
        status = {'phase': self.state(player), 'logged_in': self.client.logged_in.is_set(),
                  'search_confirmed': self.search_confirmed, 'format': self.format_id}
        if status != getattr(self, '_published', None):
            print('connection: ' + json.dumps(status), flush=True)
            self._published = status
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.json.tmp')
            temp.write_text(json.dumps(dict(status, updated_at=time.time())) + '\n')
            temp.replace(path)
