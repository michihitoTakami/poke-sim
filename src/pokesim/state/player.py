"""poke-env singles adapter with the same protocol state as offline BattleSession."""
from poke_env.battle import Pokemon
from poke_env.data import GenData
from poke_env.player import Player
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from pokesim.state.tracker import TrackedBattle


class TrackedPlayer(Player):
    async def _create_battle(self, split_message):
        # poke-env has no Battle factory hook. Keep its singles registration/notification sequence here.
        # Other formats retain the upstream implementation; this application uses Champions singles.
        if self.format_is_doubles or self.format_is_bestof or self.format != split_message[1]:
            return await super()._create_battle(split_message)
        tag = '-'.join(split_message)[1:]
        if tag in self._battles:
            return self._battles[tag]
        gen = GenData.from_format(self.format).gen
        battle = TrackedBattle(tag, self.username, self.logger, gen=gen, save_replays=self._save_replays)
        if isinstance(self._team, ConstantTeambuilder):
            battle.teampreview_team = [Pokemon(gen=gen, teambuilder=mon) for mon in self._team.team]
        await self._battle_count_queue.put(None)
        if tag in self._battles:
            await self._battle_count_queue.get()
            return self._battles[tag]
        async with self._battle_start_condition:
            self._battle_semaphore.release()
            self._battle_start_condition.notify_all()
            self._battles[tag] = battle
        if self._start_timer_on_battle_start:
            await self.ps_client.send_message('/timer on', battle.battle_tag)
        return battle
