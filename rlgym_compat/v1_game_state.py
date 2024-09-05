from typing import Dict, Optional

import numpy as np
from rlbot.flat import FieldInfo, GameTickPacket, MatchSettings, Physics, PlayerInfo

from .common_values import BLUE_TEAM, ORANGE_TEAM
from .game_state import GameState
from .physics_object import PhysicsObject
from .v1.physics_object import PhysicsObject as V1PhysicsObject
from .v1.player_data import PlayerData as V1PlayerData


class V1GameState:
    def __init__(
        self,
        field_info: FieldInfo,
        match_settings=MatchSettings(),
        tick_skip=8,
        standard_map=True,
    ):
        self._game_state = GameState.create_compat_game_state(
            field_info, match_settings, tick_skip, standard_map
        )
        self.blue_score = 0
        self.orange_score = 0
        self.last_touch: Optional[int] = -1
        self._player_infos: Dict[int, PlayerInfo] = {}
        self._boost_pickups: Dict[int, int] = {}

    @property
    def players(self):
        players = []
        for car_id, (spawn_idx, car) in enumerate(self._game_state.cars.items()):
            players.append(
                V1PlayerData.create_from_v2(
                    car,
                    self._player_infos[spawn_idx],
                    car_id,
                    self._boost_pickups[spawn_idx],
                )
            )

        return players

    @players.setter
    def players(self, value):
        raise NotImplementedError

    @property
    def ball(self):
        return V1PhysicsObject.create_from_v2(self._game_state.ball)

    @ball.setter
    def ball(self, value):
        raise NotImplementedError

    @property
    def inverted_ball(self):
        return V1PhysicsObject.create_from_v2(self._game_state.inverted_ball)

    @inverted_ball.setter
    def inverted_ball(self, value):
        raise NotImplementedError

    @property
    def boost_pads(self):
        return self._game_state.boost_pad_timers == 0

    @boost_pads.setter
    def boost_pads(self, value):
        raise NotImplementedError

    @property
    def inverted_boost_pads(self):
        return self._game_state.inverted_boost_pad_timers == 0

    @inverted_boost_pads.setter
    def inverted_boost_pads(self, value):
        raise NotImplementedError

    def update(self, packet: GameTickPacket):
        self.blue_score = packet.teams[BLUE_TEAM].score
        self.orange_score = packet.teams[ORANGE_TEAM].score
        if len(packet.balls) > 0:
            self.last_touch = packet.balls[0].latest_touch.player_index
        for player_info in packet.players:
            if player_info.spawn_id not in self._boost_pickups:
                self._boost_pickups[player_info.spawn_id] = 0
            if (
                self._game_state.cars[player_info.spawn_id].boost_amount
                < player_info.boost / 100
            ):  # This isn't perfect but with decent fps it'll work
                self._boost_pickups[player_info.spawn_id] += 1
            self._player_infos[player_info.spawn_id] = player_info
        self._game_state.update(packet)
