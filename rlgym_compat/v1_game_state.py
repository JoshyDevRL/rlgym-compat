from typing import Dict, List, Optional

import numpy as np
from rlbot.flat import (
    FieldInfo,
    GameStateType,
    GameTickPacket,
    MatchSettings,
    PlayerInfo,
)

from .common_values import BLUE_TEAM, ORANGE_TEAM
from .extra_info import ExtraPacketInfo
from .game_state import GameState
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
        self.game_type = int(match_settings.game_mode)
        self.blue_score = 0
        self.orange_score = 0
        self.last_touch: Optional[int] = -1
        self._player_infos: Dict[int, PlayerInfo] = {}
        self._boost_pickups: Dict[int, int] = {}
        self._players: Optional[List[V1PlayerData]] = None

    @property
    def players(self):
        if self._players is None:
            players: List[V1PlayerData] = []
            for spawn_idx, car in self._game_state.cars.items():
                players.append(
                    V1PlayerData.create_from_v2(
                        car,
                        self._player_infos[spawn_idx],
                        spawn_idx,
                        self._boost_pickups[spawn_idx],
                    )
                )

            return players
        return self._players

    @players.setter
    def players(self, value: List[V1PlayerData]):
        self._players = value

    @property
    def ball(self):
        return V1PhysicsObject.create_from_v2(self._game_state.ball)

    @property
    def inverted_ball(self):
        return V1PhysicsObject.create_from_v2(self._game_state.inverted_ball)

    @property
    def boost_pads(self):
        return (self._game_state.boost_pad_timers == 0).astype(np.float32)

    @property
    def inverted_boost_pads(self):
        return (self._game_state.inverted_boost_pad_timers == 0).astype(np.float32)

    def update(
        self, packet: GameTickPacket, extra_info: Optional[ExtraPacketInfo] = None
    ):
        self._players = None
        self.blue_score = packet.teams[BLUE_TEAM].score
        self.orange_score = packet.teams[ORANGE_TEAM].score
        if len(packet.balls) > 0:
            ball = packet.balls[0]
            if ball.latest_touch.player_name:
                self.last_touch = packet.balls[0].latest_touch.player_index
        old_boost_amounts = {
            **{p.spawn_id: p.boost / 100 for p in packet.players},
            **{k: v.boost_amount for (k, v) in self._game_state.cars.items()},
        }
        self._game_state.update(packet, extra_info)
        for player_info in packet.players:
            if player_info.spawn_id not in self._boost_pickups:
                self._boost_pickups[player_info.spawn_id] = 0

            if (
                packet.game_info.game_state_type
                in (GameStateType.Active, GameStateType.Kickoff)
                and old_boost_amounts[player_info.spawn_id] < player_info.boost / 100
            ):  # This isn't perfect but with decent fps it'll work
                self._boost_pickups[player_info.spawn_id] += 1
            self._player_infos[player_info.spawn_id] = player_info
