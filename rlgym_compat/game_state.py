from typing import List, Optional

import numpy as np
from rlbot.flat import AirState, FieldInfo, GameTickPacket, PlayerInfo

from .physics_object import PhysicsObject
from .player_data import PlayerData


class GameState:
    def __init__(self, field_info: FieldInfo, tick_skip=8):
        self.game_type: int = 0
        self.blue_score = 0
        self.orange_score = 0
        self.last_touch: Optional[int] = -1

        self.players: List[PlayerData] = []
        self._total_players = 64
        self._ticks_since_jump = np.zeros(self._total_players)
        self._air_time_since_jump_ended = np.zeros(self._total_players)
        self._used_double_jump_or_flip = [False for _ in range(self._total_players)]
        self._last_ball_touched_time = np.zeros(self._total_players)

        self.ball: PhysicsObject = PhysicsObject()
        self.inverted_ball: PhysicsObject = PhysicsObject()

        # List of "booleans" (1 or 0)
        self.boost_pads: np.ndarray = np.zeros(
            len(field_info.boost_pads), dtype=np.float32
        )
        self.inverted_boost_pads: np.ndarray = np.zeros_like(
            self.boost_pads, dtype=np.float32
        )
        self.last_frame_num = 0
        self.tick_skip_time = tick_skip / 120

    def decode(self, packet: GameTickPacket):
        # Increase number of players' persistent data we are tracking, if necessary
        while self._total_players < len(packet.players):
            new_ticks_since_jump = np.zeros(2 * self._total_players)
            new_ticks_since_jump[: self._total_players] = self._ticks_since_jump
            self._ticks_since_jump = new_ticks_since_jump

            new_air_time_since_jump_ended = np.zeros(2 * self._total_players)
            new_air_time_since_jump_ended[: self._total_players] = (
                self._air_time_since_jump_ended
            )
            self._air_time_since_jump_ended = new_air_time_since_jump_ended

            new_last_ball_touched_time = np.zeros(2 * self._total_players)
            new_last_ball_touched_time[: self._total_players] = (
                self._last_ball_touched_time
            )
            self._last_ball_touched_time = new_last_ball_touched_time

            new_used_double_jump_or_flip = [
                False for _ in range(2 * self._total_players)
            ]
            new_used_double_jump_or_flip[: self._total_players] = (
                self._used_double_jump_or_flip
            )
            self._used_double_jump_or_flip = new_used_double_jump_or_flip

            self._total_players *= 2

        # Get number of ticks elapsed since last decode() call
        ticks_elapsed = 0
        if self.last_frame_num > 0:
            ticks_elapsed = packet.game_info.frame_num - self.last_frame_num
        self.last_frame_num = packet.game_info.frame_num

        # Set score
        self.blue_score = packet.teams[0].score
        self.orange_score = packet.teams[1].score

        # Set boost pads
        for i, pad in enumerate(packet.boost_pad_states):
            self.boost_pads[i] = pad.is_active
        self.inverted_boost_pads[:] = self.boost_pads[::-1]

        # Set ball
        ball = packet.balls[0]
        self.ball.decode_ball_data(ball.physics)
        self.inverted_ball.invert(self.ball)

        # Set up touch tracking
        latest_touch = ball.latest_touch
        if latest_touch.game_seconds > 0:
            self.last_touch = latest_touch.player_index
        self._last_ball_touched_time[latest_touch.player_index] = (
            latest_touch.game_seconds
        )

        # Set players
        self.players = []
        for i, player in enumerate(packet.players):
            player_data = self._decode_player(player, i, ticks_elapsed)
            # Need to set player data ball touched only if player has touched ball in the last tick_skip ticks
            if (
                self._last_ball_touched_time[i] > 0
                and packet.game_info.seconds_elapsed - self._last_ball_touched_time
                < self.tick_skip_time
            ):
                player_data.ball_touched = True

            self.players.append(player)

    def _decode_player(
        self, player_info: PlayerInfo, index: int, ticks_elapsed: int
    ) -> PlayerData:
        player_data = PlayerData()

        match player_info.air_state:
            case AirState.OnGround:
                if self._ticks_since_jump[index] > 0:
                    self._ticks_since_jump[index] += ticks_elapsed
                if self._ticks_since_jump[index] > 6:
                    # We must really be on ground
                    self._ticks_since_jump[index] = 0
                    player_data.on_ground = True
                self._air_time_since_jump_ended[index] = 0
            case AirState.Jumping:
                self._ticks_since_jump[index] += ticks_elapsed
                self._air_time_since_jump_ended[index] = 0
                if self._ticks_since_jump[index] > 6:
                    # We cannot be on the ground (excluding some really weird circumstances)
                    player_data.on_ground = False
            case AirState.InAir:
                player_data.on_ground = False
                if self._air_time_since_jump_ended[index] > 0:
                    self._air_time_since_jump_ended += ticks_elapsed
                self._air_time_since_jump_ended[index] += ticks_elapsed
                if (
                    player_info.jumped
                ):  # Technically this should only start when you stop holding jump
                    self._air_time_since_jump_ended[index] += ticks_elapsed
            case AirState.DoubleJumping:
                self._used_double_jump_or_flip[index] = True
                player_data.on_ground = False
            case AirState.Dodging:
                self._used_double_jump_or_flip[index] = True
                player_data.on_ground = False

        player_data.car_id = index
        player_data.team_num = player_info.team
        player_data.match_goals = player_info.score_info.goals
        player_data.match_saves = player_info.score_info.saves
        player_data.match_shots = player_info.score_info.shots
        player_data.match_demolishes = player_info.score_info.demolitions
        if (
            player_data.boost_amount < player_info.boost / 100
        ):  # This isn't perfect but with decent fps it'll work
            if player_data.boost_pickups == -1:
                player_data.boost_pickups = 1
            else:
                player_data.boost_pickups += 1
        player_data.is_demoed = player_info.demolished_timeout > 0
        player_data.ball_touched = False
        player_data.has_jump = self._ticks_since_jump[index] == 0
        player_data.has_flip = (
            self._air_time_since_jump_ended[index] < 150
            and not self._used_double_jump_or_flip[index]
        )
        player_data.boost_amount = player_info.boost / 100
        player_data.car_data.decode_car_data(player_info.physics)
        player_data.inverted_car_data.invert(player_data.car_data)

        return player_data
