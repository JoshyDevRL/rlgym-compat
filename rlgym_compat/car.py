from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rlbot.flat import AirState, BoxShape, GameTickPacket, PlayerInfo, Touch, Vector3

from .common_values import (
    BLUE_TEAM,
    BREAKOUT,
    DOMINUS,
    DOUBLEJUMP_MAX_DELAY,
    FLIP_TORQUE_TIME,
    HYBRID,
    JUMP_RESET_TIME_PAD,
    MERC,
    MIN_BOOST_TIME,
    MIN_JUMP_TIME,
    OCTANE,
    ORANGE_TEAM,
    PLANK,
    POWERSLIDE_FALL_RATE,
    POWERSLIDE_RISE_RATE,
    TICK_TIME,
    TICKS_PER_SECOND,
)
from .physics_object import PhysicsObject
from .utils import compare_hitbox_shape, create_default_init


@dataclass(init=False)
class Car:

    # Misc Data
    team_num: int
    hitbox_type: int
    ball_touches: int  # number of ball touches since last state was sent
    bump_victim_id: Optional[int]

    # Actual State
    demo_respawn_timer: float  # 0 if alive
    # TODO add num_wheels_contact when it's available in rsim
    # num_wheels_contact: int  # Needed for stuff like AutoRoll and some steering shenanigans
    on_ground: bool  # this is just numWheelsContact >=3 TODO make property when num_w_cts is available
    supersonic_time: float  # greater than 0 when supersonic, needed for state set since ssonic threshold changes with time
    boost_amount: float
    boost_active_time: float  # you're forced to boost for at least 12 ticks
    handbrake: float

    # Jump Stuff
    has_jumped: bool
    is_holding_jump: bool  # whether you pressed jump last tick or not
    is_jumping: bool  # changes to false after max jump time
    jump_time: float  # need jump time for state set, doesn't reset to 0 because of psyonix's landing jump cooldown

    # Flip Stuff
    has_flipped: bool
    has_double_jumped: bool
    air_time_since_jump: float
    flip_time: float
    flip_torque: np.ndarray

    # AutoFlip Stuff - What helps you recover from turtling
    is_autoflipping: bool
    autoflip_timer: float
    autoflip_direction: float  # 1 or -1, determines roll direction

    physics: PhysicsObject
    _inverted_physics: PhysicsObject

    # RLBot Compat specific fields
    _tick_skip: int
    _ball_touch_ticks: deque[bool]  # history for past _tick_skip ticks
    _prev_air_state: AirState
    _game_seconds: int

    __slots__ = tuple(__annotations__)

    exec(create_default_init(__slots__))

    @property
    def is_blue(self) -> bool:
        return self.team_num == BLUE_TEAM

    @property
    def is_orange(self) -> bool:
        return self.team_num == ORANGE_TEAM

    @property
    def is_demoed(self) -> bool:
        return self.demo_respawn_timer > 0

    @property
    def is_boosting(self) -> bool:
        return self.boost_active_time > 0

    @property
    def is_supersonic(self) -> bool:
        return self.supersonic_time > 0

    @property
    def can_flip(self) -> bool:
        return (
            not self.has_double_jumped
            and not self.has_flipped
            and self.air_time_since_jump < DOUBLEJUMP_MAX_DELAY
        )

    @property  # TODO This one isn't in rsim python yet, emulate with prop
    def is_flipping(self) -> bool:
        return self.has_flipped and self.flip_time < FLIP_TORQUE_TIME

    @is_flipping.setter
    def is_flipping(self, value: bool):
        if value:
            self.has_flipped = True
            if self.flip_time >= FLIP_TORQUE_TIME:
                self.flip_time = 0
        else:
            self.flip_time = FLIP_TORQUE_TIME

    @property
    def had_car_contact(self) -> bool:
        return self.bump_victim_id is not None

    @property
    def inverted_physics(self) -> PhysicsObject:
        if self._inverted_physics is None:
            self._inverted_physics = self.physics.inverted()
        return self._inverted_physics

    # Octane: hitbox=BoxShape(length=118.00738, width=84.19941, height=36.159073), hitbox_offset=Vector3(x=13.87566, y=0, z=20.754988)
    # Dominus: hitbox=BoxShape(length=127.92678, width=83.27995, height=31.3), hitbox_offset=Vector3(x=9, y=0, z=15.75)
    # Batmobile: hitbox=BoxShape(length=128.81978, width=84.670364, height=29.394402), hitbox_offset=Vector3(x=9.008572, y=0, z=12.0942)
    # Breakout: hitbox=BoxShape(length=131.49236, width=80.521, height=30.3), hitbox_offset=Vector3(x=12.5, y=0, z=11.75)
    # Venom: hitbox=BoxShape(length=127.01919, width=82.18787, height=34.159073), hitbox_offset=Vector3(x=13.87566, y=0, z=20.754988)
    # Merc: hitbox=BoxShape(length=120.72023, width=76.71031, height=41.659073), hitbox_offset=Vector3(x=11.37566, y=0, z=21.504988)

    @staticmethod
    def detect_hitbox(hitbox_shape: BoxShape, hitbox_offset: Vector3):
        if compare_hitbox_shape(hitbox_shape, 118.00738, 84.19941, 36.159073):
            return OCTANE
        if compare_hitbox_shape(hitbox_shape, 127.92678, 83.27995, 31.3):
            return DOMINUS
        if compare_hitbox_shape(hitbox_shape, 128.81978, 84.670364, 29.394402):
            return PLANK
        if compare_hitbox_shape(hitbox_shape, 131.49236, 80.521, 30.3):
            return BREAKOUT
        if compare_hitbox_shape(hitbox_shape, 127.01919, 82.18787, 34.159073):
            return HYBRID
        if compare_hitbox_shape(hitbox_shape, 120.72023, 76.71031, 41.659073):
            return MERC
        return OCTANE

    @staticmethod
    def create_compat_car(packet: GameTickPacket, player_index: int, tick_skip: int):
        player_info = packet.players[player_index]
        car = Car()
        car.team_num = BLUE_TEAM if player_info.team == 0 else ORANGE_TEAM
        car.hitbox_type = Car.detect_hitbox(
            player_info.hitbox, player_info.hitbox_offset
        )
        car.supersonic_time = 0
        car.boost_active_time = 0
        car.handbrake = 0
        car.has_jumped = False
        car.jump_time = 0
        car._tick_skip = tick_skip
        car._ball_touch_ticks = deque([False] * tick_skip)
        car._prev_air_state = player_info.air_state
        car._game_seconds = packet.game_info.seconds_elapsed
        car.flip_torque = np.zeros(3)
        car.physics = PhysicsObject.create_compat_physics_object()
        return car

    # Latest touch should only be passed if the player index of the touch is equal to the current player. Player indices can change in a packet
    # so this is not the responsibility of the Car.
    def update(
        self, player_info: PlayerInfo, latest_touch: Optional[Touch], ticks_elapsed: int
    ):
        # Assuming hitbox_type and team_num can't change without updating spawn id (and therefore creating new compat car)
        time_elapsed = TICK_TIME * ticks_elapsed
        self._game_seconds += time_elapsed

        for _ in range(min(self._tick_skip, ticks_elapsed)):
            self._ball_touch_ticks.popleft()
        for _ in range(min(self._tick_skip, ticks_elapsed)):
            self._ball_touch_ticks.append(False)
        if latest_touch is not None:
            ticks_since_touch = int(
                round(
                    (self._game_seconds - latest_touch.game_seconds) * TICKS_PER_SECOND
                )
            )
            if ticks_since_touch < self._tick_skip:
                self._ball_touch_ticks[-(ticks_since_touch + 1)] = True
        self.ball_touches = sum(self._ball_touch_ticks)
        self.demo_respawn_timer = (
            0
            if player_info.demolished_timeout == -1
            else player_info.demolished_timeout
        )
        if player_info.is_supersonic:
            self.supersonic_time += time_elapsed
        else:
            self.supersonic_time = 0
        self.boost_amount = player_info.boost / 100
        # Taken from rocket sim
        if self.boost_active_time > 0:
            if (
                not player_info.last_input.boost
                and self.boost_active_time >= MIN_BOOST_TIME
            ):
                self.boost_active_time = 0
            else:
                self.boost_active_time += time_elapsed
        else:
            if player_info.last_input.boost:
                self.boost_active_time = time_elapsed

        if player_info.last_input.handbrake:
            self.handbrake += POWERSLIDE_RISE_RATE * time_elapsed
        else:
            self.handbrake -= POWERSLIDE_FALL_RATE * time_elapsed
        self.handbrake = min(1, max(0, self.handbrake))

        self.is_holding_jump = player_info.last_input.jump

        match player_info.air_state:
            case AirState.OnGround:
                self.on_ground = True
                self.is_jumping = False
                self.has_jumped &= self.jump_time < MIN_JUMP_TIME + JUMP_RESET_TIME_PAD
                self.has_flipped = False
                self.has_double_jumped = False
                self.air_time_since_jump = 0
                self.flip_time = 0
            case AirState.Jumping:
                if self._prev_air_state == AirState.OnGround:
                    self.jump_time = 0
                self.jump_time += TICK_TIME * ticks_elapsed
                # After pressing jump, it usually takes 6 ticks to leave the ground
                self.on_ground = self.jump_time > 6 * TICK_TIME
                self.is_jumping = True
                self.has_jumped = True
            case AirState.InAir:
                self.on_ground = False
                self.is_jumping = False
            case AirState.Dodging:
                self.on_ground = False
                self.is_jumping = False
                if self._prev_air_state != AirState.Dodging:
                    self.flip_time = 0
                    dodge_dir = np.array(
                        [
                            -player_info.last_input.pitch,
                            player_info.last_input.yaw + player_info.last_input.roll,
                            0,
                        ]
                    )
                    if (dodge_dir < 0.1).all():
                        dodge_dir *= 0
                    else:
                        dodge_dir /= np.sqrt(
                            dodge_dir[0] * dodge_dir[0] + dodge_dir[1] * dodge_dir[1]
                        )
                    self.flip_torque = np.array([-dodge_dir[1], dodge_dir[0], 0])
                    self.has_flipped = True
            case AirState.DoubleJumping:
                self.on_ground = False
                self.is_jumping = False
                self.has_double_jumped = True

        if self.has_jumped and not self.is_jumping:
            self.air_time_since_jump += time_elapsed
        else:
            self.air_time_since_jump = 0

        if self.has_flipped:
            self.flip_time += time_elapsed

        # Cannot handle these:
        # self.bump_victim_id
        # self.is_autoflipping
        # self.autoflip_timer
        # self.autoflip_direction

        self.physics.update(player_info.physics)
        self._inverted_physics = self.physics.inverted()

        self._prev_air_state = player_info.air_state
