from rlbot.flat import PlayerInfo

from ..car import Car
from ..common_values import DOUBLEJUMP_MAX_DELAY
from .physics_object import PhysicsObject


class PlayerData(object):
    def __init__(self):
        self.car_id: int = -1
        self.team_num: int = -1
        self.match_goals: int = -1
        self.match_saves: int = -1
        self.match_shots: int = -1
        self.match_demolishes: int = -1
        self.boost_pickups: int = -1
        self.is_demoed: bool = False
        self.on_ground: bool = False
        self.ball_touched: bool = False
        self.has_jump: bool = False
        self.has_flip: bool = False
        self.boost_amount: float = -1
        self.car_data: PhysicsObject = PhysicsObject()
        self.inverted_car_data: PhysicsObject = PhysicsObject()

    @staticmethod
    def create_from_v2(
        car: Car, player_info: PlayerInfo, car_id: int, boost_pickups: int
    ):
        player = PlayerData()
        player.car_id = car_id
        player.team_num = car.team_num
        player.match_goals = player_info.score_info.goals
        player.match_saves = player_info.score_info.saves
        player.match_shots = player_info.score_info.shots
        player.match_demolishes = player_info.score_info.demolitions
        player.boost_pickups = boost_pickups
        player.is_demoed = car.is_demoed
        player.on_ground = car.on_ground
        player.ball_touched = car.ball_touches > 0
        player.has_jump = not car.has_jumped
        player.has_flip = (
            not car.has_flipped
            and not car.has_double_jumped
            and car.air_time_since_jump < DOUBLEJUMP_MAX_DELAY
        )
        player.boost_amount = car.boost_amount
        player.car_data = PhysicsObject.create_from_v2(car.physics)
        player.inverted_car_data = PhysicsObject.create_from_v2(car.inverted_physics)
        return player
