from abc import ABC, abstractmethod
from enum import Enum
from main import GameState


class PlayerStatus(Enum):
    ALIVE = "alive"
    DEAD = "dead"


class Player(ABC):
    @abstractmethod
    def get_night_action(self, game_state: GameState):
        pass

    @abstractmethod
    def get_vote(self, game_state: GameState):
        pass

    @abstractmethod
    def take_turn(self, game_state: GameState):
        pass

    @abstractmethod
    def get_description(self):
        pass

    @abstractmethod
    def get_user_id(self) -> str:
        pass
