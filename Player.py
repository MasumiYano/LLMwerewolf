from abc import ABC, abstractmethod
from enum import Enum
from typing import TypedDict, List, Dict


class PlayerStatus(Enum):
    ALIVE = "alive"
    DEAD = "dead"


class GameState(TypedDict):
    phase: str  # "night", "day", "voting"
    day_count: int
    players: Dict[str, PlayerStatus]  # Mapping ids to status
    alive_players: List[str]
    last_eliminated: str
    last_night_victim: str


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
    def get_user_id(self):
        pass
