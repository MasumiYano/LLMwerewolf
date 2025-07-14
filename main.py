import os
from game_rag import GameRAG
from langchain.chat_models import init_chat_model
from Player import PlayerStatus
from typing import TypedDict, List, Dict
# from werewolf import Werewolf


class GameState(TypedDict):
    phase: str  # "night", "day", "voting"
    day_count: int
    players: Dict[str, PlayerStatus]  # Mapping ids to status
    alive_players: List[str]
    last_eliminated: str
    last_night_victim: str


def main():
    werewolf_1 = Werewolf(
        "Alice", init_chat_model("gpt-4o-mini", model_provider="openai")
    )
