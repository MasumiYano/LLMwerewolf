from main import GameState
from Player import Player
from game_rag import GameRAG
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


class Werewolf(Player):
    def __init__(self, rag: GameRAG, role_name="werewolf", side="werewolves"):
        self.rag = rag
        self.role_name = role_name
        self.side = side
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")

    def get_night_action(self, game_state: GameState):
