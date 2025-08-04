from typing import Optional, Dict, List
from Player import Player, PlayerStatus, GameState  # Import GameState from Player
from game_rag import GameRAG
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from utils import load_prompts


class Villager(Player):
    def __init__(self, user_id: str, rag: GameRAG, model_name="gpt-4o-mini"):
        self.user_id = user_id
        self.rag = rag
        self.role_name = "villager"
        self.side = "villagers"
        self.llm = init_chat_model(model_name, model_provider="openai")
        self.memory = MemorySaver()

        self.tools = [
            self._create_rule_search_tool(),
            self._create_villager_strategy_tool(),
            self._create_conversation_search_tool(),
        ]

        system_prompt = load_prompts("dumb_villager.txt", user_id=self.user_id)

        self.agent_executor = create_react_agent(
            self.llm,
            self.tools,
            checkpointer=self.memory,
            prompt=system_prompt,
        )

    def _create_rule_search_tool(self):
        @tool
        def search_rules(query: str):
            """Search for game rules and mechanics"""
            docs = self.rag.rule_vector_store.similarity_search(
                query, k=2
            )  # Fixed vector store
            return "\n\n".join([doc.page_content for doc in docs])

        return search_rules

    def _create_villager_strategy_tool(self):
        """Search for villager strategies and tactics"""

        @tool
        def search_villager_strategies(query: str):
            """Search for villager strategies and tactics"""
            docs = self.rag.villager_vector_store.similarity_search(query, k=2)
            return "\n\n".join([doc.page_content for doc in docs])

        return search_villager_strategies

    def _create_conversation_search_tool(self):
        @tool
        def search_conversations(query: str):
            """Search recent game conversations for relevant information"""
            docs = self.rag.conversation_vector_store.similarity_search(query, k=3)
            return "\n\n".join([doc.page_content for doc in docs])

        return search_conversations

    def speak_in_discussion(
        self,
        game_state: GameState,
        round_num: int,
        previous_statements: List[Dict[str, str]],
    ):
        conversation_context = "\n".join([
            f"{stmt['player']}: {stmt['message']}" for stmt in previous_statements[-10:]
        ])

        system_prompt = f"""DISCUSSION PHASE - Day {game_state["day_count"]}, Round {round_num}

        Game State:
            - Alive players: {game_state["alive_players"]}
            - Last night's victim: {game_state["last_night_victim"]}
            - Last eliminated by vote: {game_state["last_eliminated"]}

        Recent conversation:
            {conversation_context}

        Task: Make this discussion engaging.
        """

        config = {  # Fixed config structure
            "configurable": {
                "thread_id": f"villager_{self.user_id}_day_{game_state['day_count']}_round_{round_num}"
            }
        }

        messages = [  # Fixed variable name
            SystemMessage(content=system_prompt),
            HumanMessage(content="It's your turn to speak. What do you want to say?"),
        ]

        for event in self.agent_executor.stream(
            {"messages": messages}, config=config, stream_mode="values"
        ):
            response = event["messages"][-1].content

        return response

    def get_vote(self, game_state: GameState, discussion_history: List[Dict[str, str]]):
        """Vote after hearing all discussion"""
        conversation_context = "\n".join([
            f"{stmt['player']}: {stmt['message']}" for stmt in discussion_history
        ])

        system_prompt = f"""VOTING PHASE - Day {game_state["day_count"]}

        Full discussion today:
            {conversation_context}


        Available players to vote for: {[p for p in game_state["alive_players"] if p != self.user_id]}

        Task: Choose one player to vote for elimination, or response with 'none' or 'abstain'.
        """

        config = {
            "configurable": {
                "thread_id": f"villager_{self.user_id}_vote_{game_state['day_count']}"
            }
        }

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Who do you vote to eliminate?"),
        ]

        for event in self.agent_executor.stream(
            {"messages": messages}, config=config, stream_mode="values"
        ):
            response = event["messages"][-1].content

        return self._extract_target(response, game_state["alive_players"])

    def _extract_target(self, response: str, alive_players: List[str]):
        response_lower = response.lower()  # Fixed variable name

        if any(word in response_lower for word in ["none", "abstain", "no one"]):
            return None

        for player in alive_players:
            if player.lower() in response_lower:
                return player

        return None

    def get_night_action(self, game_state: GameState):
        return None

    def take_turn(self, game_state: GameState):
        return self.speak_in_discussion(game_state, 1, [])

    def get_description(self):
        return f"Villager {self.user_id} - trying to identify werewolves"

    def get_user_id(self):
        return self.user_id

    def get_role(self):
        return "villager"
