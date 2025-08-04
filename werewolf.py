from Player import Player, GameState
from game_rag import GameRAG
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from typing import List, Dict, Optional, Any
from utils import load_prompts


class Werewolf(Player):
    def __init__(
        self, user_id: str, rag: GameRAG, role_name="werewolf", side="werewolves"
    ):
        self.user_id = user_id
        self.rag = rag
        self.role_name = role_name
        self.side = side
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.memory = MemorySaver()

        self.tools = [
            self._create_rule_search_tool(),
            self._create_werewolf_strategy_tool(),
            self._create_conversation_search_tool(),
        ]

        system_prompt = load_prompts(
            "dumb_werewolf.txt",
            user_id=self.user_id,
            teammates="[teammates will be specified in each action]",
        )

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
            docs = self.rag.rule_vector_store.similarity_search(query, k=2)
            return "\n\n".join([doc.page_content for doc in docs])

        return search_rules

    def _create_werewolf_strategy_tool(self):
        @tool
        def search_werewolf_strategies(query: str):
            """Search werewolf strategies"""
            docs = self.rag.werewolf_vector_store.similarity_search(query, k=2)
            return "\n\n".join([doc.page_content for doc in docs])

        return search_werewolf_strategies

    def _create_conversation_search_tool(self):
        @tool
        def search_conversations(query: str):
            """Search recent game conversations"""
            docs = self.rag.conversation_vector_store.similarity_search(query, k=3)
            return "\n\n".join([doc.page_content for doc in docs])

        return search_conversations

    def discuss_night_target(
        self,
        game_state: GameState,
        werewolf_teammates: List[str],
        previous_discussion: List[Dict[str, str]],
    ):
        """Talk to werewolf team to decide who to eliminate"""
        discussion_context = "\n".join([
            f"{stmt['player']}: {stmt['message']}" for stmt in previous_discussion
        ])

        potential_targets = [
            p
            for p in game_state["alive_players"]
            if p not in werewolf_teammates and p != self.user_id
        ]

        # Simplified prompt - just game state and task
        system_prompt = f"""NIGHT DISCUSSION PHASE - Day {game_state["day_count"]}

        Game State:
            - Alive players: {game_state["alive_players"]}
            - Your werewolf teammates: {werewolf_teammates}
            - Potential targets: {potential_targets}

        Previous team discussion:
            {discussion_context}

        Task: Discuss strategically with your werewolf team about who to eliminate tonight (2-3 sentences).
        """

        config = {
            "configurable": {
                "thread_id": f"werewolf_{self.user_id}_team_discussion_night_{game_state['day_count']}"
            }
        }

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="What are your thoughts on who to eliminate tonight?"),
        ]

        for event in self.agent_executor.stream(
            {"messages": messages}, config=config, stream_mode="values"
        ):
            response = event["messages"][-1].content

        return response

    def speak_in_discussion(
        self,
        game_state: GameState,
        round_num: int,
        previous_discussions: List[Dict[str, str]],
        teammates: List[str],
    ):
        """Speak during the structured day discussion"""
        if not teammates:
            teammates = []

        conversation_context = "\n".join([
            f"{stmt['player']}: {stmt['message']}"
            for stmt in previous_discussions[-10:]
        ])

        # Simplified prompt - just game state and task
        system_prompt = f"""DISCUSSION PHASE - Day {game_state["day_count"]}, Round {round_num}

        Game State:
            - Alive players: {game_state["alive_players"]}
            - Your werewolf teammates: {teammates}
            - Last night's victim: {game_state.get("last_night_victim", "none")}
            - Last eliminated by vote: {game_state.get("last_eliminated", "none")}

        Recent conversation:
            {conversation_context}

        Task: Make a discussion statement while pretending to be a villager (2-3 sentences).
        Remember: NEVER cast suspicion on your teammates: {teammates}
        """

        config = {
            "configurable": {
                "thread_id": f"werewolf_{self.user_id}_day_discussion_{game_state['day_count']}_round_{round_num}"
            }
        }

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="It's your turn to speak. What do you want to say?"),
        ]

        for event in self.agent_executor.stream(
            {"messages": messages}, config=config, stream_mode="values"
        ):
            response = event["messages"][-1].content

        return response

    def get_vote(self, game_state: GameState, teammates: List[str]):
        """Voting during day phase"""
        # Simplified prompt - just game state and task
        system_prompt = f"""VOTING PHASE - Day {game_state["day_count"]}

        Game State:
            - Alive players: {game_state["alive_players"]}
            - Your werewolf teammates: {teammates}

        Available players to vote for: {[p for p in game_state["alive_players"] if p != self.user_id and p not in teammates]}

        Task: Choose one villager to vote for elimination, or respond with 'none' to abstain.
        Remember: NEVER vote for your werewolf teammates: {teammates}
        """

        config = {
            "configurable": {
                "thread_id": f"werewolf_{self.user_id}_vote_{game_state['day_count']}"
            }
        }

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Who do you vote to eliminate?"),
        ]

        for event in self.agent_executor.stream(
            {"messages": messages}, config=config, stream_mode="values"
        ):
            final_response = event["messages"][-1].content

        return self._extract_target(final_response, game_state["alive_players"])

    def get_night_action(self, game_state: GameState, teammates: List[str] = []):
        """Make a final decision on who to eliminate (used when only one werewolf left)"""
        potential_targets = [
            p
            for p in game_state["alive_players"]
            if p != self.user_id and p not in teammates
        ]

        # Simplified prompt - just game state and task
        system_prompt = f"""NIGHT ACTION PHASE - Day {game_state["day_count"]}

        Game State:
            - Alive players: {game_state["alive_players"]}
            - Your werewolf teammates: {teammates}
            - Potential targets: {potential_targets}

        Task: Choose one villager to eliminate tonight.
        Remember: NEVER target your werewolf teammates: {teammates}
        """

        config = {
            "configurable": {
                "thread_id": f"werewolf_{self.user_id}_solo_night_{game_state['day_count']}"
            }
        }

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Who do you want to eliminate tonight?"),
        ]

        for event in self.agent_executor.stream(
            {"messages": messages}, config=config, stream_mode="values"
        ):
            final_response = event["messages"][-1].content

        return self._extract_target(final_response, game_state["alive_players"])

    def _extract_target(self, response: str, alive_players: list):
        """Extract target player from LLM response"""
        response_lower = response.lower()

        if any(word in response_lower for word in ["none", "abstain", "no one"]):
            return None

        for player in alive_players:
            if player.lower() in response_lower and player != self.user_id:
                return player

        return None

    def take_turn(self, game_state: GameState):
        return self.speak_in_discussion(game_state, 1, [], [])

    def get_description(self):
        return (
            f"Werewolf player {self.user_id} - secretly working to eliminate villagers"
        )

    def get_user_id(self):
        return self.user_id

    def get_role(self):
        return "werewolf"
