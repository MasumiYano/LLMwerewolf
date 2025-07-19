from Player import Player, GameState  # Import GameState from Player
from game_rag import GameRAG
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from typing import List, Dict, Optional, Any


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

        self.agent_executor = create_react_agent(
            self.llm, self.tools, checkpointer=self.memory
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

        system_prompt = f"""You're a werewolf discussing with your werewolf teammates who to eliminate tonight.

        Current game state:
            - Day: {game_state["day_count"]}
            - Alive players: {game_state["alive_players"]}
            - Your werewolf teammates: {werewolf_teammates}
            - Potential targets: {potential_targets}

        Previous team discussion:
            {discussion_context}

        Discuss strategically with your team. Consider:
            1. Who suspects you or your teammates the most?
            2. Who has the most influence over other players?
            3. Who might be a special role? 
            4. How will this elimination affect tomorrow's voting dynamics?

        Be collaborative and strategic. Keep response concise (2-3 sentences)
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
        teammate: List[str],
    ):
        """Speak during the structured day discussion"""
        if not teammate:
            teammate = []

        conversation_context = "\n".join([
            f"{stmt['player']}: {stmt['message']}"
            for stmt in previous_discussions[-10:]
        ])

        if game_state["day_count"] == 0:
            system_prompt = f"""You are a werewolf pretending to be a villager. It's the FIRST day (Day 0), round {round_num}.

            **CRITICAL - YOUR WEREWOLF TEAMMATES: {teammate}**
            - NEVER cast suspicion on your teammates: {teammate}
            - Protect them from accusations

            Current game state:
                - Alive players: {game_state["alive_players"]}
                - Your werewolf teammates: {teammate}
                - Last night's victim: {game_state.get("last_night_victim", "none")} (you eliminated them!)

            This is the FIRST discussion. Act like a concerned villager:
                1. React "naturally" to the night elimination (act surprised/concerned)
                2. Deflect any early suspicion from yourself or teammates
                3. Build trust by appearing helpful and analytical
                4. Make observations about other players' reactions
                5. Don't reference "yesterday" or "previous rounds" - there weren't any!

            Recent conversation this round:
                {conversation_context}

            Be concrete and engaging (2-3 sentences):
            GOOD: "I'm shocked about {game_state.get("last_night_victim", "X")}! Frank, you seem nervous - any thoughts on why they were targeted?"
            BAD: "Based on yesterday's patterns..." (there weren't any!)

            Act convincingly as a concerned villager on the FIRST day.
            """
        else:
            system_prompt = f"""You are a werewolf pretending to be a villager. It's day {game_state["day_count"]}, round {round_num}.

            **CRITICAL - YOUR WEREWOLF TEAMMATES: {teammate}**
            - NEVER cast suspicion on your teammates: {teammate}
            - Protect them from accusations
            - Work together as a team

            Current game state:
                - Alive players: {game_state["alive_players"]}
                - Your werewolf teammates: {teammate}
                - Last night's victim: {game_state.get("last_night_victim", "none")}
                - Last eliminated by vote: {game_state.get("last_eliminated", "none")}

            Recent conversation: 
                {conversation_context}

            As a werewolf disguised as a villager:
                1. Act like you're hunting werewolves (but NEVER your teammates)
                2. Deflect suspicion from yourself and teammates: {teammate}
                3. Cast suspicion on VILLAGERS only
                4. Reference previous voting patterns to seem analytical
               5. If a teammate is accused, defend them subtly
                6. Make SPECIFIC observations about individual players

            Be concrete and engaging (3-4 sentences):
            EXAMPLE: "Alice, I noticed you hesitated before voting yesterday - why?"
            NOT: "We should all stay vigilant"

            Act convincingly as a villager but PROTECT YOUR TEAM: {teammate}
            """

        config = {
            "configurable": {
                "thread_id": f"werewolf_{self.user_id}_day_discussion_{game_state['day_count']}_round_{round_num}"  # Fixed variable name
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
        system_prompt = f"""You are a werewolf who must vote like a villager would.

        Current game state: 
            - Day: {game_state["day_count"]}
            - Alive players: {game_state["alive_players"]}
        
        Vote strategically as a fake villager:
            1. Don't vote for other werewolves
            2. Vote for villagers, especially ones who suspect you
            3. Make your vote seem logical from a villager perspective

        Respond with the player name you want to vote for, or 'none' to abstain.
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
            p for p in game_state["alive_players"] if p != self.user_id
        ]

        system_prompt = f"""You are the last werewolf and must decide who to eliminate tonight.

        **CRITICAL**:
            - Your werewolf teammates: {teammates}
            - NEVER vote for your teammates
            - This is the most important rule - protect your team!

        Potential targets: {potential_targets}

        Vote strategically as a fake villager:
            1. Make your vote seem logical from a villagers perspective
            2. Especially target villagers who suspect you or your teammate

        Respond with just the player name you want to eliminate.
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
            if player.lower() in response_lower:
                return player

        return None

    def take_turn(self, game_state: GameState):  # Added return type
        return self.speak_in_discussion(game_state, 1, [], [])

    def get_description(self):
        return (
            f"Werewolf player {self.user_id} - secretly working to eliminate villagers"
        )

    def get_user_id(self):
        return self.user_id

    def get_role(self):
        return "werewolf"
