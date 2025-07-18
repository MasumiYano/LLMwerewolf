from typing import Optional, Dict, List
from Player import Player, PlayerStatus, GameState  # Import GameState from Player
from game_rag import GameRAG
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver


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

        self.agent_executor = create_react_agent(
            self.llm, self.tools, checkpointer=self.memory
        )

    def _create_rule_search_tool(self):  # Removed extra parameter
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
        """Speak during the day phase"""
        conversation_context = "\n".join([
            f"{stmt['player']}: {stmt['message']}" for stmt in previous_statements[-10:]
        ])

        system_prompt = f"""You are a villager in a Werewolf game. It's day {game_state["day_count"]}, round {round_num}.

        Current game state:
            - Alive players: {game_state["alive_players"]}
            - Last night's victim: {game_state.get("last_night_victim", "none")}
            - Last eliminated: {game_state.get("last_eliminated", "none")}

        Recent conversation:
            {conversation_context}

        As a villager, your goals:
            1. Make SPECIFIC observations about individual players (not generic advice)
            2. Ask DIRECT questions to other players about their behavior
            3. Reference CONCRETE moments from pervious rounds 
            4. Make accusations or defed yourself with evidence
            5. Propose specific theories about who might be werewolves and WHY

        Be concrete and ENGAGING
        GOOD: "XX, I noticed you voted for YY yesterday, but stayed quiet about it, why?"
        BAD: "We shuold all shre observations and stay vigilant! (Too generic and not engaging)"

        Keep your response concise but meaningful (2-3 sentences).
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

        system_prompt = f"""You are a villager voting to eliminate someone you believe is a werewolf. 

        Full discussion today: 
            {conversation_context}

        Based on everything you have heard, who seems most likely to be a werewolf?
        Consider:
            1. Suspicious behavior or contradictions
            2. Who tried to deflect suspicion
            3. Voting patterns from previous days

        Respond with just the player name you want to vote for, or 'none' to abstain.
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
