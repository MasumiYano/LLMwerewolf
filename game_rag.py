import os
import bs4
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from Player import GameState
from typing import Optional, Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from config import VILLAGER_NUM, WEREWOLF_NUM, PLAYER_NUM


class GameRAG:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-3-large",
    ):
        if not os.environ.get("OPENAI_API_KEY"):
            print("OpenAI API Key not found")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.rule_vector_store = None
        self.werewolf_vector_store = None
        self.villager_vector_store = None
        self.conversation_vector_store = None
        self.initialize_all_vectors()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        self.rule = self.load_rules()
        self.rule_splits = self.text_splitter.split_documents(documents=self.rule)
        _ = self.rule_vector_store.add_documents(documents=self.rule_splits)

    def initialize_all_vectors(self):
        os.makedirs("./chroma_db/rules", exist_ok=True)
        os.makedirs("./chroma_db/werewolf", exist_ok=True)
        os.makedirs("./chroma_db/villager", exist_ok=True)

        self.rule_vector_store = Chroma(
            collection_name="shared_rules",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db/rules",
        )

        self.werewolf_vector_store = Chroma(
            collection_name="werewolf_strategies",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db/werewolf",
        )

        self.villager_vector_store = Chroma(
            collection_name="villager_strategies",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db/villager",
        )

        self.conversation_vector_store = Chroma(
            collection_name="current_conversation",
            embedding_function=self.embeddings,
        )

    def load_rules(self):
        rules_text = f"""
        WEREWOLF GAME RULES

        SETUP:
        - {PLAYER_NUM} players total: {VILLAGER_NUM} villagers and {WEREWOLF_NUM} werewolves
        - Roles are assigned secretly at the start of the game
        - Players sit in a circle for discussion phases

        GAME PHASES:

        NIGHT PHASE:
        - All players close their eyes (go to sleep)
        - Werewolves wake up and silently discuss who to eliminate
        - Werewolves choose one villager to eliminate
        - Werewolves go back to sleep
        - The eliminated player is removed from the game

        DAY PHASE:
        - All players wake up
        - The night's victim is announced and removed
        - Players discuss in rounds to identify werewolves
        - Each player gets 2 turns to speak (2 rounds of discussion)
        - Players share suspicions, ask questions, and defend themselves
        - After discussion, players vote to eliminate someone
        - The player with the most votes is eliminated and their role revealed

        WIN CONDITIONS:
        - VILLAGERS WIN: All werewolves are eliminated
        - WEREWOLVES WIN: Werewolves equal or outnumber villagers

        PLAYER ROLES:

        VILLAGERS:
        - Goal: Identify and eliminate all werewolves
        - Can only vote during day phase
        - Must use logic and observation to find werewolves
        - Should pay attention to suspicious behavior and voting patterns

        WEREWOLVES:
        - Goal: Eliminate villagers until they equal/outnumber them
        - Can eliminate one villager each night
        - Must pretend to be villagers during day phase
        - Should deflect suspicion and blend in with villagers

        STRATEGY TIPS:
        - Watch for inconsistencies in statements
        - Track voting patterns across multiple days
        - Pay attention to who benefits from eliminations
        - Notice who deflects suspicion or changes topics
        - Form alliances carefully
        - Use deductive reasoning based on game flow
        """

        return [Document(page_content=rules_text, metadata={"source": "game_rules"})]

    def add_conversations(self, conversation: Dict[str, str], game_state: GameState):
        """Add conversation and game state to data"""
        flatten_game_state = self._flatten_metadata(game_state)
        conversation_docs = self.text_splitter.create_documents(
            texts=[str(conversation)], metadatas=[flatten_game_state]
        )

        self.conversation_vector_store.add_documents(conversation_docs)

    def add_werewolf_knowledge(self, knowledge: str, game_state: GameState):
        """Add werewolf knowledge to werewolf vector"""
        flatten_game_state = self._flatten_metadata(game_state)
        knowledge_docs = self.text_splitter.create_documents(
            texts=[knowledge], metadatas=[flatten_game_state]
        )

        self.werewolf_vector_store.add_documents(knowledge_docs)

    def add_villager_knowledge(self, knowledge: str, game_state: GameState):
        """Add villager knowledge to villager vector"""
        flatten_game_state = self._flatten_metadata(game_state)
        knowledge_docs = self.text_splitter.create_documents(
            texts=[knowledge], metadatas=[flatten_game_state]
        )

        self.villager_vector_store.add_documents(knowledge_docs)

    def clear_conversation_history(self):
        """Clears the conversation vector store after each game"""
        self.conversation_vector_store = Chroma(
            collection_name="current_conversation", embedding_function=self.embeddings
        )

    def _flatten_metadata(self, game_state: GameState):
        return {
            "phase": game_state["phase"],
            "day_count": game_state["day_count"],
            "alive_players_count": len(game_state["alive_players"]),
            "alive_players": ",".join(game_state["alive_players"]),
            "last_eliminated": game_state["last_eliminated"],
            "last_night_victim": game_state["last_night_victim"],
            "total_players": len(game_state["players"]),
        }
