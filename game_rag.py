import os
import bs4
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from .Player import GameState
from typing import Optional, Any, Dict
from langchain_openai import ChatOpenAI


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
        self.rule_vector_store = Chroma(
            collection_name="shared_rules",
            embedding_function=self.embeddings,
        )

        self.werewolf_vector_store = Chroma(
            collection_name="werewolf_strategies",
            embedding_function=self.embeddings,
        )

        self.villager_vector_store = Chroma(
            collection_name="villager_strategies",
            embedding_function=self.embeddings,
        )

        self.conversation_vector_store = Chroma(
            collection_name="current_conversation",
            embedding_function=self.embeddings,
        )

    def load_rules(
        self, url: str = "https://teambuilding.com/blog/werewolf-game-rules"
    ):
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )

        return loader.load()

    def add_conversations(self, conversation: Dict[str, str], game_state: GameState):
        """Add conversation and game state to data"""
        conversation_docs = self.text_splitter.create_documents(
            texts=[str(conversation)], metadatas=[dict(game_state)]
        )

        self.conversation_vector_store.add_documents(conversation_docs)

    def add_werewolf_knowledge(self, knowledge: str, game_state: GameState):
        """Add werewolf knowledge to werewolf vector"""
        knowledge_docs = self.text_splitter.create_documents(
            texts=[knowledge], metadatas=[dict(game_state)]
        )

        self.werewolf_vector_store.add_documents(knowledge_docs)

    def add_villager_knowledge(self, knowledge: str, game_state: GameState):
        """Add villager knowledge to villager vector"""
        knowledge_docs = self.text_splitter.create_documents(
            texts=[knowledge], metadatas=[dict(game_state)]
        )

        self.villager_vector_store.add_documents(knowledge_docs)

    def clear_conversation_history(self):
        """Clears the conversation vector store after each game"""
        self.conversation_vector_store = Chroma(
            collection_name="current_conversation", embedding_function=self.embeddings
        )
