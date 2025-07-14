import os
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing import List, TypedDict, Annotated
import bs4
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

if not os.environ.get("OPENAI_API"):
    print("OpenAI API not found")

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
)

# Load and chunk content from the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)


template = """
Use the following pieces of context to answer the questions at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
When giving them an answer, always make sure to use starwars reference, and keep the answer as concise as possible.
Always say "May the force be with you" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

# Or, set the custom promptcustom_rag_prompt = PromptTemplate.from_template(template)


class Search(BaseModel):
    """Search query."""

    query: str = Field(..., description="Search query to run")


# Define state for application
class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str


def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    analysis_prompt = f"""
    Given the user's question "{state["question"]}"
    Generate an optimized search query that would help find the most relevant information
    to answer this question. The search query should be concise and focused on the key concepts.
    """
    query_result = structured_llm.invoke(analysis_prompt)
    return {"query": query_result}


def retrieve(state: State):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(query.query, k=4)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = custom_rag_prompt.invoke({
        "question": state["question"],
        "context": docs_content,
    })
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

response = graph.invoke({"question": "What does she say about AI?"})
print("Original Question: ", response["question"])
print("Analyzed Query: ", response["query"].query)
print("Answer: ", response["answer"])
