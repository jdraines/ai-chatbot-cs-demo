"""Model layer: data loading, embedding-based retrieval, LLM fallback."""

import importlib.resources
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore

from .providers import get_embedder, get_llm, get_similarity_threshold

_FALLBACK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a helpful customer support agent for Thoughtful AI, a company "
                "that builds AI-powered automation agents for healthcare. "
                "Answer the user's question as helpfully as you can. "
                "If the question is outside your knowledge, say so honestly."
            ),
        ),
        ("human", "{input}"),
    ]
)


def _load_qa() -> list[dict]:
    ref = importlib.resources.files("ai_chatbot_cs_demo").joinpath(
        "../../data/qa_data.json"
    )
    with ref.open() as f:
        return json.load(f)["questions"]


class AgentModel:
    def __init__(self) -> None:
        self._chain = _FALLBACK_PROMPT | get_llm()
        self._threshold = get_similarity_threshold()
        qa = _load_qa()
        self._store = InMemoryVectorStore.from_texts(
            texts=[item["question"] for item in qa],
            embedding=get_embedder(),
            metadatas=qa,
        )

    def answer(self, user_input: str) -> str:
        results = self._store.similarity_search_with_score(user_input, k=1)
        if results and results[0][1] >= self._threshold:
            return results[0][0].metadata["answer"]
        return str(self._chain.invoke({"input": user_input}).content)
