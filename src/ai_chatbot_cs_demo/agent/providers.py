"""Provider selection: real OpenAI clients or naive stubs when no API key is set."""

import os
from typing import Iterator

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer

_STUB_MSG = (
    "I'm sorry, I wasn't able to find an answer to your question. To enable a live "
    "response using OpenAI embeddings and text generation, set the OPENAI_API_KEY "
    "environment variable."
)


class TfidfEmbeddings(Embeddings):
    """TF-IDF BoW embedder. Vocabulary is fitted on the first call to embed_documents."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer()
        self._fitted = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        matrix = self._vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix.toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        if not self._fitted:
            raise RuntimeError("embed_documents must be called before embed_query")
        return self._vectorizer.transform([text]).toarray()[0].tolist()


def _stub_messages() -> Iterator[AIMessage]:
    while True:
        yield AIMessage(content=_STUB_MSG)


def get_similarity_threshold() -> float:

    if os.getenv("OPEN_API_KEY"):
        if os.getenv("OPEN_API_KEY") != "your-api-key-here":
            return 0.85
    return 0.60


def get_embedder() -> Embeddings:
    if os.getenv("OPENAI_API_KEY"):
        if os.getenv("OPEN_API_KEY") != "your-api-key-here":
            return OpenAIEmbeddings(model="text-embedding-3-small")
    return TfidfEmbeddings()


def get_llm() -> BaseChatModel:
    if os.getenv("OPENAI_API_KEY"):
        if os.getenv("OPEN_API_KEY") != "your-api-key-here":
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    return GenericFakeChatModel(messages=_stub_messages())
