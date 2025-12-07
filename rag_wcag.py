import requests
from typing import List
import os

# ---------- LlamaIndex imports ----------
from llama_index.core.llms import (
    CustomLLM,
    LLMMetadata,
    CompletionResponse,
    CompletionResponseGen,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import load_index_from_storage


# ================================================================
#  LM STUDIO LLM (chat completions backend)
# ================================================================
class LMStudioLLM(CustomLLM):
    model: str
    api_base: str

    def __init__(self, model: str, api_base: str, **kwargs):
        super().__init__(model=model, api_base=api_base.rstrip("/"), **kwargs)

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        resp = requests.post(
            f"{self.api_base}/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.2),
            },
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return CompletionResponse(text=content)

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        yield self.complete(prompt, **kwargs)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model)


# ================================================================
#  LM STUDIO EMBEDDINGS BACKEND
# ================================================================
class LMStudioEmbedding(BaseEmbedding):
    model: str
    api_base: str

    def __init__(self, model: str, api_base: str, **kwargs):
        super().__init__(model=model, api_base=api_base.rstrip("/"), **kwargs)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = requests.post(
            f"{self.api_base}/embeddings",
            json={"model": self.model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in data]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_batch([query])[0]

    async def _aget_query_embedding(self, query: str):
        return self._embed_batch([query])[0]


# ================================================================
#  CONFIGURE LM STUDIO (LLM + Embeddings)
# ================================================================
API_BASE = "http://localhost:1234/v1"

llm = LMStudioLLM(
    model="meta-llama-3.1-8b-instruct",
    api_base=API_BASE,
)

embed_model = LMStudioEmbedding(
    model="text-embedding-nomic-embed-text-v1.5",
    api_base=API_BASE,
)

Settings.llm = llm
Settings.embed_model = embed_model


# ================================================================
#  PERSISTENCE CONFIGURATION
# ================================================================
PERSIST_DIR = "./wcag_index"

if os.path.exists(PERSIST_DIR):
    print("Loading existing vector index from disk...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    print("Index not found. Building and persisting...")
    documents = SimpleDirectoryReader(
        input_files=["wcag-2.2.pdf"]
    ).load_data()
    print(f"Loaded {len(documents)} document(s).")

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("Index persisted to disk.")


query_engine = index.as_query_engine(similarity_top_k=5)


# ================================================================
#  RUN RAG QUERY
# ================================================================
question = "What does WCAG 2.2 require when a user interface relies on dragging movements?"

print("\nQUESTION:")
print(question)

response = query_engine.query(question)

print("\nANSWER:")
print(response)

print("\n--- Retrieved source snippets ---")
for i, node_with_score in enumerate(response.source_nodes, start=1):
    print(f"\n[{i}] score={node_with_score.score:.3f}")
    print(node_with_score.node.get_content()[:500])
