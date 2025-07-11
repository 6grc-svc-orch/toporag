import json
from uuid import uuid4
from typing import Dict, List
from utils import read_json

from llama_index.core import Document

from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.core.schema import NodeRelationship
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.embeddings.openai import OpenAIEmbedding

from dotenv import load_dotenv


# import any embedding model on HF hub (https://huggingface.co/spaces/mteb/leaderboard)
def initialize_rag(model_name_or_path: str):
    if model_name_or_path in [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ]:
        load_dotenv()
        Settings.embed_model = OpenAIEmbedding(model=model_name_or_path)
    else:
        Settings.embed_model = HuggingFaceEmbedding(model_name=model_name_or_path)
        Settings.llm = None


def query_engine_fn(
    documents: List[Document],
    top_k: int = None,
):
    # store docs into vector DB
    index = VectorStoreIndex.from_documents(documents)

    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )

    # assemble query engine
    query_engine: RetrieverQueryEngine = RetrieverQueryEngine(retriever=retriever)
    return query_engine


def rag_fn(data: List[Dict] = None, q_len: int = 1, top_k: int = 3):
    if not data:
        CONTEXT_LOG_PATH = "logs/a.json"
        data = read_json(CONTEXT_LOG_PATH)
    # TODO: use d["timestamp"] for doc_id
    documents = [Document(text=json.dumps(d), doc_id=str(uuid4())) for d in data]

    query_engine = query_engine_fn(documents, top_k=top_k//q_len)

    # Query
    query_ = []
    for i in range(q_len):
        query = data[-i-1]  # Last recent record
        query_.append(query)
        response = query_engine.query(json.dumps(query))
        responses = [
            {
                "key": node.node.relationships.get(NodeRelationship.SOURCE).node_id,
                "score": node.score,
                "text": node.text,
            }
            for node in response.source_nodes
        ]
        query_.extend([json.loads(r["text"]) for r in responses])
    # print("query_", [q["timestamp"] for q in query_])
    return query_


if __name__ == "__main__":
    initialize_rag("BAAI/bge-small-en-v1.5")
    q_ = rag_fn(q_len=2)
    from pprint import pprint
