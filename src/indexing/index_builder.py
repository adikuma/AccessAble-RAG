from typing import Dict, List, Tuple
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.schema import Document
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from pathlib import Path
import logging


class IndexBuilder:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir

        self.reranker = LLMRerank(
            choice_batch_size=5,
            top_n=2,
        )
        logging.info("initialized llm-based reranker for improved retrieval precision")

    def build_indexes(
        self, doc_name: str, nodes: List[Document]
    ) -> Tuple[VectorStoreIndex, SummaryIndex]:
        persist_dir = Path(self.base_dir) / doc_name

        if not persist_dir.exists():
            logging.info(
                f"index for '{doc_name}' not found. building and persisting new index..."
            )
            vector_index = VectorStoreIndex(nodes)
            vector_index.storage_context.persist(persist_dir=str(persist_dir))
            logging.info(
                f"created and persisted new vector index for '{doc_name}' at {persist_dir}"
            )
        else:
            logging.info(
                f"loading existing vector index for '{doc_name}' from {persist_dir}..."
            )
            vector_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(persist_dir))
            )
            logging.info(f"successfully loaded existing vector index for '{doc_name}'")

        summary_index = SummaryIndex(nodes)
        return vector_index, summary_index

    def create_query_engine(self, index, reranking=True):
        if reranking:
            retriever = index.as_retriever(similarity_top_k=5)

            return RetrieverQueryEngine.from_args(
                retriever=retriever, node_postprocessors=[self.reranker]
            )
        else:
            return index.as_query_engine()
