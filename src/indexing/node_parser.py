from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.schema import Document
from typing import List, Dict
from src.config.settings import Settings
from dotenv import load_dotenv
import os
import logging

load_dotenv()


class NodeParser:
    def __init__(
        self, buffer_size=1, chunk_size=512, chunk_overlap=50, percentile_threshold=95
    ):
        self.parser = SemanticSplitterNodeParser(
            api_key=os.getenv("OPENAI_API_KEY"),
            buffer_size=buffer_size,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            breakpoint_percentile_threshold=percentile_threshold,
            embed_model=Settings.embed_model,
        )
        logging.info("semantic splitter initialized")

    def parse_documents(
        self, documents: Dict[str, List[Document]]
    ) -> Dict[str, List[Document]]:
        parsed_nodes = {}
        for doc_name, docs in documents.items():
            parsed_nodes[doc_name] = self.parser.get_nodes_from_documents(docs)
        return parsed_nodes
