from typing import List, Optional
from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata

class ToolBuilder:
    """
    utility class for creating document-specific query tools.
    """
    @staticmethod
    def create_document_tools(
        doc_name: str,
        vector_index: VectorStoreIndex,
        summary_index: SummaryIndex,
        index_builder: Optional[any] = None  
    ) -> List[QueryEngineTool]:
        """
        create vector and summary tools for a specific document.
        
        args:
            doc_name: name of the document
            vector_index: vector store index for specific queries
            summary_index: summary index for holistic document overview
            index_builder: optional builder with custom query engine settings
            
        returns:
            list of query engine tools for the document
        """
        # create query engines based on whether a custom index builder is provided
        if index_builder:
            vector_query_engine = index_builder.create_query_engine(vector_index, reranking=True)
            summary_query_engine = summary_index.as_query_engine()
        else:
            vector_query_engine = vector_index.as_query_engine()
            summary_query_engine = summary_index.as_query_engine()

        # create and return the tools list
        return [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name="vector_tool",
                    description=(
                        f"useful for questions related to specific aspects of {doc_name}."
                    ),
                ),
            ),
            QueryEngineTool(
                query_engine=summary_query_engine,
                metadata=ToolMetadata(
                    name="summary_tool",
                    description=(
                        f"useful for any requests that require a holistic summary "
                        f"of everything about {doc_name}. for questions about "
                        "more specific sections, please use the vector_tool."
                    ),
                ),
            ),
        ]