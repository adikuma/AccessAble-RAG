from typing import List, Dict, Optional, Any
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI

# note: ramp parameters model is now defined in models.py
class AgentFactory:
    """
    factory class for creating specialized agents for compliance-related tasks.
    """
    @staticmethod
    def create_document_agent(
        doc_name: str, tools: List[QueryEngineTool]
    ) -> OpenAIAgent:
        """
        create an agent specializing in a single document.
        """
        return OpenAIAgent.from_tools(
            tools,
            llm=Settings.llm,
            verbose=True,
            system_prompt=f"""
            you are a compliance specialist for {doc_name}. your task is to:
            1. find exact references in the document
            2. quote relevant clauses/specifications
            3. return raw text from documents without interpretation
            """,
        )

    @staticmethod
    def create_top_agent(
        tool_retriever: Any,
        system_prompt: Optional[str] = None,
    ) -> OpenAIAgent:
        """
        create a top-level agent that orchestrates information retrieval.
        """
        system_prompt = """
        you are a knowledgeable agent. your task is to:
        1. find relevant references in the documents
        2. present information in a clear, coherent manner
        3. combine information from all available sources into a single comprehensive answer
        4. identify any specific clause references (like D3.3(a)(i)) in the sources
        5. structure your responses like this:
        "according to what i could find, [combined answer with all relevant information, 
        including specific clause references when available, formatted as 'According to 
        [Document], Clause [X.Y.Z]']
        
        sources: [list all document names used in your answer]"
        
        guidelines for responses:
        - quote specific clauses when relevant
        - include clause numbers (e.g., "according to BCA 2019, Clause D3.3(a)(i)")
        - present all information in one continuous answer, don't split by source
        - maintain a speculative tone without making definitive statements
        - list all sources at the end of your response in a "sources: [list]" format
        - never intermix sources within the answer text
        """

        return OpenAIAgent.from_tools(
            tool_retriever=tool_retriever,
            llm=Settings.llm,
            verbose=True,
            system_prompt=system_prompt,
            max_function_calls=3,
        )

    @staticmethod
    def create_clauses_agent(
        tool_retriever: Any,
    ) -> OpenAIAgent:
        """
        create a specialized agent for extracting specific clauses from standards.
        """
        system_prompt = """
        you are a clause extraction agent. your task is to:
        1. carefully analyze queries about architectural accessibility standards
        2. identify the specific document (like BCA, AS 1428.1) and exact clause numbers that address the query
        3. search thoroughly through multiple accessibility documents to find relevant clauses
        4. return information in the following format:
        
        clause: [exact clause number like D3.3(a)(i) or 3.2.1]
        
        guidelines:
        - be extremely diligent in searching for relevant clauses - almost all accessibility queries will have a matching clause
        - perform extensive searches across multiple documents
        - for bathroom or toilet facilities, search for terms like "sanitary", "WC", "toilet", "bathroom", "shower"
        - specifically check Accessibility Standards documents for dimensional requirements
        - if absolutely cannot find a specific clause after thorough searching, return "clause: not found"
        - include the document reference if possible (e.g., "clause: BCA D3.3(a)(i)" or "clause: AS1428.1 3.2.1")
        """
        return OpenAIAgent.from_tools(
            tool_retriever=tool_retriever,
            llm=Settings.llm,
            verbose=True,
            system_prompt=system_prompt,
            max_function_calls=3,
        )
