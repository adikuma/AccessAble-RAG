import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import json
import re

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models import Relevance, DynamicRampCompliance, BaseRampCompliance
from llama_index.core.llms import ChatMessage
from exa_py import Exa
from dotenv import load_dotenv
import asyncio

load_dotenv()

MAX_HISTORY = 10
exa = Exa(os.getenv("EXA_API_KEY"))

# base compliance parameters to extract (common for all compliance checks)
BASE_COMPLIANCE_QUERIES = [
    {
        "field": "ramp_run",
        "query": "What is the maximum ramp run length (distance of a single sloped section) before a landing is required according to accessibility standards?",
        "type": float,
    },
    {
        "field": "ramp_landing_length",
        "query": "What is the minimum required size, dimension, or length for a level platform on a ramp landing?",
        "type": float,  # assuming mm, store as float
    },
    {
        "field": "ramp_width",
        "query": "What is the minimum required clear width for an accessible ramp?",
        "type": float,
    },
    {
        "field": "path_width",
        "query": "What is the minimum required clear width for the path of travel leading to or from an accessible ramp?",
        "type": float,
    },
]

# special query to get all gradients and their max run lengths in one go
GRADIENT_QUERY = {
    "field": "gradient_max_lengths",
    "query": "List all the different ramp gradients (like 1:12, 1:14, etc.) mentioned in the standards, and for each gradient, specify the maximum allowed horizontal run length before a landing is required.",
    "type": dict,
}

# add history
def add_history(user_query: str, assistant_response: str):
    from main import conversation_history

    conversation_history.append({"role": "user", "content": user_query})
    conversation_history.append({"role": "assistant", "content": assistant_response})

    # remove oldest messages if history exceeds the limit
    # use // 2 because we add messages in pairs (user, assistant)
    max_pairs = MAX_HISTORY // 2
    current_pairs = len(conversation_history) // 2
    while current_pairs > max_pairs:
        # remove the oldest pair (user and assistant)
        conversation_history.pop(0)
        conversation_history.pop(0)  # remove twice for the pair
        current_pairs -= 1
        logging.debug(
            f"trimmed conversation history. new length: {len(conversation_history)}"
        )


# create context
def create_context(current_query: str) -> str:
    from main import conversation_history

    if not conversation_history:
        # if history is empty, just use the current query
        # adding the "user:" prefix for consistency with history format
        return f"User: {current_query}"  # agent might handle final prompt structure

    # format existing history
    formatted_history = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history
    )

    # combine history with the current query
    # let the agent handle the final "Assistant:" prompt if needed by its internal logic
    full_prompt = f"Conversation History:\n{formatted_history}\n\nCurrent User Query: {current_query}"
    logging.debug(
        f"created context prompt:\n{full_prompt}"
    )  # log the created prompt for debugging
    return full_prompt


# llm as a judge
def llm_as_judge(query: str, response: str) -> Optional[Relevance]:
    from main import global_llm  # import from main

    if global_llm is None:
        logging.error("llm_as_judge: Global LLM not initialized.")
        return None

    try:
        sllm = global_llm.as_structured_llm(output_cls=Relevance)

        # define the prompt for the judge LLM
        prompt = f"""
        You are an impartial relevance judge. Your task is to evaluate how relevant the provided 'Response' is to the given 'Query'. 
        Focus *only* on relevance: Does the response address the core subject and intent of the query? 
        Do not evaluate factual accuracy, completeness, or writing style.

        Assign a relevance score as a float between 0.0 and 1.0, according to these guidelines:
        - 1.0: The response directly and clearly addresses the main subject and intent of the query.
        - 0.7 - 0.9: The response addresses the main subject of the query but might miss some nuances or focus slightly tangentially.
        - 0.4 - 0.6: The response is related to the topic of the query but doesn't directly answer the specific question asked. It might answer a related question.
        - 0.1 - 0.3: The response mentions keywords from the query but is largely off-topic or irrelevant to the user's likely intent.
        - 0.0: The response is completely irrelevant to the query topic.

        **Additional instruction:** If the response includes statements of uncertainty (e.g., "I don't know", "not sure", etc.), assign a lower relevance score because such language suggests that the core subject or intent of the query was not adequately addressed.

        Query: '{query}'

        Response: '{response}'

        Provide your assessment of the relevance as a float between 0.0 and 1.0, along with a brief explanation.
        """

        #convert to chat message
        prompt = ChatMessage.from_str(prompt)
    
        # call the LLM with the structured output class
        result = sllm.chat([prompt])
        # i get a json result as an output convert to dict
        result_json = result.raw.dict()
        logging.info(f"Result from llm_as_judge: {result_json}\n")
        logging.info(f"Relevance score: {result_json['relevance']} - {result_json['reasoning']}")
        return result_json

    except Exception as e:
        logging.error(f"Error in llm_as_judge: {str(e)}")
        return None


async def web_search(query: str) -> str:
    from main import global_llm
    try:
        # Use exa for search
        search_results = exa.search_and_contents(
            query, num_results=5, use_autoprompt=True
        )

        if not search_results or not getattr(search_results, "results", None):
            logging.error("No search results were returned from EXA.")
            return "No search results found for your query."

        logging.info("Search results were found for your query.")
        logging.info(f"Search results: {search_results}")

        # Build context for LLM from search results, handling None values
        search_context = "\n\n".join(
            [
                f"Source {i+1}: {result.title or 'No Title'}\n"
                f"URL: {result.url or 'No URL'}\n"
                f"Content: {(result.text or '')[:1000]}..."
                for i, result in enumerate(search_results.results)
            ]
        )

        # format the prompt for the LLM
        prompt = f"""
        SEARCH QUERY: {query}

        SEARCH RESULTS:
        {search_context}

        Using the search results above, provide a comprehensive answer to the search query.
        Focus on providing factual information from the search results.
        If the search results don't contain relevant information to answer the query, state that clearly.
        """
        logging.info(f"Prompt: {prompt}")

        # since `global_llm.complete` is synchronous, run it in a thread.
        response = await asyncio.to_thread(global_llm.complete, prompt)
        logging.info(f"Response: {response}")
        return response

    except Exception as e:
        logging.error(f"Error in web search: {str(e)}")
        return f"I apologize, but I couldn't find reliable information to answer your question. Error: {str(e)}"
