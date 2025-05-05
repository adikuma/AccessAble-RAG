# standard library imports
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# third-party imports
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, SummaryIndex
from llama_index.core.llms import ChatMessage
from llama_index.core.objects import ObjectIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI

# project path setup
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# local module imports
from src.config.settings import initialize_settings
from src.indexing.document_loader import DocumentLoader
from src.indexing.node_parser import NodeParser
from src.indexing.index_builder import IndexBuilder
from src.agents.tool_builder import ToolBuilder
from src.agents.agent_factory import AgentFactory
from src.models import Query, DynamicRampCompliance, BaseRampCompliance
from src.utils import add_history, create_context, llm_as_judge, web_search, BASE_COMPLIANCE_QUERIES, GRADIENT_QUERY
from prompt_templates import EXTRACTION_PROMPT, GRADIENT_EXTRACTION_PROMPT
from data.drive_sync import DriveSync

# environment setup
load_dotenv()

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# global constants
conversation_history: List[Dict[str, str]] = []
MAX_HISTORY = 10
USE_CONTEXT: bool = True
RESEARCH_FILE = "Accessibility Standards for Inclusivity"
CLAUSE_FILE = "Accessibility Clauses 30"
USER_INSIGHTS_FILE = "user insights"

# initialize fastapi app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# check and log credentials
google_creds = os.getenv('GOOGLE_CREDENTIALS')
if not google_creds:
    logging.error("google_credentials not found in environment")
else:
    logging.info("google_credentials found in environment")

folder_id = os.getenv('DRIVE_FOLDER_ID')
if not folder_id:
    logging.error("drive_folder_id not found in environment")
else:
    logging.info("drive_folder_id found in environment")

# global variables
global_agent = None
global_llm = None 
global_research_agent = None
global_clause_agent = None
global_insights_agent = None
drive_handler = None
rag_lock = asyncio.Lock()

# rag initialization
async def intialize_rag():
    # make agents global
    global drive_handler, global_agent, global_llm, global_research_agent, global_clause_agent, global_insights_agent

    # define the specific document name for the research agent (no extension)
    research_document_name = RESEARCH_FILE

    try:
        logging.info("starting drive and rag initialization")
        if drive_handler is None:
            drive_handler = DriveSync()
        else:
            logging.info("using existing drive handler")

        # acquire lock to prevent race conditions during sync/init
        async with rag_lock:
            logging.info("--- starting drive sync ---")
            num_files = drive_handler.sync_files()
            logging.info(f"synced {num_files} files from drive to data directory")

            # log files present in data directory after sync
            data_dir = Path('data')
            logging.info("files currently in data directory:")
            for file in data_dir.glob('*'):
                if file.is_file():
                    logging.info(f"- {file.name} ({file.stat().st_size} bytes)")

            logging.info("initializing rag system")
            settings = initialize_settings()

            # load documents from the data directory
            loader = DocumentLoader()
            documents_dict = loader.load_documents()
            logging.info(f"loaded {len(documents_dict)} document sets from loader")

            # initialize components
            index_builder = IndexBuilder()
            all_tools = []
            node_parser = NodeParser()
            research_specific_tools = None
            clause_specific_tools = None
            user_insights_tools = None

            # process each document set
            for idx, (doc_name, docs) in enumerate(documents_dict.items()):
                logging.info(f"processing '{doc_name}'...")

                # check if index already exists
                persist_dir = Path(index_builder.base_dir) / doc_name
                if persist_dir.exists():
                    # load existing index without parsing documents
                    logging.info(f"loading existing vector index for '{doc_name}'...")
                    vector_index = load_index_from_storage(
                        StorageContext.from_defaults(persist_dir=str(persist_dir))
                    )
                    # create a small summary index from the document metadata
                    summary_index = SummaryIndex(docs)
                    logging.info(f"loaded existing index for '{doc_name}' without re-parsing")
                else:
                    # only parse documents if the index doesn't exist
                    logging.info(f"index not found for '{doc_name}', parsing documents...")
                    nodes = node_parser.parser.get_nodes_from_documents(docs)
                    vector_index, summary_index = index_builder.build_indexes(
                        doc_name, nodes
                    )

                # create tools for this document
                doc_tools = ToolBuilder.create_document_tools(
                    doc_name, vector_index, summary_index, index_builder=index_builder
                )

                # check if this is a special document and store its tools accordingly
                if doc_name == research_document_name:
                    logging.info(f"found target document '{research_document_name}', storing its tools for research agent.")
                    research_specific_tools = doc_tools
                    continue
                
                if doc_name == CLAUSE_FILE:
                    logging.info(f"found target document '{CLAUSE_FILE}', storing its tools for clause agent.")
                    clause_specific_tools = doc_tools
                    continue

                if doc_name == USER_INSIGHTS_FILE:
                    logging.info(f"found target document '{USER_INSIGHTS_FILE}', storing its tools for user insights agent.")
                    user_insights_tools = doc_tools
                    continue

                # create a specific agent for this document (for global agent tools)
                agent = AgentFactory.create_document_agent(doc_name, doc_tools)

                # sanitize tool name
                sanitized_doc_name = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_name)
                tool_name_prefix = f"tool_{sanitized_doc_name.lower()}"
                tool_name = re.sub(r'_+', '_', tool_name_prefix).strip('_')
                if not tool_name:
                    tool_name = f"tool_document_{idx}"

                # handle tool name length limit
                max_len = 64
                if len(tool_name) > max_len:
                    original_name = tool_name
                    tool_name = tool_name[:max_len]
                    tool_name = tool_name.strip('_-')
                    logging.warning(f"tool name '{original_name}' for document '{doc_name}' truncated to '{tool_name}' due to 64-character limit.")

                tool_description = (
                    f"contains compliance standards and guidelines about '{doc_name}'. "
                    f"use this tool for specific questions, validation, or summaries regarding '{doc_name}'."
                )

                logging.info(f"using sanitized tool name '{tool_name}' for document '{doc_name}'")

                # create query engine tool
                doc_tool = QueryEngineTool(
                    query_engine=agent,
                    metadata=ToolMetadata(
                        name=tool_name,
                        description=tool_description,
                    ),
                )
                all_tools.append(doc_tool)
                logging.info(f"created agent and tool '{tool_name}' for '{doc_name}' (for global agent)")

            # create specialized agents
            # create the top-level agent if there are any tools
            if not all_tools:
                logging.warning("no tools were created for the global agent.")
                global_agent = None
            else:
                logging.info(f"creating object index from {len(all_tools)} tools for global agent...")
                obj_index = ObjectIndex.from_objects(
                    all_tools,
                    index_cls=VectorStoreIndex,
                )

                logging.info("creating top-level global agent...")
                tool_retriever = obj_index.as_retriever(similarity_top_k=1)
                clause_retriever = obj_index.as_retriever(similarity_top_k=3)

                global_agent = AgentFactory.create_top_agent(
                    tool_retriever=tool_retriever
                )
                logging.info("created global agent.")

            # create the research-specific agent if its tools were found
            if research_specific_tools:
                logging.info(f"creating dedicated agent for '{research_document_name}'...")
                global_research_agent = AgentFactory.create_document_agent(
                    doc_name=research_document_name,
                    tools=research_specific_tools
                )
                logging.info(f"created dedicated research agent for '{research_document_name}'.")
            else:
                logging.warning(f"tools for '{research_document_name}' not found. research agent not created.")
                global_research_agent = None
                
            # create the clauses-specific agent if its tools were found
            if clause_specific_tools:
                logging.info(f"creating dedicated agent for '{CLAUSE_FILE}'...")
                global_clause_agent = AgentFactory.create_clauses_agent(
                    tool_retriever=clause_retriever
                )
                logging.info(f"created dedicated clauses agent for '{CLAUSE_FILE}'.")
            else:
                logging.warning(f"tools for '{CLAUSE_FILE}' not found. clauses agent not created.")
                global_clause_agent = None
                
            # create the user insights agent if its tools were found
            if user_insights_tools:
                logging.info(f"creating dedicated agent for '{USER_INSIGHTS_FILE}'...")
                global_insights_agent = AgentFactory.create_document_agent(
                    doc_name=USER_INSIGHTS_FILE, tools=user_insights_tools
                )
                logging.info(f"Created dedicated user insights agent for '{USER_INSIGHTS_FILE}'.")
            else:
                logging.warning(f"tools for '{USER_INSIGHTS_FILE}' not found. User insights agent not created.")
                global_insights_agent = None

            # assign global llm from settings
            global_llm = settings.llm

            if global_llm is None:
                logging.error("settings.llm was not initialized correctly!")
                global_llm = OpenAI(model="gpt-4o-mini")
                logging.warning("using fallback openai model gpt-4o-mini for global_llm")

            logging.info("rag system initialization complete. ready for queries.")

    except Exception as e:
        logging.error(f"error during rag initialization: {str(e)}", exc_info=True)
        raise

# update rag system when changes detected
async def update_rag():
    global drive_handler
    
    if drive_handler and drive_handler.check_for_updates():
        logging.info("updates detected in drive. starting reinitialization process...")
        await intialize_rag()
    else:
        logging.info("no updates found in drive.")

# periodic check for updates
async def periodic_check():
    while True:
        try:
            log_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
            logging.info(f"running periodic drive check at {log_time}")
            await update_rag()
        except Exception as e:
            logging.error(f"error in periodic check: {str(e)}", exc_info=True)
        await asyncio.sleep(86400) 

# compliance processing logic
async def process_compliance(agent, query_text):
    if global_llm is None:
        logging.error("Global LLM not initialized")
        raise HTTPException(status_code=503, detail="RAG system not initialized (LLM)")

    try:
        # create all query tasks at once including gradient query
        all_tasks = []
        query_context = {}

        # add base parameter queries
        for idx, item in enumerate(BASE_COMPLIANCE_QUERIES):
            specific_query = item["query"]
            field_name = item["field"]
            task = agent.aquery(specific_query)
            all_tasks.append(task)
            query_context[idx] = {"field": field_name, "query": specific_query}

        # add gradient discovery query
        gradient_idx = len(all_tasks)
        gradient_query = GRADIENT_QUERY["query"]
        all_tasks.append(agent.aquery(gradient_query))
        query_context[gradient_idx] = {"field": "gradient_max_lengths", "query": gradient_query}

        # run all queries concurrently in a single gather operation
        logging.info(f"Running {len(all_tasks)} agent queries concurrently...")
        all_results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # process results
        responses = []
        gradient_text = ""

        for idx, result in enumerate(all_results):
            context = query_context[idx]
            field = context["field"]
            original_query = context["query"]

            if isinstance(result, Exception):
                logging.error(f"Agent query failed for {field}: {result}")
                response_text = f"Error: {result}"
            elif result is None:
                logging.warning(f"Agent query returned None for {field}")
                response_text = "No response"
            else:
                response_text = str(result)

            if field == "gradient_max_lengths":
                gradient_text = response_text
            else:
                responses.append(f"--- response for query ({field}):\n{response_text}\n---")

        # extract structured data from responses
        base_combined_text = "\n\n".join(responses)
        base_extraction_prompt = EXTRACTION_PROMPT.format(combined_text=base_combined_text)
        gradient_extraction_prompt = GRADIENT_EXTRACTION_PROMPT.format(text=gradient_text)

        base_msg = ChatMessage.from_str(base_extraction_prompt)
        gradient_msg = ChatMessage.from_str(gradient_extraction_prompt)

        # create the structured LLM instance for base parameters
        base_structured_llm = global_llm.as_structured_llm(output_cls=BaseRampCompliance)

        # run both LLM calls concurrently
        base_task = base_structured_llm.achat([base_msg])
        gradient_task = global_llm.achat([gradient_msg])

        base_response, gradient_response = await asyncio.gather(base_task, gradient_task)

        # extract and format results
        base_params = base_response.raw.dict()
        gradient_json_str = gradient_response.message.content

        try:
            # try to extract just the JSON part if there's extra text
            json_match = re.search(r'(\{.*\})', gradient_json_str, re.DOTALL)
            if json_match:
                gradient_json_str = json_match.group(1)

            gradient_data = json.loads(gradient_json_str)
            gradient_lengths = gradient_data.get("gradient_max_lengths", {})
            dynamic_compliance = DynamicRampCompliance(
                ramp_run=base_params.get("ramp_run"),
                ramp_landing_length=base_params.get("ramp_landing_length"),
                ramp_width=base_params.get("ramp_width"),
                path_width=base_params.get("path_width"),
                gradient_max_lengths=gradient_lengths
            )

            return dynamic_compliance.dict()

        except Exception as e:
            logging.error(f"Error processing gradient data: {str(e)}")
            logging.debug(f"Raw gradient response: {gradient_json_str}")

            return {
                "ramp_run": base_params.get("ramp_run"),
                "ramp_landing_length": base_params.get("ramp_landing_length"),
                "ramp_width": base_params.get("ramp_width"),
                "path_width": base_params.get("path_width"),
                "error": f"Failed to parse gradient data: {str(e)}"
            }

    except Exception as e:
        logging.error(f"Unhandled error during compliance processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing compliance query: {str(e)}")

# api endpoints
@app.post("/query")
async def process_query(query: Query):
    logging.info(f"processing query with clauses: {query.text[:100]}...")
            
    if global_agent is None:
        logging.error("rag system not initialized. cannot process query.")
        raise HTTPException(status_code=503, detail="rag system not initialized or failed to initialize.")
    if global_llm is None:
        logging.error("global llm not initialized. cannot process query.")
        raise HTTPException(status_code=503, detail="global llm not initialized.")
    if global_clause_agent is None:
        logging.error("clause agent not initialized. cannot process query with clauses.")
        raise HTTPException(status_code=503, detail="clause agent not initialized.")
    
    try:
        # get normal response first
        logging.info("sending query to top agent...")
        if USE_CONTEXT:
            prompt = create_context(query.text)
        else:
            prompt = query.text
        response = global_agent.query(prompt)
        
        # get clause information
        clause_prompt = (
            f"find the exact clause reference for this information. only return 'clause: [clause number]', nothing else.\n\n"
            f"question: {query.text}\nanswer: {response}"
        )
        clause_info_obj = global_clause_agent.query(clause_prompt)
        clause_text = str(clause_info_obj)
        print("clause info:", type(clause_info_obj), "\n", clause_text)
        
        # if clause text indicates no clause is found, remove it from the final response
        logging.info(f"clause text: {clause_text}")
        if clause_text.strip().lower() in ["clause: not found", "not found"]:
            final_response = response
        else:
            final_response = f"{response}\n\n{clause_text}"
        
        # check relevance and fallback to web search if needed
        relevance_result = llm_as_judge(query.text, final_response)
        logging.info(f"relevance score: {relevance_result}")
        
        if relevance_result and relevance_result['relevance'] < 0.6:
            logging.info("searching the web for more information")
            final_response = await web_search(query.text)
        
        logging.info("query processed successfully.")
        add_history(query.text, str(final_response))  # only add original to history
        return {"response": str(final_response)}
    except Exception as e:
        logging.error(f"error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"error processing query: {str(e)}")

@app.post("/empath", response_model=Dict)
async def empath(query: Query):
    logging.info(f"processing empath query for user insights. Question: {query.text}")
    if global_llm is None:
        logging.error("LLM not initialized, cannot process empath query.")
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    try:
        # load user insights
        insights_file_path = Path(__file__).parent / "data" / "user insights.txt"
        if not insights_file_path.is_file():
            logging.error(f"User insights file not found at: {insights_file_path}")
            static_insights = (
                "Key Insights:\n"
                "1. Users often face navigational challenges due to unclear signage and confusing routes.\n"
                "2. They experience frustration, embarrassment, and a loss of independence when confronting accessibility barriers.\n"
                "3. Inclusive design, better signage, and wider pathways are needed.\n"
            )
            logging.warning("Using default fallback insights as file was not found.")
        else:
            with open(insights_file_path, "r", encoding="utf-8") as f:
                static_insights = f.read()
            logging.info(f"Loaded static insights: {static_insights[:100]}...")

        # prepare prompt
        system_content = (
            f"{static_insights}\n\n"
            "Instructions: You are an empathetic design consultant. Examine the following question to determine whether it pertains to "
            "architectural design, accessibility, or related design decisions. If the question is not related to these topics, return an empty response. "
            "Otherwise, provide a concise empathetic response (1-2 sentences) explaining how the design decision or topic might affect users emotionally."
        )
        user_content = f"Question: {query.text}"
        messages = [
            ChatMessage(role="system", content=system_content),
            ChatMessage(role="user", content=user_content),
        ]

        # generate response
        chat_response = await global_llm.achat(messages)
        insights_response_text = chat_response.message.content

        logging.info(f"Generated empathy response: {insights_response_text}")
        return {"response": insights_response_text}

    except FileNotFoundError:
        logging.error(f"Critical error: User insights file not found at {insights_file_path} and no fallback handled.")
        raise HTTPException(status_code=500, detail="Internal server error: Could not load required insights data.")
    except Exception as e:
        logging.error(f"Error during empath processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing empathy query: {str(e)}")

@app.post("/general-compliance", response_model=Dict) 
async def general_compliance(query: Query):
    logging.info(f"Processing general compliance query with dynamic gradients. Context: {query.text[:100]}...")

    if global_agent is None:
        logging.error("Global agent not initialized")
        raise HTTPException(status_code=503, detail="RAG system not initialized (agent)")
    
    # process compliance with the global agent using the dynamic approach
    return await process_compliance(global_agent, query.text)

@app.post("/research-compliance", response_model=Dict)
async def research_compliance(query: Query):
    logging.info(f"Processing research compliance query with dynamic gradients. Context: {query.text[:100]}...")

    if global_research_agent is None:
        logging.error("Research agent not initialized")
        raise HTTPException(status_code=503, detail="Research agent not available")
    
    # process compliance with the research agent using the dynamic approach
    return await process_compliance(global_research_agent, query.text)

@app.post("/force-sync")
async def force_sync():
    logging.info("force sync requested via api.")
    try:
        await intialize_rag() 
        logging.info("force sync and reinitialization completed successfully.")
        return {"status": "success", "message": "sync and reinitialization completed"}
    except Exception as e:
        logging.error(f"error during force sync: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"error during force sync: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "time": datetime.now().isoformat()}

# startup events
@app.on_event("startup")
async def startup_event():
    logging.info("application startup: initiating rag system...")
    try:
        await intialize_rag()
        if "RENDER" in os.environ:
            asyncio.create_task(periodic_check())
            logging.info("started periodic document check task for render deployment")
    except Exception as e:
        logging.error(f"startup error during rag initialization: {str(e)}", exc_info=True)

# main entry point
if __name__ == "__main__":
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        logging.critical(f"application crashed: {str(e)}", exc_info=True)