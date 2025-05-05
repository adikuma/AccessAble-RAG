# Compliance-RAG

A retrieval-augmented generation system for architectural accessibility compliance.

## Directory Structure

```
.
├── .env                   # Environment variables
├── .gitignore             # Git ignore patterns
├── data/                  # Compliance documents and indices
│   ├── Accessibility Clauses 30.xlsx
│   ├── Accessibility Standards for Inclusivity.pdf
│   ├── Accessibility in the built environment 2019.pdf
│   ├── House of Quality.csv
│   ├── drive_config.py    # Google Drive sync configuration
│   ├── drive_sync.py      # Google Drive synchronization
│   └── user insights.txt  # User empathy data
├── main.py                # FastAPI application entry point
├── prompt_templates.py    # LLM prompt templates
├── requirements.txt       # Dependencies
└── src/                   # Core application code
    ├── agents/            # Agent system components
    ├── config/            # Application settings
    ├── indexing/          # Document processing and retrieval
    ├── models.py          # Data models
    └── utils.py           # Shared utilities
```

## How It Works

1. **Document Synchronization**

   - Syncs accessibility standards documents from Google Drive
   - Stores them locally with versioning
2. **Document Indexing**

   - Parses PDFs, spreadsheets, and text into semantic chunks
   - Creates vector embeddings for semantic search
   - Builds persistent indices for each document
3. **Multi-Agent System**

   - Document agents: Extract exact information from specific documents
   - Clauses agent: Identifies specific accessibility clause references
   - Top agent: Orchestrates document agents to answer complex queries
   - Insights agent: Provides empathetic context for accessibility concerns
4. **API Endpoints**

   - `/query`: Answers compliance questions with clause references
   - `/empath`: Provides empathetic accessibility design insights
   - `/general-compliance`: Extracts technical parameters for ramps
   - `/research-compliance`: Research-specific compliance parameters
5. **LLM Processing**

   - Uses OpenAI models for text generation and embeddings
   - Implements structured output extraction for compliance parameters
   - Provides gradient calculations for ramp specifications

Built with FastAPI, LlamaIndex, and OpenAI.
