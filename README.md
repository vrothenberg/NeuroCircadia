# NeuroCircadia

NeuroCircadia is an advanced conversational AI designed to support health optimization and circadian rhythm insights. Powered by LangChain, Chainlit, and Google Vertex AI, the application utilizes Retrieval-Augmented Generation (RAG) to answer user questions with relevant knowledge base insights. This repository includes two main components: `chainlit-app` for user-facing chat and `langserve-app` for backend management and retrieval.

## Prerequisites

### Installation

To install dependencies, ensure you have [Python 3.8+](https://www.python.org/downloads/) and [Conda](https://docs.conda.io/en/latest/miniconda.html) or another virtual environment solution installed.

**Chainlit**  
```bash
pip install chainlit
```

**Langserve**  
```bash
pip install "langserve[all]"
```

**Google SDK and Vertex AI**  
```bash
conda install -c conda-forge google-cloud-sdk
pip install google-cloud-aiplatform
```

**LangChain & Google Vertex AI Integrations**  
```bash
pip install langchain langchain-core langchain-community langchain-google-vertexai
```

### Configuration

1. **Google Service Account**  
   Add the path to your service account keys in your `~/.bashrc` or `~/.zshrc` file:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/rbio-p-datasharing-b5c1d9a2deba.json"
   ```

2. **Environment Variables**  
   Ensure you have any additional environment variables configured (e.g., for Qdrant, database credentials, etc.).

## Directory Structure

```
NeuroCircadia/
├── README.md
├── chainlit-app/
│   ├── app.py               # Chainlit app with chat initialization and message handling
│   ├── chatbot_memory.py     # Pipeline setup for conversation memory
│   ├── data_layer.py         # Data layer integration
│   ├── langchain_nb.ipynb    # Notebook for LangChain model development
│   ├── models.py             # Model configuration and setup
│   ├── public/               # Public assets (static files, icons)
│   └── rag_memory.py         # Retrieval-Augmented Generation (RAG) pipeline setup
└── langserve-app/
    ├── Dockerfile            # Dockerfile for LangServe deployment
    ├── app/                  
    │   ├── chatbot.py        # Chatbot-specific endpoint
    │   ├── rag_qa.py         # RAG-based question-answering endpoint
    │   └── server.py         # FastAPI server setup
    └── pyproject.toml        # Project configuration and dependencies
```

## Quick Start

### Run Chainlit Server

From the `NeuroCircadia/chainlit-app` directory, start the Chainlit server:
```bash
chainlit run app.py -w
```

### Run LangServe Server

From the `NeuroCircadia/langserve-app` directory, you can build and run the LangServe backend. Customize the Dockerfile and pyproject.toml for deployment environments as needed.
