# SparkAI Chatbot

A Python-based AI chatbot application that leverages various language models and vector databases for document processing, retrieval, and generation.

## Project Overview

This project is a FastAPI-based application that provides AI-powered chatbot functionality with the following features:

- Document processing and vector storage using ChromaDB
- Integration with multiple LLM providers (OpenAI, Anthropic, Google, etc.)
- PDF and document parsing capabilities
- Diagram generation
- Evidence retrieval from documents
- Template generation

## Setup and Installation

### Quick Setup

1. Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Run uv sync to create a virtual environment and install dependencies:

```bash
uv sync
```

That's it! This will create a virtual environment with Python and install all dependencies.

### VSCode Setup

After running `uv sync`, select the Python interpreter in VSCode:

1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
2. Type "Python: Select Interpreter"
3. Select the interpreter in the `.venv` folder that was created by uv

### Alternative Setup (if needed)

If you encounter issues with the quick setup, you can try these more specific commands:

```bash
# Create virtual environment
uv venv --python=python1

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows

# Install dependencies
uv install -e .
```

## Running the Application

Start the application with:

```bash
python main.py
```

## Testing

The project uses pytest for testing. Run tests with:

```bash
uv run pytest
```

### Test Structure

- Tests are located in the `tests/` directory
- Test data is stored in `tests/test_data/`
- Configuration for pytest is in `pytest.ini`

Key test files:
- `test_evidence_retrieval.py`: Tests for document evidence retrieval
- `test_chroma.py`: Tests for ChromaDB integration
- `test_diagram.py`: Tests for diagram generation
- `test_generate_templates.py`: Tests for template generation

## Code Quality

### Pylint

The project uses pylint for code quality checks. Configuration is in `.pylintrc`.

Run pylint with:

```bash
uv run pylint api.py api_modules/ services/ utils.py
```

## Project Structure

- `api.py`: Main FastAPI application
- `api_modules/`: API endpoint modules
- `services/`: Service layer components
- `utils.py`: Utility functions
- `config.py`: Configuration settings
- `tests/`: Test files
- `chroma_db/`: Vector database storage

## Environment Variables

The application uses environment variables for configuration. Copy `.env.example` to `.env` and adjust settings as needed.

## Contributing

1. Ensure code passes pylint checks
2. Write tests for new functionality
3. Follow existing code style and patterns
