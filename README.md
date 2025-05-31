# MemoryOS
![logo](logo.png)
**MemoryOS** is a Python library designed to provide conversational AI agents with a with a memory operation system, enabling more coherent, personalized, and context-aware interactions. To achieve comprehensive and efficient memory management for AI agents. Inspired by the memory management principles in operating systems, MemoryOS designs a hierarchical storage architecture and consists of four key modules: memory Storage, Updating, Retrieval, and Generation.

## Latest News

*   **[2025-05-30]**: Initial version of MemoryOS launched! Featuring short-term, mid-term, and long-term persona Memory with automated user profile and knowledge updating.

## Features

*   **Multi-Layered Memory:**
    *   **Short-Term Memory:** Captures the most recent interactions, providing immediate context for ongoing conversations.
    *   **Mid-Term Memory:** Stores significant conversation segments, analyzed and consolidated from short-term memory. It uses a "heat" metric to identify important topics for potential promotion to long-term storage.
    *   **Long-Term Persona Memory:**
        *   **User Profile:** Gradually builds a profile of the user, including preferences, facts, and personality traits.
        *   **User Knowledge:** Stores specific pieces of information provided by the user.
        *   **Assistant Knowledge:** Stores knowledge relevant to the assistant's domain or specific user needs, allowing for more informed responses.
*   **Dynamic Profile & Knowledge Updates:** Automatically analyzes "hot" topics in mid-term memory to update the user's profile and knowledge base, as well as the assistant's knowledge.
*   **Contextual Retrieval:** Retrieves relevant information from all memory layers to provide rich context for generating responses.
*   **Modular Design:** Consists of distinct modules for memory storage (`ShortTermMemory`, `MidTermMemory`, `LongTermMemory`), memory processing (`Updater`), and information retrieval (`Retriever`).
*   **Flexible Configuration:** Allows customization of memory capacities, update thresholds, and LLM models.
*   **OpenAI Integration:** Utilizes OpenAI's language models for tasks like personality analysis, profile updates, and response generation.

## Project Structure

```
memoryos/
├── __init__.py            # Initializes the MemoryOS package
├── __pycache__/           # Python cache directory (auto-generated)
├── long_term.py           # Manages long-term persona memory (user profile, knowledge)
├── memoryos.py            # Main class for MemoryOS, orchestrating all components
├── mid_term.py            # Manages mid-term memory, consolidating short-term interactions
├── prompts.py             # Contains prompts used for LLM interactions (e.g., summarization, analysis)
├── retriever.py           # Retrieves relevant information from all memory layers
├── short_term.py          # Manages short-term memory for recent interactions
├── updater.py             # Processes memory updates, including promoting information between layers
└── utils.py               # Utility functions used across the library
```

## How It Works

1.  **Initialization:** `Memoryos` is initialized with user and assistant IDs, API keys, data storage paths, and various capacity/threshold settings. It sets up dedicated storage for each user and assistant.
2.  **Adding Memories:** User inputs and agent responses are added as QA pairs. These are initially stored in short-term memory.
3.  **Short-Term to Mid-Term Processing:** When short-term memory is full, the `Updater` module processes these interactions, consolidating them into meaningful segments and storing them in mid-term memory.
4.  **Mid-Term Analysis & LPM Updates:** Mid-term memory segments accumulate "heat" based on factors like visit frequency and interaction length. When a segment's heat exceeds a threshold, its content is analyzed:
    *   User profile insights are extracted and used to update the long-term user profile.
    *   Specific user facts are added to the user's long-term knowledge.
    *   Relevant information for the assistant is added to the assistant's long-term knowledge base.
5.  **Response Generation:** When a user query is received:
    *   The `Retriever` module fetches relevant context from short-term history, mid-term memory segments, the user's profile & knowledge, and the assistant's knowledge base.
    *   This comprehensive context is then used, along with the user's query, to generate a coherent and informed response via an LLM.

## Getting Started

### Prerequisites

*   Python >= 3.10
*   pip install -r requirements.txt

### Installation

```bash
git clone https://github.com/your_username/memoryos.git  
cd memoryos
```

### Basic Usage

The `example.py` file provides a simple demonstration:

```python
# example.py - Simple MemoryOS Basic Demo

import os
from memoryos import Memoryos

# --- Basic Configuration ---
USER_ID = "demo_user"
ASSISTANT_ID = "demo_assistant"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"  # Replace with your key
OPENAI_BASE_URL = ""  # Optional: if using a custom OpenAI endpoint
DATA_STORAGE_PATH = "./simple_demo_data"
LLM_MODEL = "gpt-4o-mini"

def simple_demo():
    print("MemoryOS Simple Demo")
    
    # 1. Initialize MemoryOS
    print("Initializing MemoryOS...")
    try:
        memo = Memoryos(
            user_id=USER_ID,
            openai_api_key=OPENAI_API_KEY,
            openai_base_url=OPENAI_BASE_URL,
            data_storage_path=DATA_STORAGE_PATH,
            llm_model=LLM_MODEL,
            assistant_id=ASSISTANT_ID,
            short_term_capacity=7,  
            mid_term_heat_threshold=5,  
        )
        print("MemoryOS initialized successfully!\n")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Add some basic memories
    print("Adding some memories...")
    
    memo.add_memory(
        user_input="Hi! I'm Tom, I work as a data scientist in San Francisco.",
        agent_response="Hello Tom! Nice to meet you. Data science is such an exciting field. What kind of data do you work with?"
    )
     
    test_query = "What do you remember about my job?"
    print(f"User: {test_query}")
    
    response = memo.get_response(
        query=test_query,
    )
    
    print(f"Assistant: {response}")

if __name__ == "__main__":
    simple_demo()
```

To run the example:

1.  Ensure you have your OpenAI API key set in `example.py`.
2.  Execute the script: `python example.py`


## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Citation

If you find this project useful, please consider citing our paper:

```bibtex
```

