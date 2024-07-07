# Langchain Text Summarization

## Overview

This repository demonstrates six different methods for text summarization using Langchain and OpenAI's GPT-3.5-turbo model. Each method is implemented as a separate Python class, and a script is provided to run all six methods sequentially.

## Project Structure

- `summarize_using_langchain.py`: Summarizes text using Langchain's ChatOpenAI model.
- `prompt_templates_text_summarization.py`: Summarizes text using prompt templates.
- `stuff_document_chain_summarization.py`: Summarizes text from a PDF document using the Stuff Document Chain method.
- `map_reduce_summarization.py`: Summarizes large documents using the Map Reduce method.
- `map_reduce_with_custom_prompts.py`: Summarizes text using Map Reduce with custom prompts.
- `refine_chain_summarization.py`: Summarizes text using the Refine Chain method.

## Getting Started

### Prerequisites

Ensure you have Python 3.6 or higher installed on your machine.

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/sairam-penjarla/langchain-text-summarization.git
    ```

2. Navigate to the project directory:

    ```bash
    cd langchain-text-summarization
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Examples

Each summarization method is implemented in its own script. You can run any of these scripts to see the respective summarization method in action.

For example, to run the `SummarizationUsingLangchain` class:

```bash
python summarize_using_langchain.py
