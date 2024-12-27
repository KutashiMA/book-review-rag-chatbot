
# Book Review RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot system that allows users to query book reviews using semantic search. The system can handle both specific book queries and general questions about book reviews, using the power of embeddings for semantic search and language models for natural response generation.

## Overview

This project implements a RAG system that:
- Loads and processes book reviews from a CSV dataset
- Creates embeddings for semantic search
- Maintains a vector database of reviews
- Handles both specific book queries and general questions
- Generates natural language responses based on relevant reviews

## Requirements

- Python 3.8+
- Ollama
- CSV module

## Installation

1. Download or clone the repository:

- `git clone https://github.com/KutashiMA/book-review-rag-chatbot.git`


2. Install required packages:

- `pip install ollama`

3. Download and set up Ollama models:

First, install ollama from project's website: 

- [ollama.com](http://ollama.com)

After installed, open a terminal and run the following command to download the required models:

- `ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf`
- `ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF`


## Usage

1. Run the chatbot:
- `python RAG_Model_Books_v8.py`

2. Enter your questions when prompted. The system can handle:
   - Specific book queries (e.g., "What do people think of [Book Title]?")
   - General questions about reviews (e.g., "What books are least recommended based on reviews?")

3. Type "exit" to quit the chatbot.

## Features

- **Semantic Search**: Uses embeddings to find relevant reviews based on meaning, not just keywords
- **Dual Query Handling**: 
  - Book-specific queries filter reviews by exact title matches
  - General queries use semantic similarity across all reviews
- **Natural Language Responses**: Generates coherent responses based on retrieved reviews
- **Review Context Preservation**: Maintains review context in responses including book titles and ratings

## System Architecture

1. **Data Loading & Preprocessing**
   - Loads reviews from CSV
   - Preprocesses text and truncates long reviews
   - Creates embeddings for each review

2. **Vector Database**
   - Stores review content and embeddings
   - Enables efficient similarity search

3. **Query Processing**
   - Detects query type (book-specific vs general)
   - Creates query embeddings
   - Retrieves relevant reviews

4. **Response Generation**
   - Uses language model to generate natural responses
   - Incorporates retrieved review context

## Limitations

- Reviews must be in the specified CSV format
- Requires local installation of Ollama models
- Title matching is case-sensitive
- Review length is truncated to 1000 characters

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the [RAG implementation tutorial from Hugging Face](https://huggingface.co/blog/ngxson/make-your-own-rag)
- Uses Ollama for embeddings and language model inference
