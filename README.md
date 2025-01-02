# Book Review RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot system that allows users to query book reviews using semantic search. The system can handle both specific book queries and general questions about book reviews, leveraging embeddings for semantic search and language models for natural response generation.

## Overview

This project implements a RAG system that:
- Loads and processes book reviews from a CSV dataset.
- Creates embeddings for semantic search.
- Maintains a vector database of reviews.
- Handles both specific book queries and general questions.
- Generates natural language responses based on relevant reviews.
- Includes a factuality-check mechanism to validate generated responses against retrieved reviews.

## Requirements

- Python 3.8+
- Ollama
- CSV module
- NLTK (Natural Language Toolkit)

## Installation

1. Download or clone the repository:
   ```bash
   git clone https://github.com/KutashiMA/book-review-rag-chatbot.git
   ```

2. Install required packages:
   ```bash
   pip install ollama nltk
   ```

3. Download and set up Ollama models:

   First, install Ollama from its official website:  
   [ollama.com](http://ollama.com)

   Then, open a terminal and download the required models using the following commands:
   ```bash
   ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
   ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
   ollama pull bespoke-minicheck
   ```

## Usage

1. Run the chatbot:
   ```bash
   python RAG_Model_Books_v8.py
   ```

2. Interact with the chatbot by entering your questions when prompted. The system supports:
   - Specific book queries (e.g., "What do people think of [Book Title]?")
   - General questions about reviews (e.g., "What books are least recommended based on reviews?")

3. Type "exit" to quit the chatbot.

## Features

- **Semantic Search**: Uses embeddings to retrieve reviews based on meaning, not just keywords.
- **Dual Query Handling**:
  - Book-specific queries filter reviews by exact title matches.
  - General queries use semantic similarity across all reviews.
- **Natural Language Responses**: Generates coherent responses based on retrieved reviews.
- **Factuality Validation**: Ensures generated claims align with retrieved reviews using the `bespoke-minicheck` model.
- **Dynamic Progress Indicators**: Includes visual feedback during data indexing, retrieval, and factuality checks.

## System Architecture

1. **Data Loading & Preprocessing**
   - Loads reviews from a CSV file.
   - Preprocesses text to remove stopwords and truncate long reviews for concise embeddings.
   - Tokenizes and cleans text for embedding generation.

2. **Vector Database**
   - Stores review content and embeddings for efficient similarity search.

3. **Query Processing**
   - Detects query type (book-specific vs. general).
   - Creates query embeddings.
   - Retrieves relevant reviews from the vector database.

4. **Response Generation**
   - Uses a language model to generate responses based on retrieved reviews.
   - Validates generated responses using factuality checks.

5. **Factuality Check**
   - Utilizes the `bespoke-minicheck` model to ensure claims align with the retrieved review context.

## Limitations

- Reviews must follow the specified CSV format.
- Requires local installation of Ollama and specific models.
- Title matching is case-sensitive.
- Review length is truncated to 1000 characters.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the [RAG implementation tutorial from Hugging Face](https://huggingface.co/blog/ngxson/make-your-own-rag).
- Uses Ollama for embeddings and language model inference.
- Inspired by the blog article on reducing hallucinations with Bespoke Minicheck: [Reduce Hallucinations with Bespoke Minicheck](https://ollama.com/blog/reduce-hallucinations-with-bespoke-minicheck)
