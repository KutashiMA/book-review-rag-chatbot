# Book Review RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot system that allows users to query book reviews using semantic search. The system can handle both specific book queries and general questions about book reviews, leveraging embeddings for semantic search and language models for natural response generation.

## Overview

This project implements a RAG system that:
- Loads and processes text data (cat facts or book reviews) for demonstration
- Creates embeddings for semantic search  
- Maintains a vector database of content
- Handles various types of queries
- Generates natural language responses based on relevant content
- Optionally includes factuality checking for response validation

## Running in Google Colab

1. Open the ```Run_In_Colab.ipynb``` notebook in Google Colab
2. Select GPU runtime (Edit -> Notebook settings -> Hardware accelerator -> GPU)
3. Install and setup Ollama in Xterm section
4. In the opened terminal window, run:
   - ```curl -fsSL https://ollama.com/install.sh | sh```
   - ```ollama serve &```
5. Run the notebook sections:
   - Cat Facts Demo: Simple RAG implementation with a small dataset
   - Book Reviews Chatbot: Advanced RAG system for querying book reviews  
   - Book Reviews with Factuality: Enhanced version with response validation

## Running Locally

### Requirements
- Python 3.8+
- Ollama
- Required Python packages:
   - ```pip install langchain_community scikit-learn==1.6.0 nltk```

### Ollama Setup

1. Install Ollama from [ollama.com](http://ollama.com)
2. Pull required models:
   - ```ollama pull llama3.2```
   - ```ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf```
   - ```ollama pull bespoke-minicheck```

### Running the Scripts

1. Simple Cat Facts Demo:
   - ```python cat_rag.py```

2. Book Reviews Chatbot:
   - ```python book_review_rag.py```

3. Book Reviews with Factuality Check:
   - ```python fact_book_review_rag.py```

## Features

- **Semantic Search**: Uses embeddings to retrieve content based on meaning
- **Multiple Query Types**:
   - Specific book queries
   - Content keyword searches
   - General questions
- **Natural Language Responses**: Generates coherent responses from retrieved content
- **Optional Factuality Validation**: Ensures generated claims align with source content
- **Progress Tracking**: Visual feedback during processing steps

## System Architecture

1. **Data Loading & Processing**
   - Loads content from text/CSV files
   - Preprocesses text for embedding generation

2. **Vector Database**
   - Stores content and embeddings for similarity search

3. **Query Processing**
   - Creates query embeddings
   - Retrieves relevant content
   - Combines semantic and keyword search results

4. **Response Generation**
   - Generates responses using retrieved content
   - Optional factuality validation

## Limitations

- Requires GPU for optimal performance
- Depends on Ollama installation
- Limited by quality and coverage of source content
- Response generation time varies with dataset size

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Uses [Ollama](https://ollama.com) for embeddings and language model inference
- Based on [RAG implementation guide](https://huggingface.co/blog/ngxson/make-your-own-rag) from Hugging Face
- [Colab implementation](https://medium.com/cool-devs/how-to-run-ollama-on-google-colab-ffc1713b64c9) based on Medium article
- Incorporates factuality checking using [Bespoke Minicheck](https://ollama.com/library/bespoke-minicheck)
