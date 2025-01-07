import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.schema import SystemMessage, HumanMessage
import numpy as np
from typing import List, Tuple

# Set up Ollama environment
os.environ['OLLAMA_HOST'] = '127.0.0.1:11434'

# Load the dataset
dataset = []
with open('datasets/cat-facts.txt', 'r', encoding="utf8") as file: # Change path here
    dataset = file.readlines()
    print(f'Loaded {len(dataset)} entries')

## Define Cat facts Chatbot

class CatFactsChatbot:
    def __init__(self):
        # Initialize LangChain models
        self.embedding_model = OllamaEmbeddings(
            model="hf.co/CompendiumLabs/bge-base-en-v1.5-gguf",
            model_kwargs={"device": "cuda"}
        )
        self.language_model = ChatOllama(
            model="llama3.2",
            temperature=0.7
        )
        self.vector_db = []  # Will store (chunk, embedding) tuples
        
    def add_chunk_to_database(self, chunk: str):
        """Create embedding for a chunk and add it to vector database"""
        try:
            embedding = self.embedding_model.embed_query(chunk)
            self.vector_db.append((chunk, embedding))
        except Exception as e:
            print(f"Failed to process chunk. Error: {e}")

    def initialize_database(self, dataset: List[str]):
        """Initialize the vector database with all chunks"""
        for i, chunk in enumerate(dataset):
            self.add_chunk_to_database(chunk)
            print(f'Added chunk {i+1}/{len(dataset)} to the database')

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def retrieve(self, query: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """Find most relevant facts for a given query"""
        try:
            query_embedding = self.embedding_model.embed_query(query)
            
            similarities = []
            for chunk, embedding in self.vector_db:
                similarity = self.cosine_similarity(query_embedding, embedding)
                similarities.append((chunk, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_n]
            
        except Exception as e:
            print(f"Failed to retrieve context for query: {query}. Error: {e}")
            return []

    def get_response(self, query: str, retrieved_knowledge: List[Tuple[str, float]]) -> str:
        """Generate response using the language model"""
        context_text = '\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])
        instruction_prompt = f'''You are a helpful chatbot.
        Use only the following pieces of context to answer the question. Don't make up any new information:
        {context_text}'''
        
        messages = [
            SystemMessage(content=instruction_prompt),
            HumanMessage(content=query)
        ]
        
        return self.language_model.stream(messages)


## Main

def main():
    # Initialize chatbot
    chatbot = CatFactsChatbot()
    chatbot.initialize_database(dataset)
    
    while True:  # Continuous loop
        # Get user query
        input_query = input('\nAsk me a question (or type "exit" to quit): ')
        
        # Check for exit command
        if input_query.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Retrieve relevant knowledge
        retrieved_knowledge = chatbot.retrieve(input_query)
        
        print('\nRetrieved knowledge:')
        for chunk, similarity in retrieved_knowledge:
            print(f' - (similarity: {similarity:.2f}) {chunk}')
        
        # Generate and stream response
        print('\nChatbot response:')
        response_stream = chatbot.get_response(input_query, retrieved_knowledge)
        
        for chunk in response_stream:
            print(chunk.content, end='', flush=True)
        print('\n')  # Add extra newline for better formatting

## Run Cat facts ChatBot

if __name__ == "__main__":
    main()