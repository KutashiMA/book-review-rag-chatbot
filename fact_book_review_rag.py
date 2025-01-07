# Required imports
import os
import csv
import nltk
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.schema import SystemMessage, HumanMessage
import numpy as np
from typing import List, Tuple, Dict
nltk.download("punkt_tab", quiet=True)

# Set up Ollama environment
os.environ['OLLAMA_HOST'] = '127.0.0.1:11434'

### Define Functions

def load_csv(file_path: str) -> List[Dict]:
    """Load and parse the CSV dataset containing book reviews"""
    dataset = []
    with open(file_path, 'r', encoding="utf8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['product_title'] and row['review_body']:
                review_data = {
                    'product_title': row['product_title'],
                    'review_body': row['review_body']
                }
                dataset.append(review_data)
    print(f'Loaded {len(dataset)} reviews from CSV')
    return dataset


def is_general_chat(query: str) -> bool:
    """Check if query is general chat"""
    general_chat_phrases = {
        'hi', 'hello', 'hey', 'how are you', 'good morning', 'good afternoon', 
        'good evening', 'whats up', "what's up"
    }
    return query.lower().strip() in general_chat_phrases

def is_book_related_query(query: str) -> bool:
    """Check if query is related to books/reviews"""
    book_related_terms = {
        'book', 'review', 'read', 'author', 'novel', 'story', 'recommend', 
        'reading', 'books', 'reviews', 'reader', 'readers', 'recommendation',
        'recommendations', 'title', 'titles', 'literature'
    }
    query_words = set(query.lower().split())
    return bool(query_words & book_related_terms)

### Define Chatbot

class BookReviewChatbot:
    def __init__(self):
        self.embedding_model = OllamaEmbeddings(
            model="hf.co/CompendiumLabs/bge-base-en-v1.5-gguf",
            model_kwargs={"device": "cuda"}
        )
        self.language_model = ChatOllama(
            model="llama3.2",
            temperature=0.7
        )
        self.factuality_model = Ollama(
            model="bespoke-minicheck",
            temperature=0.0
        )
        self.vector_db = []
        self.help_message = """I am a chatbot that can help you with:
        1. Finding specific book reviews (e.g., "Show me reviews that mention romance")
        2. Recommending books based on topics or interests (e.g., "Recommend health books")
        3. Answering questions about books in the Amazon reviews dataset
        Please ask me anything related to these topics!"""

    def add_review_to_database(self, review_data: Dict):
        """Create embedding for a review and add it to vector database"""
        try:
            review_text = f"Book: {review_data['product_title']}\n" \
                         f"Review: {review_data['review_body']}"

            embedding = self.embedding_model.embed_query(review_text)
            self.vector_db.append((review_data, embedding))
        except Exception as e:
            print(f"Failed to process review for {review_data['product_title']}. Error: {e}")

    def initialize_database(self, dataset: List[Dict]):
        """Initialize the vector database with all reviews"""
        for i, review_data in enumerate(dataset):
            self.add_review_to_database(review_data)
            if (i + 1) % 100 == 0:
                print(f'Added review {i+1}/{len(dataset)} to the database')

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def find_explicit_matches(self, query: str, top_n: int = 5) -> List[Tuple[Dict, float]]:
        """Find reviews that explicitly contain search terms"""
        search_terms = query.lower().split()
        # Filter out common words
        stop_words = {'reviews', 'that', 'mention', 'about', 'contain',
                     'have', 'the', 'with', 'and', 'or', 'in', 'on', 'a', 'of'}
        key_terms = [term for term in search_terms if term not in stop_words]

        matches = []
        for review_data, _ in self.vector_db:
            combined_text = (review_data['product_title'] + ' ' +
                           review_data['review_body']).lower()

            # Calculate match score based on number of terms found
            match_score = sum(term in combined_text for term in key_terms)
            if match_score > 0:
                matches.append((review_data, match_score / len(key_terms)))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_n]

    def retrieve(self, query: str, top_n: int = 5) -> List[Tuple[Dict, float]]:
        """Enhanced retrieval combining semantic and keyword search"""
        try:
            # Get semantic search results
            query_embedding = self.embedding_model.embed_query(query)
            semantic_matches = []
            for review_data, embedding in self.vector_db:
                similarity = self.cosine_similarity(query_embedding, embedding)
                semantic_matches.append((review_data, similarity))

            # Get explicit keyword matches
            explicit_matches = self.find_explicit_matches(query)

            # Combine and deduplicate results
            seen_titles = set()
            combined_results = []

            # First add explicit matches
            for review_data, score in explicit_matches:
                if review_data['product_title'] not in seen_titles:
                    combined_results.append((review_data, score))
                    seen_titles.add(review_data['product_title'])

            # Then add semantic matches
            semantic_matches.sort(key=lambda x: x[1], reverse=True)
            for review_data, score in semantic_matches:
                if review_data['product_title'] not in seen_titles:
                    combined_results.append((review_data, score))
                    seen_titles.add(review_data['product_title'])

                if len(combined_results) >= top_n:
                    break

            return combined_results[:top_n]

        except Exception as e:
            print(f"Failed to retrieve context for query: {query}. Error: {e}")
            return []

    def handle_query(self, query: str) -> str:
        """Handle different types of queries appropriately"""
        if is_general_chat(query):
            return f"Hello! {self.help_message}"
        
        if not is_book_related_query(query):
            return f"I apologize, but I can only help with book-related queries. {self.help_message}"
        
        retrieved_knowledge = self.retrieve(query)
        
        if not retrieved_knowledge:
            return "I couldn't find any relevant reviews in my database. Could you try rephrasing your question?"
        
        print('\nRelevant reviews found:')
        for review_data, similarity in retrieved_knowledge:
            print(f"\n- Book: {review_data['product_title']}")
            print(f"  Similarity: {similarity:.2f}")
            print(f"  Review snippet: {review_data['review_body'][:150]}...")
        
        return self.get_response(query, retrieved_knowledge)

    def check_factuality(self, context: str, claim: str) -> str:
        """
        Check if claim is supported by context using bespoke-minicheck
        Args:
            context: The source text to check against
            claim: The statement to verify
        Returns:
            "Yes" if claim is supported, "No" if not
        """
        try:
            prompt = f"Document: {context}\nClaim: {claim}"
            response = self.factuality_model.invoke(prompt)
            return response.strip()
        except Exception as e:
            print(f"Factuality check failed: {e}")
            return "No"  # Default to No if check fails

    def get_response(self, query: str, retrieved_knowledge: List[Tuple[Dict, float]]) -> str:
        """Generate response using the language model with factuality checking"""
        # Format reviews into context
        context_reviews = []
        for review_data, similarity in retrieved_knowledge:
            review_text = (f"Book: {review_data['product_title']}\n"
                        f"Review: {review_data['review_body']}")
            context_reviews.append(review_text)
        
        context_text = "\n\n".join(context_reviews)
        instruction_prompt = f"""You are a helpful book recommendation chatbot. 
        Use only the following book reviews to answer the question. 
        If you can't answer based on these reviews alone, say so.
        Don't make up any information.
        Provide your response in clear, separate sentences that can be fact-checked.

        Reviews:
        {context_text}"""
        
        messages = [
            SystemMessage(content=instruction_prompt),
            HumanMessage(content=query)
        ]
        
        # Get initial response
        initial_response = self.language_model.invoke(messages)
        
        # Split response into sentences for fact checking
        sentences = nltk.sent_tokenize(initial_response.content)
        
        # Perform factuality check on each sentence
        verified_sentences = []
        print("\nFactuality Checking:")
        for sentence in sentences:
            is_factual = self.check_factuality(context_text, sentence)
            if is_factual.lower() == "yes":
                verified_sentences.append(sentence)
                print(f"✓ Verified: {sentence}")
            else:
                print(f"✗ Removed unsupported claim: {sentence}")
        
        # Combine verified sentences
        if verified_sentences:
            return " ".join(verified_sentences)
        else:
            return "I couldn't make any factual claims based on the available reviews. Please try rephrasing your question."


## Main

def main():
    dataset = load_csv('datasets/amazon_books_Data.csv') # Change file path here if necessary

    print("\nInitializing chatbot, this may take a few minutes...")
    chatbot = BookReviewChatbot()
    chatbot.initialize_database(dataset)

    print("\nChatbot ready! You can ask questions about books and reviews.")
    print("Example questions:")
    print(" - What are some good children's books?")
    print(" - Tell me about reviews for [book title]")
    print(" - What do reviewers say about the writing style of popular books?")

    while True:
        input_query = input('\nAsk me about books (or type "exit" to quit): ')
        
        if input_query.lower() == 'exit':
            print("Goodbye!")
            break
        
        response = chatbot.handle_query(input_query)
        print("\nFinal Response:")
        print(response)
        print('\n')


## Run Default Chatbot

if __name__ == "__main__":
    main()

