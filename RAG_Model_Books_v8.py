import ollama  # Import the Ollama library for LLM interaction
import csv     # Import CSV module for reading dataset files

def load_csv(file_path):
    """
    Load and parse the CSV dataset containing book reviews
    Args:
        file_path: Path to the CSV file
    Returns:
        List of dictionaries containing book review data
    """
    dataset = []
    with open(file_path, 'r', encoding="utf8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Only include entries that have both a title and review
            if row['product_title'] and row['review_body']:
                chunk = {
                    "product_id": row['product_id'],
                    "product_title": row['product_title'],
                    "review_body": row['review_body'],
                    "star_rating": row['star_rating']
                }
                dataset.append(chunk)
    print(f'Loaded {len(dataset)} entries from CSV')
    return dataset

def preprocess_chunk(chunk):
    """
    Clean and format each review chunk
    Args:
        chunk: Dictionary containing review data
    Returns:
        Preprocessed chunk with cleaned text and truncated review
    """
    return {
        "product_id": chunk["product_id"],
        "product_title": chunk["product_title"],
        "review_body": chunk["review_body"].strip()[:1000],  # Remove whitespace and limit review length
        "star_rating": chunk["star_rating"]
    }

# Load and preprocess the dataset
dataset = [preprocess_chunk(chunk) for chunk in load_csv('datasets/amazon_books_Data.csv')]

# Define the models to use
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'  # Model for creating embeddings
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'   # Model for generating responses

# Initialize vector database to store embeddings
VECTOR_DB = []  # Will store tuples of (chunk, embedding)

def add_chunk_to_database(chunk):
    """
    Create embedding for a chunk and add it to the vector database
    Args:
        chunk: Dictionary containing review data
    """
    try:
        # Create comprehensive text for embedding
        embedding_input = f"""
        Title: {chunk['product_title']}
        Review Content: {chunk['review_body']}
        Rating: {chunk['star_rating']}
        """
        # Generate embedding using Ollama
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=embedding_input)['embeddings'][0]
        VECTOR_DB.append((chunk, embedding))
    except Exception as e:
        print(f"Failed to process chunk: {chunk['product_title'][:50]}... Error: {e}")

# Process each chunk and add to vector database
for i, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)
    print(f'Added chunk {i+1}/{len(dataset)} to the database')

def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors
    Args:
        a, b: Vectors to compare
    Returns:
        Similarity score between 0 and 1
    """
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=5):
    """
    Find most relevant reviews for a given query
    Args:
        query: User's question
        top_n: Number of results to return
    Returns:
        List of (chunk, similarity_score) tuples
    """
    try:
        # Create embedding for the query
        query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
        similarities = []
        # Calculate similarity with each review in database
        for chunk, embedding in VECTOR_DB:
            similarity = cosine_similarity(query_embedding, embedding)
            similarities.append((chunk, similarity))
        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

    except Exception as e:
        print(f"Failed to retrieve context for query: {query}. Error: {e}")
        return []

def is_book_specific_query(query):
    """
    Check if query is about a specific book by looking for book titles
    Args:
        query: User's question
    Returns:
        Boolean indicating if query contains a book title
    """
    for chunk, _ in VECTOR_DB:
        if chunk['product_title'].lower() in query.lower():
            return True
    return False

# Main chatbot loop
while True:
    # Get user input
    input_query = input('Ask me a question (or type "exit" to quit): ')
    if input_query.lower() == "exit":
        print("Exiting the chatbot. Goodbye!")
        break

    # Retrieve relevant reviews
    retrieved_knowledge = retrieve(input_query)

    if not retrieved_knowledge:
        print("I couldn't find any relevant context in the dataset. Please try a different query.")
        continue

    # Filter results based on query type
    if is_book_specific_query(input_query):
        # For book-specific queries, only use reviews matching the title
        context_chunks = [chunk for chunk, similarity in retrieved_knowledge 
                         if chunk['product_title'].lower() in input_query.lower()]
    else:
        # For general queries, use all retrieved reviews
        context_chunks = [chunk for chunk, similarity in retrieved_knowledge]

    if not context_chunks:
        print("No relevant information found.")
        continue

    # Create prompt for the language model
    instruction_prompt = f'''Use these reviews to answer the question:
{''.join([f"Review for {chunk['product_title']}:\n{chunk['review_body']}\nRating: {chunk['star_rating']}\n\n" for chunk in context_chunks])}'''

    try:
        # Generate and stream the response
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': input_query},
            ],
            stream=True,
        )

        # Print the response as it's generated
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
        print()
    except Exception as e:
        print(f"An error occurred during the chatbot response: {e}")