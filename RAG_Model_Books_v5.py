import ollama
import csv

# Load the dataset with specific columns
def load_csv(file_path):
    dataset = []
    with open(file_path, 'r', encoding="utf8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Combine title, review, and rating into a single chunk with unique identifiers
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
    """Preprocess chunks to remove extra whitespace or problematic patterns."""
    return {
        "product_id": chunk["product_id"],
        "product_title": chunk["product_title"],
        "review_body": chunk["review_body"].strip()[:1000],  # Truncate review body to 1000 characters
        "star_rating": chunk["star_rating"]
    }

dataset = [preprocess_chunk(chunk) for chunk in load_csv('datasets/amazon_books_Data.csv')]

# Implement the retrieval system
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
VECTOR_DB = []

def add_chunk_to_database(chunk):
    try:
        embedding_input = f"Title: {chunk['product_title']}\nReview: {chunk['review_body']}"
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=embedding_input)['embeddings'][0]
        VECTOR_DB.append((chunk, embedding))
    except Exception as e:
        print(f"Failed to process chunk: {chunk['product_title'][:50]}... Error: {e}")

for i, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)
    print(f'Added chunk {i+1}/{len(dataset)} to the database')

def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=5):
    try:
        query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
        similarities = []
        for chunk, embedding in VECTOR_DB:
            similarity = cosine_similarity(query_embedding, embedding)
            similarities.append((chunk, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Keep top_n results
        return similarities[:top_n]

    except Exception as e:
        print(f"Failed to retrieve context for query: {query}. Error: {e}")
        return []

# Chatbot loop
while True:
    input_query = input('Ask me a question (or type "exit" to quit): ')
    if input_query.lower() == "exit":
        print("Exiting the chatbot. Goodbye!")
        break

    retrieved_knowledge = retrieve(input_query)

    if not retrieved_knowledge:
        print("I couldn't find any relevant context in the dataset. Please try a different query.")
        continue

    # Filter chunks by checking if the product_title matches the query
    matching_chunks = [chunk for chunk, similarity in retrieved_knowledge if chunk['product_title'].lower() in input_query.lower()]

    if not matching_chunks:
        print("The provided context does not contain an answer to the question.")
        continue

    # Add validation instructions for the LLM
    instruction_prompt = f'''You are an expert assistant providing answers about books and reviews.
Use only the following pieces of context to answer the question accurately and concisely:
{''.join([f" - Title: {chunk['product_title']}\n  Review: {chunk['review_body']}\n  Rating: {chunk['star_rating']}\n" for chunk in matching_chunks])}
If none of the context is relevant to the query, respond with: "The provided context does not contain an answer to the question."'''

    try:
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': input_query},
            ],
            stream=True,
        )

        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
        print()
    except Exception as e:
        print(f"An error occurred during the chatbot response: {e}")
