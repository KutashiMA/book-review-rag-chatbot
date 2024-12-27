import ollama
import csv

# Load the dataset
def load_csv(file_path):
    dataset = []
    with open(file_path, 'r', encoding="utf8") as file:
        reader = csv.reader(file)
        for row in reader:
            dataset.append(' '.join(row))
    print(f'Loaded {len(dataset)} entries from CSV')
    return dataset

dataset = load_csv('datasets/amazon_books_Data.csv')

# Implement the retrieval system
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
# The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
VECTOR_DB = []

def add_chunk_to_database(chunk):
    if not chunk.strip():  # Skip empty chunks
        print("Skipped an empty chunk")
        return
    try:
        # Truncate large chunks to a manageable length
        truncated_chunk = chunk[:1000]  # Adjust the limit as needed
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=truncated_chunk)['embeddings'][0]
        VECTOR_DB.append((chunk, embedding))
    except Exception as e:
        print(f"Failed to process chunk: {chunk[:50]}... Error: {e}")


for i, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)
    print(f'Added chunk {i+1}/{len(dataset)} to the database')

def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Chatbot loop
while True:
    input_query = input('Ask me a question (or type "exit" to quit): ')
    if input_query.lower() == "exit":
        print("Exiting the chatbot. Goodbye!")
        break

    retrieved_knowledge = retrieve(input_query)

    instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{''.join([f' - {chunk}\n' for chunk, similarity in retrieved_knowledge])}'''

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
