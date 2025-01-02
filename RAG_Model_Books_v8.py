# Import required libraries
import ollama  # For interacting with language models
import csv     # For handling CSV file operations
import nltk    # Natural Language Toolkit for text processing
from nltk.corpus import stopwords  # For removing common words that don't add meaning
import itertools  # For creating infinite iterators (used in spinner animation)
import time       # For timing operations and delays
from threading import Thread  # For running spinner animation concurrently

# Download required NLTK data silently
nltk.download("punkt", quiet=True)     # For sentence tokenization
nltk.download("stopwords", quiet=True)  # For stop words

# Load English stop words into a set for efficient lookup
STOP_WORDS = set(stopwords.words("english"))

def load_csv(file_path):
   """
   Load and validate dataset from a CSV file.
   Args:
       file_path (str): Path to the CSV file
   Returns:
       list: List of dictionaries containing valid entries
   """
   dataset = []
   with open(file_path, 'r', encoding="utf8") as file:
       reader = csv.DictReader(file)
       for row in reader:
           # Only include rows that have both product title and review body
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
   Preprocess a single dataset chunk.
   Args:
       chunk (dict): Raw data chunk containing review information
   Returns:
       dict: Processed chunk with trimmed review body
   """
   return {
       "product_id": chunk["product_id"],
       "product_title": chunk["product_title"],
       "review_body": chunk["review_body"].strip()[:1000],  # Limit review length to 1000 chars
       "star_rating": chunk["star_rating"]
   }

def tokenize_and_remove_stop_words(text):
   """
   Process text by tokenizing and removing stop words.
   Args:
       text (str): Input text to process
   Returns:
       list: Clean tokens without stop words
   """
   # Convert to lowercase and tokenize
   tokens = nltk.word_tokenize(text.lower())
   # Filter out non-alphanumeric tokens and stop words
   return [word for word in tokens if word.isalnum() and word not in STOP_WORDS]

def show_dynamic_loading_with_progress(message, current, total):
   """
   Display a loading spinner with progress counter.
   Args:
       message (str): Message to display
       current (int): Current progress number
       total (int): Total items to process
   """
   spinner = next(itertools.cycle(['|', '/', '-', '\\']))
   print(f"\r{message} {spinner} ({current}/{total})", end="", flush=True)

def show_dynamic_loading_with_message(message):
   """
   Display a continuous loading spinner with message.
   Args:
       message (str): Message to display alongside spinner
   Returns:
       tuple: (spinner_task function, stop_spinner function)
   """
   spinner = itertools.cycle(['|', '/', '-', '\\'])
   is_running = True

   def spinner_task():
       """Inner function to run the spinner animation"""
       while is_running:
           print(f"\r{message} {next(spinner)}", end="", flush=True)
           time.sleep(0.1)
       # Clear the entire line when done
       print("\r" + " " * (len(message) + 2) + "\r", end="")

   def stop_spinner():
       """Inner function to stop the spinner"""
       nonlocal is_running
       is_running = False

   return spinner_task, stop_spinner

def check_factuality_individually(context, claims):
   """
   Validate individual claims against provided context.
   Args:
       context (str): Source text to validate against
       claims (list): List of claims to verify
   Returns:
       list: Claims that are supported by the context
   """
   supported_claims = []
   
   # Setup and start spinner animation
   spinner_task, stop_spinner = show_dynamic_loading_with_message("Processing factual checks...")
   spinner_thread = Thread(target=spinner_task)
   spinner_thread.start()

   try:
       for claim in claims:
           # Create prompt for factuality check
           prompt = (
               f"Document: {context}\n"
               f"Claim: {claim}\n"
               f"Does the claim accurately reflect the information in the document? Respond with 'yes' or 'no'."
           )
           # Generate response and check if claim is supported
           response = ollama.generate(
               model="bespoke-minicheck",
               prompt=prompt,
               options={"num_predict": 1, "temperature": 0.0}
           )
           if response["response"].strip().lower().startswith("yes"):
               supported_claims.append(claim)
   finally:
       stop_spinner()
       spinner_thread.join()

   return supported_claims

def keyword_search(query, dataset):
   """
   Search dataset for keyword matches.
   Args:
       query (str): Search query
       dataset (list): Collection of documents to search
   Returns:
       list: Matching documents
   """
   query_tokens = tokenize_and_remove_stop_words(query)
   return [chunk for chunk in dataset if any(word in chunk['review_body'].lower() for word in query_tokens)]

def is_keyword_query(query):
   """
   Check if query is keyword-based.
   Args:
       query (str): User query
   Returns:
       bool: True if query contains keyword indicators
   """
   keyword_indicators = ["mention", "contain", "include", "refer to", "talk about"]
   return any(indicator in query.lower() for indicator in keyword_indicators)

def is_book_specific_query(query):
   """
   Check if query targets specific book title.
   Args:
       query (str): User query
   Returns:
       bool: True if query contains book title from database
   """
   return any(chunk['product_title'].lower() in query.lower() for chunk, _ in VECTOR_DB)

def retry_logic(prompt_function, max_retries=2):
   """
   Implement retry mechanism for prompt functions.
   Args:
       prompt_function: Function to retry
       max_retries (int): Maximum number of retry attempts
   Returns:
       Result from successful attempt or empty list
   """
   for attempt in range(max_retries):
       if attempt > 0:
           print("\nReevaluating response for better accuracy...🔄\n")
       result = prompt_function()
       if result:
           return result
   return []

def handle_response(context, input_query):
   """
   Generate and validate response based on context.
   Args:
       context (str): Source context for response
       input_query (str): User query
   Returns:
       list: Validated claims from response
   """
   start_time = time.time()

   # Generate response using language model
   response = ollama.chat(
       model=LANGUAGE_MODEL,
       messages=[
           {'role': 'system', 'content': f'Use only these reviews to answer: {context}'},
           {'role': 'user', 'content': input_query},
       ]
   )

   # Process and validate response
   answer = response['message']['content']
   claims = nltk.sent_tokenize(answer)
   supported_claims = check_factuality_individually(context, claims)

   # Calculate and display processing time
   end_time = time.time()
   elapsed_time = end_time - start_time
   minutes, seconds = divmod(elapsed_time, 60)
   print(f"\nResponse generated in {int(minutes)} minutes and {seconds:.2f} seconds.\n")

   return supported_claims

def add_chunk_to_database(chunk, index, total):
   """
   Add chunk to vector database with embedding.
   Args:
       chunk (dict): Data chunk to add
       index (int): Current processing index
       total (int): Total chunks to process
   """
   try:
       embedding_input = f"""
       Title: {chunk['product_title']}
       Review Content: {chunk['review_body']}
       Rating: {chunk['star_rating']}
       """
       embedding = ollama.embed(model=EMBEDDING_MODEL, input=embedding_input)['embeddings'][0]
       VECTOR_DB.append((chunk, embedding))
   except Exception as e:
       print(f"\nFailed to process chunk: {chunk['product_title'][:50]}... Error: {e}")

def cosine_similarity(a, b):
   """
   Calculate cosine similarity between vectors.
   Args:
       a (list): First vector
       b (list): Second vector
   Returns:
       float: Similarity score
   """
   dot_product = sum([x * y for x, y in zip(a, b)])
   norm_a = sum([x ** 2 for x in a]) ** 0.5
   norm_b = sum([x ** 2 for x in b]) ** 0.5
   return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=5):
   """
   Retrieve most relevant chunks for query.
   Args:
       query (str): Search query
       top_n (int): Number of results to return
   Returns:
       list: Top matching chunks with similarity scores
   """
   spinner_task, stop_spinner = show_dynamic_loading_with_message("Retrieving relevant reviews...")
   spinner_thread = Thread(target=spinner_task)
   spinner_thread.start()

   try:
       query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
       similarities = []
       for chunk, embedding in VECTOR_DB:
           similarity = cosine_similarity(query_embedding, embedding)
           similarities.append((chunk, similarity))
       similarities.sort(key=lambda x: x[1], reverse=True)
       return similarities[:top_n]
   finally:
       stop_spinner()
       spinner_thread.join()

# Initialize main program components
dataset = [preprocess_chunk(chunk) for chunk in load_csv('datasets/amazon_books_Data.csv')]

# Define model constants
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Initialize vector database
VECTOR_DB = []

# Populate vector database
print("Starting database indexing...")
try:
   for i, chunk in enumerate(dataset, start=1):
       show_dynamic_loading_with_progress("Indexing chunks into the database...", i, len(dataset))
       add_chunk_to_database(chunk, i, len(dataset))
   print("\nFinished indexing all chunks.\n")
except Exception as e:
   print(f"\nAn error occurred during indexing: {e}")

# Main interaction loop
while True:
   input_query = input('\nAsk me a question (or type "exit" to quit): ')
   
   if input_query.lower() == "exit":
       print("\nExiting the chatbot. Goodbye!\n")
       break

   # Handle different query types
   if is_keyword_query(input_query):
       context_chunks = keyword_search(input_query, dataset)
   else:
       retrieved_knowledge = retrieve(input_query)
       if not retrieved_knowledge:
           print("\nI couldn't find any relevant context in the dataset. Please try a different query.\n")
           continue

       if is_book_specific_query(input_query):
           context_chunks = [chunk for chunk, similarity in retrieved_knowledge 
                            if chunk['product_title'].lower() in input_query.lower()]
       else:
           context_chunks = [chunk for chunk, similarity in retrieved_knowledge]

   if not context_chunks:
       print("\nNo relevant information found.\n")
       continue

   # Build context from relevant chunks
   context = '\n'.join([
       f"Review for {chunk['product_title']}:\n{chunk['review_body']}\nRating: {chunk['star_rating']}" 
       for chunk in context_chunks
   ])

   # Generate and validate response
   supported_claims = retry_logic(lambda: handle_response(context, input_query))

   # Present results
   if supported_claims:
       print("Here's what I found based on the reviews:\n")
       print("\n".join(supported_claims))
       print("\n")
   else:
       print("\nSorry, I couldn't find a suitable response...😥\n")