import chromadb
from sentence_transformers import SentenceTransformer
import time
import argparse # To accept query from command line

# --- Configuration ---

# 1. Model Configuration (Must match the model used for embedding)
EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2' 
# EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' 

# 2. Vector Database Configuration (Must point to the existing DB)
CHROMA_DB_PATH = "./chroma_db_semantic_final" # Directory where the database is stored
COLLECTION_NAME = "semantic_parameters"      # Name of the collection used

# --- Function Definitions ---

def load_embedding_model(model_name):
    """Loads the Sentence Transformer model."""
    print(f"--- Loading Embedding Model ---")
    print(f"Model: {model_name}")
    start_time = time.time()
    try:
        model = SentenceTransformer(model_name)
        end_time = time.time()
        print(f"Model loaded in {end_time - start_time:.2f} seconds.")
        return model
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        return None

def connect_to_chroma_db(db_path, collection_name):
    """Connects to the existing ChromaDB collection."""
    print(f"\n--- Connecting to Vector Database ---")
    print(f"Database path: {db_path}")
    try:
        chroma_client = chromadb.PersistentClient(path=db_path)
        # Get the existing collection (don't create)
        collection = chroma_client.get_collection(name=collection_name)
        print(f"Successfully connected to collection '{collection_name}'.")
        print(f"Collection contains {collection.count()} items.")
        return collection
    except Exception as e:
        # Catch specific errors e.g., if collection doesn't exist
        print(f"Error connecting to ChromaDB collection '{collection_name}': {e}")
        print("Please ensure the index has been built using '3_build_index.py'.")
        return None

def query_knowledge_base(collection, model, query_text, n_results=3):
    """Embeds a query and searches the ChromaDB collection."""
    print(f"\n--- Querying Knowledge Base ---")
    if collection is None or model is None:
        print("Error: Collection or embedding model not available.")
        return None

    print(f"User Query: '{query_text}'")
    
    # Embed the query
    start_time = time.time()
    query_embedding = model.encode([query_text])
    embed_time = time.time()
    print(f"Query embedded in {embed_time - start_time:.4f} seconds.")
    
    # Query ChromaDB
    try:
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=['metadatas', 'distances'] 
        )
        query_time = time.time()
        print(f"Search completed in {query_time - embed_time:.4f} seconds.")
        return results
    except Exception as e:
        print(f"Error querying ChromaDB collection: {e}")
        return None

def display_results(query_text, results):
     """Formats and prints the query results."""
     print(f"\n--- Search Results for '{query_text}' ---")
     if not results:
         print("  Query execution failed or returned no results.")
         return

     ids = results.get('ids', [[]])[0]
     distances = results.get('distances', [[]])[0]
     metadatas = results.get('metadatas', [[]])[0]
     
     if not ids:
          print("  No similar items found in the knowledge base.")
     else:
          for i in range(len(ids)):
              print(f"\n  Rank {i+1}:")
              print(f"    Similarity Score (Distance): {distances[i]:.4f}") # Lower distance = more similar
              # Print relevant metadata fields nicely
              print(f"    Matched Actual Value : '{metadatas[i].get('actual_value', 'N/A')}'")
              print(f"    Standard Value       : '{metadatas[i].get('standard_value', 'N/A')}'")
              print(f"    Standard Code        : '{metadatas[i].get('standard_code', 'N/A')}'")
              print(f"    Parameter Type       : '{metadatas[i].get('parameter_type', 'N/A')}'")
              print(f"    Component Part       : '{metadatas[i].get('component_part', 'N/A')}'")
              # print(f"    Raw Metadata: {metadatas[i]}") # Uncomment for full metadata debugging

# --- Main Execution ---
if __name__ == "__main__":
    # Set up argument parser to accept query from command line
    parser = argparse.ArgumentParser(description="Query the semantic knowledge base.")
    parser.add_argument("query", type=str, help="The actual parameter value to search for.")
    parser.add_argument("-n", "--num_results", type=int, default=3, help="Number of results to return.")
    
    args = parser.parse_args()
    
    query_text = args.query
    num_results = args.num_results

    # 1. Load Model
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    
    if embedding_model:
        # 2. Connect to DB
        chroma_collection = connect_to_chroma_db(CHROMA_DB_PATH, COLLECTION_NAME)
        
        if chroma_collection:
            # 3. Perform Query
            search_results = query_knowledge_base(chroma_collection, embedding_model, query_text, n_results=num_results)
            
            # 4. Display Results
            display_results(query_text, search_results)

    print("\n--- Query Script Finished ---")