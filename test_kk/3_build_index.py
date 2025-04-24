import pandas as pd
import numpy as np
import chromadb
import time
import os

# --- Configuration ---

# 1. Input Files (from embedding script)
EMBEDDINGS_PATH = 'embeddings.npy'
METADATA_PATH = 'metadata_for_index.csv'

# 2. Vector Database Configuration
CHROMA_DB_PATH = "./chroma_db_semantic_final" # Directory to store the database
COLLECTION_NAME = "semantic_parameters"      # Name for the collection

# 3. Metadata Column Names (Ensure these match the columns in METADATA_PATH)
# These names will be used as keys in the ChromaDB metadata dictionary
META_COL_STANDARD_VALUE = '规格书代码的说明（多个值用|隔开）' 
META_COL_STANDARD_CODE = '对应代码' 
META_COL_ACTUAL_VARIATION = 'Actual_Value_Variation' 
META_COL_STANDARD_PARAM = '标准参数' 
META_COL_COMPONENT_PART = '元器件部位' 

# --- Main Logic ---

def load_embeddings_and_metadata(embeddings_path, metadata_path):
    """Loads embeddings and metadata from files."""
    print("--- Loading Embeddings and Metadata ---")
    if not os.path.exists(embeddings_path):
        print(f"Error: Embeddings file not found at '{embeddings_path}'")
        return None, None
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at '{metadata_path}'")
        return None, None
        
    try:
        print(f"Loading embeddings from {embeddings_path}...")
        embeddings = np.load(embeddings_path)
        print(f"Embeddings loaded. Shape: {embeddings.shape}")
        
        print(f"Loading metadata from {metadata_path}...")
        df_metadata = pd.read_csv(metadata_path, dtype=str).fillna('') # Load as string, handle NaN
        print(f"Metadata loaded. Shape: {df_metadata.shape}")
        
        if len(embeddings) != len(df_metadata):
             print("Error: Mismatch between number of embeddings and metadata rows.")
             print(f"Embeddings count: {len(embeddings)}, Metadata rows: {len(df_metadata)}")
             return None, None
             
        return embeddings, df_metadata
        
    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None

def setup_chroma_db(db_path, collection_name):
    """Initializes ChromaDB client and gets/creates the collection."""
    print(f"\n--- Setting up ChromaDB ---")
    print(f"Using database path: {db_path}")
    try:
        chroma_client = chromadb.PersistentClient(path=db_path)
        print(f"Getting or creating collection: {collection_name}")
        collection = chroma_client.get_or_create_collection(name=collection_name)
        print(f"ChromaDB setup complete. Collection '{collection_name}' ready.")
        return collection, chroma_client # Return client too if needed for delete
    except Exception as e:
        print(f"Error setting up ChromaDB: {e}")
        return None, None

def populate_chroma_index(collection, embeddings, df_metadata):
    """Populates the ChromaDB collection."""
    print(f"\n--- Populating ChromaDB Index ---")
    if collection is None or embeddings is None or df_metadata is None:
         print("Error: Missing collection, embeddings, or metadata for population.")
         return False
         
    # Check if populated
    current_count = collection.count()
    print(f"Collection currently contains {current_count} items.")
    target_count = len(df_metadata)
    
    if current_count >= target_count:
        print(f"Collection appears to be already populated with {current_count} items (expected {target_count}). Skipping.")
        return True 
    elif current_count > 0:
         print(f"Warning: Collection has {current_count} items, but expected {target_count}. Re-populating.")
         # Optional: Clear collection before re-populating if needed
         # print("Note: Consider clearing the collection manually or adding deletion logic if a full refresh is desired.")
         # Example: client.delete_collection(collection_name) # Needs client object

    print("Preparing data for ChromaDB...")
    embeddings_list = embeddings.tolist()
    
    metadatas = []
    # Use the META_COL_ constants defined in config
    for index, row in df_metadata.iterrows():
        metadata_item = {
            'standard_value': row[META_COL_STANDARD_VALUE],
            'standard_code': row[META_COL_STANDARD_CODE],
            'actual_value': row[META_COL_ACTUAL_VARIATION],
            'parameter_type': row[META_COL_STANDARD_PARAM],
            'component_part': row[META_COL_COMPONENT_PART]
        }
        metadatas.append(metadata_item)
        
    ids = [str(i) for i in df_metadata.index] # Use DataFrame index as unique string IDs

    print(f"Adding {len(ids)} items to the collection...")
    start_time = time.time()
    try:
        # Add in batches for potentially large datasets (ChromaDB handles this)
        # Upsert might be safer if re-running with potential ID overlap is possible
        collection.add( # or collection.upsert(...)
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids
        )
        end_time = time.time()
        print(f"Data added successfully in {end_time - start_time:.2f} seconds.")
        final_count = collection.count()
        print(f"Collection count after adding: {final_count}")
        # Basic check if final count matches expected count
        if final_count != target_count:
             print(f"Warning: Final count ({final_count}) does not match expected count ({target_count}).")
        return True
    except Exception as e:
        print(f"Error adding data to Chroma collection: {e}")
        return False

# --- Execution ---
if __name__ == "__main__":
    # 1. Load Embeddings and Metadata
    embeddings_array, df_meta = load_embeddings_and_metadata(EMBEDDINGS_PATH, METADATA_PATH)

    if embeddings_array is not None and df_meta is not None:
        # 2. Setup ChromaDB
        chroma_collection, chroma_client = setup_chroma_db(CHROMA_DB_PATH, COLLECTION_NAME)

        if chroma_collection is not None:
            # 3. Populate Index
            success = populate_chroma_index(chroma_collection, embeddings_array, df_meta)
            
            if success:
                 print("\n--- Index Building Script Finished Successfully ---")
            else:
                 print("\n--- Index Building Script Finished with Errors ---")