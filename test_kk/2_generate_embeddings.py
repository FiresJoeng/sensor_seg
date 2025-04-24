import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import os

# --- Configuration ---

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
    """连接到现有的 ChromaDB 集合。"""
    print(f"\n--- 正在连接向量数据库 ---")
    print(f"数据库路径: {db_path}")
    try:
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_collection(name=collection_name)
        print(f"成功连接到集合 '{collection_name}'。")
        print(f"集合包含 {collection.count()} 条数据。")
        return collection
    except Exception as e:
        print(f"连接 ChromaDB 集合 '{collection_name}' 时出错: {e}")
        print("请确保索引已使用 '3_build_index.py' 构建，并检查数据库路径和集合名称是否正确。")
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
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="查询语义知识库。")
    parser.add_argument("query", type=str, nargs="?", default="默认查询参数", help="要搜索的实际参数值。")
    parser.add_argument("-n", "--num_results", type=int, default=3, help="返回的结果数量。")
    
    args = parser.parse_args()
    
    query_text = args.query
    num_results = args.num_results

    # 1. 加载模型
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    
    if embedding_model:
        # 2. 连接数据库
        chroma_collection = connect_to_chroma_db(CHROMA_DB_PATH, COLLECTION_NAME)
        
        if chroma_collection:
            # 3. 执行查询
            search_results = query_knowledge_base(chroma_collection, embedding_model, query_text, n_results=num_results)
            
            # 4. 显示结果
            display_results(query_text, search_results)

    print("\n--- 查询脚本执行完成 ---")

# 2. Model Configuration (Must match the model used for querying)
EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2' 
# EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' 

# 3. Output Files
EMBEDDINGS_OUTPUT_PATH = 'embeddings.npy'           # To save the numpy array of embeddings
METADATA_OUTPUT_PATH = 'metadata_for_index.csv' # To save the corresponding metadata

# 4. Column Names (from the preprocessed CSV)
COL_ACTUAL_VARIATION = 'Actual_Value_Variation' 
COL_STANDARD_PARAM = '标准参数' 
# Also include columns needed for metadata in the next step
COL_STANDARD_VALUE = '规格书代码的说明（多个值用|隔开）' 
COL_STANDARD_CODE = '对应代码' 
COL_COMPONENT_PART = '元器件部位' 

# 定义预处理数据文件路径
PREPROCESSED_CSV_PATH = './prepared_semantic_data.csv'  # 请根据实际路径修改

import os

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 使用 os.path.join 构建绝对路径
PREPROCESSED_CSV_PATH = os.path.join(SCRIPT_DIR, 'prepared_semantic_data.csv')
CHROMA_DB_PATH = os.path.join(SCRIPT_DIR, 'chroma_db_semantic_final')
EMBEDDINGS_OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'embeddings.npy')
METADATA_OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'metadata_for_index.csv')

# --- Main Logic ---

def load_data(csv_path):
    print(f"--- 正在加载预处理数据 ---")
    if not os.path.exists(csv_path):
        print(f"错误: 未找到预处理数据文件 '{csv_path}'")
        print("请先运行预处理脚本生成该文件，或检查路径是否正确。")
        return None
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna('') # Read all as string, fill NaN
        print(f"已加载数据 '{csv_path}'，形状: {df.shape}")
        # Ensure required columns exist
        required_cols = [COL_ACTUAL_VARIATION, COL_STANDARD_PARAM, COL_STANDARD_VALUE, 
                         COL_STANDARD_CODE, COL_COMPONENT_PART]
        for col in required_cols:
            if col not in df.columns:
                 print(f"错误: 缺少所需列 '{col}'，请检查数据文件。")
                 return None
        return df
    except Exception as e:
        print(f"加载数据 '{csv_path}' 时出错: {e}")
        return None

def load_embedding_model(model_name):
    """Loads the Sentence Transformer model."""
    print(f"\n--- Loading Embedding Model ---")
    print(f"Loading model: {model_name}")
    start_time = time.time()
    try:
        model = SentenceTransformer(model_name)
        end_time = time.time()
        print(f"Model loaded successfully in {end_time - start_time:.2f} seconds.")
        return model
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        return None

def generate_and_save_embeddings(model, df, text_col, context_col, embeddings_path, metadata_path):
    """Generates embeddings and saves them along with metadata."""
    print(f"\n--- Generating and Saving Embeddings ---")
    if model is None or df is None:
        print("Error: Model or DataFrame is missing.")
        return False
        
    # Prepare texts with context
    texts_to_embed = [
        f"{row[context_col]}: {row[text_col]}"
        for index, row in df.iterrows()
    ]
    print(f"Prepared {len(texts_to_embed)} texts for embedding.")

    print("Generating embeddings (this might take time)...")
    start_time = time.time()
    try:
        embeddings = model.encode(texts_to_embed, show_progress_bar=True, batch_size=32)
        end_time = time.time()
        print(f"Embeddings generated. Shape: {embeddings.shape}. Time taken: {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"An error occurred during embedding generation: {e}")
        return False

    # Save embeddings
    try:
        np.save(embeddings_path, embeddings)
        print(f"Embeddings successfully saved to: {embeddings_path}")
    except Exception as e:
        print(f"Error saving embeddings to '{embeddings_path}': {e}")
        return False
        
    # Save corresponding metadata (ensure index alignment)
    # Select only the necessary columns for the index metadata
    metadata_df = df[[COL_STANDARD_VALUE, COL_STANDARD_CODE, COL_ACTUAL_VARIATION, 
                      COL_STANDARD_PARAM, COL_COMPONENT_PART]].copy()
    try:
        metadata_df.to_csv(metadata_path, index=False, encoding='utf-8-sig')
        print(f"Metadata successfully saved to: {metadata_path}")
    except Exception as e:
        print(f"Error saving metadata to '{metadata_path}': {e}")
        return False
        
    return True

# --- Execution ---
if __name__ == "__main__":
    # 1. Load Data
    dataframe = load_data(PREPROCESSED_CSV_PATH)
    
    if dataframe is not None:
        # 2. Load Model
        embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
        
        if embedding_model is not None:
            # 3. Generate and Save Embeddings & Metadata
            success = generate_and_save_embeddings(embedding_model, dataframe, 
                                                 text_col=COL_ACTUAL_VARIATION, 
                                                 context_col=COL_STANDARD_PARAM,
                                                 embeddings_path=EMBEDDINGS_OUTPUT_PATH,
                                                 metadata_path=METADATA_OUTPUT_PATH)
            if success:
                 print("\n--- Embedding Generation Script Finished Successfully ---")
            else:
                 print("\n--- Embedding Generation Script Finished with Errors ---")