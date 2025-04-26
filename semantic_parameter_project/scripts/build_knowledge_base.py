# scripts/build_knowledge_base.py
import sys
import os
import time

# --- 动态添加 src 目录到 Python 路径 ---
scripts_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(scripts_dir)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- 导入必要的模块 ---
try:
    from data_processor import DataProcessor
    from embedding_generator import EmbeddingGenerator
    from vector_store_manager import VectorStoreManager
    import config
    # from utils import logger, time_it # 可选
except ImportError as e:
    print(f"错误: 无法导入 src 目录下的模块。")
    print(f"详细错误: {e}")
    print(f"当前 Python 路径 (sys.path): {sys.path}")
    sys.exit(1)

# @time_it # 可选
def main():
    """
    主函数，按顺序执行完整的知识库构建流程。
    """
    print("=============================================")
    print("=== 开始执行知识库构建流程 ===")
    print("=============================================")
    start_total_time = time.time()

    # --- 步骤 1: 数据预处理 ---
    print("\n>>> 步骤 1: 数据预处理 <<<")
    step1_start = time.time()
    data_processor = DataProcessor()
    processed_df = data_processor.run_pipeline()
    step1_end = time.time()
    if processed_df is None: # 允许空 DataFrame，但不允许 None
        print("错误: 数据预处理失败，流程终止。")
        sys.exit(1)
    if processed_df.empty:
        print("警告: 数据预处理后未生成任何有效数据行。后续步骤可能无法正常工作。")
        # 可以选择在这里退出，或者继续尝试处理空数据
        # sys.exit(1)
    print(f"--- 数据预处理完成，耗时: {step1_end - step1_start:.2f} 秒 ---")
    print(f"生成预处理数据文件: {config.PREPROCESSED_DATA_PATH}")

    # --- 步骤 2: 生成嵌入向量和元数据 ---
    print("\n>>> 步骤 2: 生成嵌入向量 <<<")
    step2_start = time.time()
    embedding_generator = EmbeddingGenerator()
    embeddings_generated = embedding_generator.run_pipeline()
    step2_end = time.time()
    if not embeddings_generated:
        print("错误: 嵌入向量生成或保存失败，流程终止。")
        sys.exit(1)
    print(f"--- 嵌入向量生成完成，耗时: {step2_end - step2_start:.2f} 秒 ---")
    print(f"生成嵌入向量文件: {config.EMBEDDINGS_PATH}")
    print(f"生成元数据文件: {config.METADATA_PATH}")


    # --- 步骤 3: 构建向量索引 ---
    print("\n>>> 步骤 3: 构建向量索引 <<<")
    step3_start = time.time()
    vector_store = VectorStoreManager()
    if not vector_store.connect():
        print("错误: 连接向量数据库失败，流程终止。")
        sys.exit(1)

    embeddings, df_metadata = vector_store.load_data_for_indexing()
    if embeddings is None or df_metadata is None:
         print("错误: 加载嵌入向量或元数据以供索引失败，流程终止。")
         sys.exit(1)

    # 如果 df_metadata 为空，则不需要构建索引
    if df_metadata.empty:
        print("警告: 元数据为空，无需构建索引。请检查数据预处理步骤。")
        index_built = True # 认为“构建”成功，因为无事可做
    else:
        index_built = vector_store.build_index(embeddings, df_metadata)

    step3_end = time.time()
    if not index_built:
        print("错误: 构建向量索引失败，流程终止。")
        sys.exit(1)
    print(f"--- 构建向量索引完成，耗时: {step3_end - step3_start:.2f} 秒 ---")
    print(f"向量数据库位于: {config.CHROMA_DB_PATH}")


    # --- 流程结束 ---
    end_total_time = time.time()
    print("\n=============================================")
    print(f"=== 知识库构建流程全部完成 ===")
    print(f"=== 总耗时: {end_total_time - start_total_time:.2f} 秒 ===")
    print("=============================================")

if __name__ == "__main__":
    main()
