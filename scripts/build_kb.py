# new_sensor_project/scripts/build_kb.py
import logging
import sys
import time
from pathlib import Path
import numpy as np # 添加 NumPy 导入

# --- Setup Project Root Path ---
# Add project root to sys.path to allow importing 'src' and 'config'
# Assumes the script is run from the project root directory (new_sensor_project)
# or that the project is installed / PYTHONPATH is set correctly.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"添加到 sys.path: {project_root}") # Use print before logging is set up

# --- Import Modules ---
try:
    from config import settings
    from src.utils import logging_config
    # DataProcessor is needed here, assuming it will be refactored next
    from src.parameter_standardizer.data_processor import DataProcessor
    from src.parameter_standardizer.embedding_generator import EmbeddingGenerator
    from src.parameter_standardizer.vector_store_manager import VectorStoreManager
except ImportError as e:
    print(f"导入错误: {e}. 请确保从项目根目录 (new_sensor_project) 运行此脚本，")
    print("或者项目结构和 PYTHONPATH 设置正确。")
    print(f"当前 sys.path: {sys.path}")
    sys.exit(1)

# --- Configure Logging ---
try:
    logging_config.setup_logging()
    logger = logging.getLogger(__name__) # Get logger after setup
except Exception as e:
    print(f"致命错误：无法配置日志系统: {e}")
    sys.exit(1)

def main():
    """
    主函数，按顺序执行完整的知识库构建或更新流程。
    1. 数据预处理 (Excel -> prepared_data.csv)
    2. 生成嵌入向量和元数据 (prepared_data.csv -> embeddings.npy, metadata.csv)
    3. 构建/更新向量索引 (embeddings.npy, metadata.csv -> ChromaDB)
    """
    logger.info("=============================================")
    logger.info("=== 开始执行知识库构建/更新流程 ===")
    logger.info("=============================================")
    start_total_time = time.time()

    # --- 步骤 1: 数据预处理 ---
    logger.info("\n>>> 步骤 1: 数据预处理 <<<")
    step1_start = time.time()
    # 定义预处理后的数据输出路径 (虽然 DataProcessor 内部也可能定义，但这里明确指定)
    # 注意：DataProcessor 需要被重构以接受输出路径或从 settings 读取
    processed_data_output_path = settings.OUTPUT_DIR / "prepared_data_for_kb.csv"
    metadata_output_path = settings.OUTPUT_DIR / "metadata_for_kb.csv" # 元数据也由此步骤生成

    try:
        # 假设 DataProcessor 初始化时不需要参数，或从 settings 读取输入路径
        data_processor = DataProcessor(
             excel_path=settings.KB_SOURCE_FILE,
             output_prepared_data_path=processed_data_output_path,
             output_metadata_path=metadata_output_path # 假设 DP 也负责生成最终元数据
        )
        # 假设 run_pipeline 返回处理后的 DataFrame (用于嵌入) 和元数据 DataFrame
        processed_df_for_embedding, df_metadata = data_processor.run_pipeline()
        step1_end = time.time()

        if processed_df_for_embedding is None or df_metadata is None:
            logger.critical("数据预处理失败，流程终止。")
            sys.exit(1)
        if processed_df_for_embedding.empty or df_metadata.empty:
            logger.warning("数据预处理后未生成任何有效数据行。后续步骤可能无效。")
            # 可以选择退出或继续

        logger.info(f"--- 数据预处理完成，耗时: {step1_end - step1_start:.2f} 秒 ---")
        logger.info(f"生成预处理数据文件 (用于嵌入): {processed_data_output_path}")
        logger.info(f"生成元数据文件 (用于索引): {metadata_output_path}")

    except Exception as e:
        logger.exception(f"数据预处理步骤发生意外错误: {e}")
        sys.exit(1)


    # --- 步骤 2: 生成嵌入向量 ---
    logger.info("\n>>> 步骤 2: 生成嵌入向量 <<<")
    step2_start = time.time()
    # 定义嵌入向量输出路径
    embeddings_output_path = settings.KB_VECTOR_STORE_DIR / "embeddings.npy"

    try:
        embedding_generator = EmbeddingGenerator() # 使用 settings 中的模型名
        if not embedding_generator.load_model():
             logger.critical("嵌入模型加载失败，流程终止。")
             sys.exit(1)

        # 从预处理步骤得到的 DataFrame 生成嵌入
        embeddings = embedding_generator.generate_embeddings_from_df(processed_df_for_embedding)
        step2_end = time.time()

        if embeddings is None:
            logger.critical("嵌入向量生成失败，流程终止。")
            sys.exit(1)

        # 保存嵌入向量
        try:
            embeddings_output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(embeddings_output_path), embeddings) # np.save 需要字符串路径
            logger.info(f"--- 嵌入向量生成完成，耗时: {step2_end - step2_start:.2f} 秒 ---")
            logger.info(f"嵌入向量已保存至: {embeddings_output_path}")
        except Exception as e:
            logger.critical(f"保存嵌入向量到 '{embeddings_output_path}' 时出错: {e}", exc_info=True)
            sys.exit(1)

    except Exception as e:
        logger.exception(f"生成嵌入向量步骤发生意外错误: {e}")
        sys.exit(1)


    # --- 步骤 3: 构建/更新向量索引 ---
    logger.info("\n>>> 步骤 3: 构建/更新向量索引 <<<")
    step3_start = time.time()
    index_built = False
    try:
        vector_store = VectorStoreManager() # 使用 settings 中的 DB 路径和集合名
        if not vector_store.connect():
            logger.critical("连接向量数据库失败，流程终止。")
            sys.exit(1)

        # **重要**: 决定是完全重建索引还是更新。
        # 为了简单起见，这里总是清空并重建。如果需要增量更新，逻辑会更复杂。
        logger.warning("即将清空并重建向量数据库集合...")
        if not vector_store.clear_collection():
             logger.critical("清空向量数据库集合失败，流程终止。")
             sys.exit(1)
        logger.info("旧集合已清空，准备构建新索引。")

        # 检查是否有数据需要索引
        if df_metadata.empty or embeddings.size == 0:
            logger.warning("没有有效的元数据或嵌入向量需要索引。")
            index_built = True # 认为构建成功（无事可做）
        else:
            # 使用步骤 1 生成的元数据和步骤 2 生成的嵌入来构建索引
            index_built = vector_store.build_index(embeddings, df_metadata)

        step3_end = time.time()
        if not index_built:
            logger.critical("构建向量索引失败，流程终止。")
            sys.exit(1)

        logger.info(f"--- 构建/更新向量索引完成，耗时: {step3_end - step3_start:.2f} 秒 ---")
        logger.info(f"向量数据库位于: {settings.KB_VECTOR_STORE_DIR}")
        logger.info(f"使用的集合名称: {settings.VECTOR_DB_COLLECTION_NAME}")

    except Exception as e:
        logger.exception(f"构建向量索引步骤发生意外错误: {e}")
        sys.exit(1)

    # --- 流程结束 ---
    end_total_time = time.time()
    logger.info("\n=============================================")
    logger.info(f"=== 知识库构建/更新流程全部完成 ===")
    logger.info(f"=== 总耗时: {end_total_time - start_total_time:.2f} 秒 ===")
    logger.info("=============================================")

if __name__ == "__main__":
    main()
