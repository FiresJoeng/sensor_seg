# scripts/build_knowledge_base.py
import sys
import os
import time

# --- 动态添加 src 目录到 Python 路径 ---
# 这是为了确保脚本能够找到 src 目录下的模块，无论从哪里运行脚本
# 获取当前脚本文件所在的目录 (scripts/)
scripts_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 scripts 目录的上级目录 (即项目根目录)
project_root = os.path.dirname(scripts_dir)
# 构建 src 目录的绝对路径
src_path = os.path.join(project_root, 'src')
# 将 src 目录添加到 Python 解释器搜索模块的路径列表的最前面
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- 导入必要的模块 ---
try:
    # 从 src 目录下的模块中导入类
    from data_processor import DataProcessor
    from embedding_generator import EmbeddingGenerator
    from vector_store_manager import VectorStoreManager
    # 导入配置文件，方便访问路径等设置
    import config
    # (可选) 导入日志记录器或计时工具
    # from utils import logger, time_it
except ImportError as e:
    # 如果导入失败，打印错误信息并退出
    print(f"错误: 无法导入 src 目录下的模块。请确保项目结构正确且 src 目录位于 Python 路径中。")
    print(f"详细错误: {e}")
    # 打印当前的 Python 路径，帮助调试
    print(f"当前 Python 路径 (sys.path): {sys.path}")
    sys.exit(1) # 退出脚本

# @time_it # (可选) 使用装饰器测量整个 main 函数的执行时间
def main():
    """
    主函数，按顺序执行完整的知识库构建流程：
    1. 数据预处理
    2. 生成嵌入向量和元数据
    3. 构建向量数据库索引
    """
    print("=============================================")
    print("=== 开始执行知识库构建流程 ===")
    print("=============================================")
    start_total_time = time.time() # 记录流程开始时间

    # --- 步骤 1: 数据预处理 ---
    print("\n>>> 步骤 1: 数据预处理 <<<")
    step1_start = time.time()
    # 实例化数据处理器
    data_processor = DataProcessor()
    # 执行数据处理流水线 (加载 -> 处理 -> 保存)
    processed_df = data_processor.run_pipeline()
    step1_end = time.time()
    # 检查处理结果
    if processed_df is None or processed_df.empty:
        print("错误: 数据预处理失败或未生成有效数据，流程终止。")
        sys.exit(1) # 预处理失败，则退出脚本
    print(f"--- 数据预处理完成，耗时: {step1_end - step1_start:.2f} 秒 ---")
    print(f"生成预处理数据文件: {config.PREPROCESSED_DATA_PATH}")

    # --- 步骤 2: 生成嵌入向量和元数据 ---
    print("\n>>> 步骤 2: 生成嵌入向量 <<<")
    step2_start = time.time()
    # 实例化嵌入生成器
    embedding_generator = EmbeddingGenerator()
    # 执行嵌入生成流水线 (加载模型 -> 加载数据 -> 生成嵌入 -> 保存结果)
    # 注意: EmbeddingGenerator 内部会重新加载预处理后的数据
    embeddings_generated = embedding_generator.run_pipeline()
    step2_end = time.time()
    # 检查嵌入生成结果
    if not embeddings_generated:
        print("错误: 嵌入向量生成或保存失败，流程终止。")
        sys.exit(1) # 嵌入生成失败，则退出脚本
    print(f"--- 嵌入向量生成完成，耗时: {step2_end - step2_start:.2f} 秒 ---")
    print(f"生成嵌入向量文件: {config.EMBEDDINGS_PATH}")
    print(f"生成元数据文件: {config.METADATA_PATH}")


    # --- 步骤 3: 构建向量索引 ---
    print("\n>>> 步骤 3: 构建向量索引 <<<")
    step3_start = time.time()
    # 实例化向量存储管理器
    vector_store = VectorStoreManager()
    # 首先连接到数据库
    if not vector_store.connect():
        print("错误: 连接向量数据库失败，流程终止。")
        sys.exit(1) # 连接失败，则退出脚本

    # 加载用于构建索引的嵌入向量和元数据
    embeddings, df_metadata = vector_store.load_data_for_indexing()
    # 检查加载结果
    if embeddings is None or df_metadata is None:
         print("错误: 加载嵌入向量或元数据以供索引失败，流程终止。")
         sys.exit(1) # 加载失败，则退出脚本

    # 执行索引构建（将数据添加到 ChromaDB）
    index_built = vector_store.build_index(embeddings, df_metadata)
    step3_end = time.time()
    # 检查索引构建结果
    if not index_built:
        print("错误: 构建向量索引失败，流程终止。")
        sys.exit(1) # 构建失败，则退出脚本
    print(f"--- 构建向量索引完成，耗时: {step3_end - step3_start:.2f} 秒 ---")
    print(f"向量数据库位于: {config.CHROMA_DB_PATH}")


    # --- 流程结束 ---
    end_total_time = time.time() # 记录流程结束时间
    print("\n=============================================")
    print(f"=== 知识库构建流程全部完成 ===")
    print(f"=== 总耗时: {end_total_time - start_total_time:.2f} 秒 ===")
    print("=============================================")

# 当直接运行 build_knowledge_base.py 时，执行 main 函数
if __name__ == "__main__":
    main()
