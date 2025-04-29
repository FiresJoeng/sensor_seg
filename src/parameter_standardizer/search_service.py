# new_sensor_project/src/parameter_standardizer/search_service.py
import time
import logging
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import sys
from pathlib import Path

# --- Module Imports using Absolute Paths ---
# Ensure project root is in sys.path (similar logic to main_pipeline.py)
# This makes absolute imports from the project root reliable.
project_root = Path(__file__).resolve().parent.parent.parent # Should be new_sensor_project
if str(project_root) not in sys.path:
    # This might not be strictly necessary if main_pipeline already added it,
    # but makes the module potentially runnable independently or in tests.
    sys.path.insert(0, str(project_root))
    # Add a print or basic log here if needed for debugging standalone runs
    # print(f"DEBUG: Added {project_root} to sys.path in search_service.py")

try:
    from config import settings
    from src.parameter_standardizer.embedding_generator import EmbeddingGenerator
    from src.parameter_standardizer.vector_store_manager import VectorStoreManager
except ImportError as e:
    # Use basic logging/print if full logging isn't set up yet
    print(f"ERROR in search_service.py: Failed to import modules - {e}. Check project structure and PYTHONPATH.", file=sys.stderr)
    # Re-raise or exit if these imports are critical for the module to load
    raise

# 获取日志记录器实例
# Ensure logging is configured before use, typically by the entry point (main_pipeline.py)
logger = logging.getLogger(__name__)

class SearchService:
    """
    封装参数知识库的核心搜索逻辑。
    接收实际参数名和值，查询向量数据库，返回最匹配的标准参数名、值和代码。
    """
    def __init__(self):
        """初始化搜索服务，加载所需组件。"""
        logger.debug("--- 初始化参数标准化搜索服务 ---") # DEBUG: Changed from INFO
        self.embedding_generator = EmbeddingGenerator() # 用于加载和访问模型
        self.vector_store = VectorStoreManager()
        self.model_loaded = False
        self.db_connected = False
        self._initialize_components()

    def _initialize_components(self):
        """加载嵌入模型并连接到向量数据库。"""
        logger.debug("正在加载嵌入模型...") # DEBUG: Changed from INFO
        # 注意：EmbeddingGenerator 内部应使用 logging
        self.model_loaded = self.embedding_generator.load_model()
        if not self.model_loaded:
            logger.error("嵌入模型加载失败。搜索服务将不可用。")
            return

        logger.debug("正在连接向量数据库...") # DEBUG: Changed from INFO
        # 注意：VectorStoreManager 内部应使用 logging
        self.db_connected = self.vector_store.connect()
        if not self.db_connected:
            logger.error("向量数据库连接失败。搜索服务将不可用。")

    def is_ready(self) -> bool:
        """检查搜索服务是否准备就绪。"""
        ready = self.model_loaded and self.db_connected
        if not ready:
            logger.warning(f"搜索服务未就绪 (模型加载: {self.model_loaded}, DB连接: {self.db_connected})")
        return ready

    def _encode_query(self, query_text: str) -> Optional[np.ndarray]:
        """对组合后的查询文本进行编码。"""
        if not self.embedding_generator.model:
            logger.error("无法编码查询：嵌入模型不可用。")
            return None
        try:
            # EmbeddingGenerator.encode 应返回 np.ndarray
            embedding = self.embedding_generator.encode([query_text])
            if embedding is None or embedding.size == 0:
                 logger.error(f"查询文本 '{query_text}' 编码返回空结果。")
                 return None
            # 确保返回的是单个向量的 ndarray
            return embedding[0] if isinstance(embedding, list) or (isinstance(embedding, np.ndarray) and embedding.ndim > 1) else embedding
        except Exception as e:
            logger.error(f"查询文本 '{query_text}' 编码时出错: {e}", exc_info=True)
            return None

    def search(self, actual_param_name: str, actual_param_value: str,
               n_results: int = 1) -> Optional[Tuple[str, str, str]]:
        """
        执行参数搜索，找到最匹配的标准参数。

        根据输入的实际参数名和值，在向量知识库中查找最相似的条目，
        并返回该条目的标准参数名、标准参数值和标准代码。

        Args:
            actual_param_name (str): 从文档提取的实际参数名称。
            actual_param_value (str): 从文档提取的实际参数值。
            n_results (int): 希望向量数据库返回的初始结果数量（通常 > 1 以便选择最佳）。
                             注意：此方法最终只返回最佳的 1 个结果。

        Returns:
            Optional[Tuple[str, str, str]]: 包含最匹配的 (标准参数名, 标准参数值, 标准代码) 的元组。
                                           如果找不到匹配项或发生错误，则返回 None。
        """
        # Keep this INFO as it marks the start of a specific search operation
        logger.info(f"--- 开始参数标准化搜索: '{actual_param_name}'='{actual_param_value}' ---")
        # logger.info(f"输入 -> 实际参数名: '{actual_param_name}', 实际参数值: '{actual_param_value}'") # Redundant with above

        if not self.is_ready():
            logger.error("搜索失败：搜索服务未就绪。")
            return None
        if not actual_param_name or not actual_param_value:
            logger.error("搜索失败：实际参数名和实际参数值不能为空。")
            return None

        # 1. 构造组合查询文本并编码
        # 使用实际参数名和值来构建查询，以找到语义上最接近的标准条目
        query_text_combined = f"{actual_param_name}: {actual_param_value}"
        logger.debug(f"构造组合查询文本进行编码: '{query_text_combined}'")
        query_embedding_array = self._encode_query(query_text_combined)

        if query_embedding_array is None:
            logger.error("查询文本编码失败。")
            return None
        # ChromaDB 需要 list of lists
        query_embedding_list = [query_embedding_array.tolist()]

        # 2. 在向量数据库中执行查询
        # 查询比最终需要的结果稍多的数量，以防最佳匹配不是第一个
        initial_n_results = max(n_results, settings.DEFAULT_N_RESULTS) * settings.INITIAL_QUERY_MULTIPLIER
        logger.debug(f"向向量数据库查询 {initial_n_results} 个初始结果...")

        search_start_time = time.time()
        try:
            initial_vector_results = self.vector_store.query_collection(
                query_embeddings=query_embedding_list,
                n_results=initial_n_results,
                # where_filter=None, # 根据 V4 逻辑，不强制过滤类型
                include_fields=['metadatas', 'distances'] # 确保包含所需字段
            )
        except Exception as e:
            logger.error(f"向量数据库查询时出错: {e}", exc_info=True)
            return None
        finally:
            search_end_time = time.time()
            logger.debug(f"向量数据库查询耗时: {search_end_time - search_start_time:.4f} 秒")

        if initial_vector_results is None:
            logger.error("向量数据库查询失败或未返回结果。")
            return None

        # 3. 处理和选择最佳结果
        logger.debug("处理和格式化查询结果...")
        ids_list = initial_vector_results.get('ids', [[]])[0]
        distances_list = initial_vector_results.get('distances', [[]])[0]
        metadatas_list = initial_vector_results.get('metadatas', [[]])[0]

        if not ids_list:
            logger.warning(f"对于查询 '{query_text_combined}' 未找到任何语义相似的结果。")
            return None # 表示未找到匹配

        # 选择距离最近（最相似）的结果
        best_match_index = 0 # 结果默认按距离排序，第一个是最近的
        best_meta = metadatas_list[best_match_index] if metadatas_list else None
        best_distance = distances_list[best_match_index] if distances_list else float('inf')

        if best_meta is None:
            logger.error("最佳匹配结果缺少元数据。")
            return None

        # 4. 从最佳匹配的元数据中提取所需信息
        standard_param_name = best_meta.get(settings.META_FIELD_PARAM_TYPE)
        standard_param_value = best_meta.get(settings.META_FIELD_STANDARD_VALUE)
        standard_code = best_meta.get(settings.META_FIELD_CODE)

        # 检查提取的值是否有效
        if not standard_param_name or not standard_param_value:
             logger.warning(f"最佳匹配结果 (距离: {best_distance:.4f}) 的元数据中缺少标准参数名或标准值。元数据: {best_meta}")
             # 根据需求决定是返回 None 还是尝试下一个结果
             return None # 当前选择返回 None

        # Keep this INFO as it shows the result of the search operation
        logger.info(f"搜索完成。最佳匹配 (距离: {best_distance:.4f}) -> 标准名: '{standard_param_name}', 标准值: '{standard_param_value}', 标准代码: '{standard_code}'")

        return standard_param_name, standard_param_value, standard_code

# 注意：原 __main__ 部分已移除，因为此类应作为模块被 pipeline 调用。
