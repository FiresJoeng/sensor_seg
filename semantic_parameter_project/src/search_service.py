# src/search_service.py
import time
from typing import Optional, List, Dict, Any
import numpy as np

# 从其他模块导入类和配置
try:
    from . import config
    from .embedding_generator import EmbeddingGenerator # 需要访问模型
    from .vector_store_manager import VectorStoreManager
except ImportError:
    import config
    from embedding_generator import EmbeddingGenerator
    from vector_store_manager import VectorStoreManager

class SearchService:
    """
    封装参数知识库的核心搜索逻辑。
    ** V4: 查询时组合 '标准参数类型' 和 '实际参数值' 进行编码，
           但不强制使用 '标准参数类型' 进行精确过滤。**
    """
    def __init__(self):
        """初始化搜索服务，加载所需组件。"""
        print("--- 初始化搜索服务 ---")
        self.embedding_generator = EmbeddingGenerator() # 用于加载和访问模型
        self.vector_store = VectorStoreManager()
        self.model_loaded = False
        self.db_connected = False
        self._initialize_components()

    def _initialize_components(self):
        """加载嵌入模型并连接到向量数据库。"""
        print("正在加载嵌入模型...")
        self.model_loaded = self.embedding_generator.load_model()
        if not self.model_loaded: print("错误: 嵌入模型加载失败。"); return

        print("\n正在连接向量数据库...")
        self.db_connected = self.vector_store.connect()
        if not self.db_connected: print("错误: 向量数据库连接失败。")

    def is_ready(self) -> bool:
        """检查搜索服务是否准备就绪。"""
        return self.model_loaded and self.db_connected

    def _encode_query(self, query_text: str) -> Optional[np.ndarray]:
        """对组合后的查询文本进行编码。"""
        if not self.embedding_generator.model: print("错误: 嵌入模型不可用。"); return None
        try:
            return self.embedding_generator.model.encode([query_text])
        except Exception as e: print(f"查询文本编码时出错: {e}"); return None

    def search(self, parameter_type: str, parameter_value: str,
               n_results: int = config.DEFAULT_N_RESULTS) -> Optional[List[Dict[str, Any]]]:
        """
        执行参数搜索。
        ** V4: 组合 '标准参数类型' 和 '实际参数值' 进行语义搜索，
               不再强制使用 '标准参数类型' 进行精确过滤。**

        Args:
            parameter_type (str): 用户输入的标准参数类型 (用于构造查询文本)。
            parameter_value (str): 用户输入的实际参数值 (用于构造查询文本)。
            n_results (int): 希望最终返回的结果数量。

        Returns:
            Optional[List[Dict[str, Any]]]: 包含搜索结果字典的列表，按相关性排序。
        """
        print(f"\n--- 开始搜索 (V4 - 无强制类型过滤) ---")
        # 注意：虽然我们不再强制过滤，但类型信息仍然用于构造查询向量
        print(f"查询类型（用于构造向量）: '{parameter_type}', 搜索值: '{parameter_value}'")

        if not self.is_ready(): print("错误: 搜索服务未就绪。"); return None
        if not parameter_type or not parameter_value: print("错误: 参数类型和参数值不能为空。"); return None

        # 1. 构造组合查询文本并编码
        query_text_combined = f"{parameter_type}: {parameter_value}"
        print(f"构造组合查询文本进行编码: '{query_text_combined}'")
        query_embedding_array = self._encode_query(query_text_combined)

        if query_embedding_array is None: print("查询文本编码失败。"); return None
        query_embedding_list = query_embedding_array.tolist()

        # 2. 在向量数据库中执行查询，**不再使用 where_filter**
        initial_n_results = n_results * config.INITIAL_QUERY_MULTIPLIER
        print(f"向向量数据库查询 {initial_n_results} 个初始结果 (无类型过滤)...")
        # where_filter = {config.META_FIELD_PARAM_TYPE: parameter_type} # V4: 注释掉或移除这行
        # print(f"应用元数据过滤: {where_filter}") # V4: 注释掉或移除这行

        search_start_time = time.time()
        initial_vector_results = self.vector_store.query_collection(
            query_embeddings=query_embedding_list,
            n_results=initial_n_results,
            # where_filter=where_filter, # V4: 不再传递 where_filter
            include_fields=['metadatas', 'distances']
        )
        search_end_time = time.time()
        print(f"向量数据库查询耗时: {search_end_time - search_start_time:.4f} 秒")

        if initial_vector_results is None: print("向量数据库查询失败。"); return None

        # 3. 处理和格式化结果
        print("处理和格式化查询结果...")
        results_data = []
        ids_list = initial_vector_results.get('ids', [[]])[0]
        distances_list = initial_vector_results.get('distances', [[]])[0]
        metadatas_list = initial_vector_results.get('metadatas', [[]])[0]

        if not ids_list:
            # 现在如果找不到，就是纯粹的语义不匹配了
            print(f"对于查询 '{query_text_combined}' 未找到任何语义相似的结果。")
            return []

        for i in range(len(ids_list)):
            meta = metadatas_list[i]
            if meta is None: meta = {}
            distance = distances_list[i]

            formatted_result = {
                "distance": round(distance, 4),
                config.META_FIELD_COMPONENT: meta.get(config.META_FIELD_COMPONENT, 'N/A'),
                config.META_FIELD_PARAM_TYPE: meta.get(config.META_FIELD_PARAM_TYPE, 'N/A'),
                config.META_FIELD_ACTUAL_PARAM_DESC: meta.get(config.META_FIELD_ACTUAL_PARAM_DESC, 'N/A'),
                config.META_FIELD_STANDARD_VALUE: meta.get(config.META_FIELD_STANDARD_VALUE, 'N/A'),
                config.META_FIELD_ACTUAL_VALUE: meta.get(config.META_FIELD_ACTUAL_VALUE, 'N/A'),
                config.META_FIELD_DEFAULT: meta.get(config.META_FIELD_DEFAULT, 'N/A'),
                config.META_FIELD_CODE: meta.get(config.META_FIELD_CODE, 'N/A'),
                config.META_FIELD_FIELD_DESC: meta.get(config.META_FIELD_FIELD_DESC, 'N/A'),
                config.META_FIELD_REMARK: meta.get(config.META_FIELD_REMARK, 'N/A')
            }
            results_data.append(formatted_result)

        # 4. (可选) 结果后处理/重排序
        # 因为没有预先过滤，返回的结果可能包含其他类型的参数。
        # 可以在这里根据元数据中的 'parameter_type' 与用户输入的 'parameter_type' 的匹配度进行重排序。
        # 例如，完全匹配的排在前面，或者计算类型字符串的相似度。
        # 这里暂时只按距离排序。
        print("（可选步骤：可以根据类型匹配度进行重排序）")
        final_results = sorted(results_data, key=lambda x: x['distance'])[:n_results]


        # 5. 添加排名
        for rank, item in enumerate(final_results):
            item['rank'] = rank + 1

        print(f"搜索完成，找到 {len(final_results)} 条相关结果 (可能包含不同类型)。")
        return final_results

# --- 可选的直接执行入口 (用于测试服务) ---
if __name__ == "__main__":
    print("正在测试搜索服务 (V4 - 无类型过滤)...")
    search_service = SearchService()
    if search_service.is_ready():
        print("\n搜索服务已准备就绪。")
        # 测试之前失败的查询
        test_queries = [
             {"type": "电气接口", "value": "M20×1.5"},
             {"type": "Connection Size", "value": "PN15.0 RJ"},
             {"type": "接线盒材质", "value": "合金铝"},
        ]
        for query in test_queries:
             test_type = query["type"]
             test_value = query["value"]
             print(f"\n--- 正在执行测试查询: 类型='{test_type}', 值='{test_value}' ---")
             results = search_service.search(test_type, test_value, n_results=5) # 获取前5个结果
             if results is not None:
                 if results:
                     print(f"--- 测试查询结果 ---")
                     for res in results:
                         print(f"\n  排名 {res['rank']} (距离: {res['distance']})")
                         # 打印所有字段以供检查
                         print(f"    Component: {res.get(config.META_FIELD_COMPONENT, 'N/A')}")
                         print(f"    Param Type: {res.get(config.META_FIELD_PARAM_TYPE, 'N/A')}") # 注意看这个类型是否相关
                         print(f"    Actual Param Desc: {res.get(config.META_FIELD_ACTUAL_PARAM_DESC, 'N/A')}")
                         print(f"    Standard Value: {res.get(config.META_FIELD_STANDARD_VALUE, 'N/A')}")
                         print(f"    Actual Value: {res.get(config.META_FIELD_ACTUAL_VALUE, 'N/A')}")
                         print(f"    Default: {res.get(config.META_FIELD_DEFAULT, 'N/A')}")
                         print(f"    Code: {res.get(config.META_FIELD_CODE, 'N/A')}")
                         print(f"    Field Desc: {res.get(config.META_FIELD_FIELD_DESC, 'N/A')}")
                         print(f"    Remark: {res.get(config.META_FIELD_REMARK, 'N/A')}")
                 else:
                      print(f"未找到关于 查询='{test_type}: {test_value}' 的结果。")
             else:
                 print("测试查询执行失败。")
             print("-" * 20)
    else:
        print("\n搜索服务初始化失败。")
