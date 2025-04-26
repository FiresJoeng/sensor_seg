# src/search_service.py
import time
from typing import Optional, List, Dict, Any
import numpy as np
# 可选：如果需要更复杂的文本相似度计算，可以导入 scikit-learn
# from sklearn.metrics.pairwise import cosine_similarity

# 从其他模块导入类和配置
try:
    from . import config
    from .embedding_generator import EmbeddingGenerator
    from .vector_store_manager import VectorStoreManager
except ImportError:
    import config
    from embedding_generator import EmbeddingGenerator
    from vector_store_manager import VectorStoreManager

class SearchService:
    """
    封装参数知识库的核心搜索逻辑。
    接收用户输入，利用嵌入模型和向量存储进行语义搜索，并对结果进行处理和过滤。
    """
    def __init__(self):
        """初始化搜索服务，加载所需组件（嵌入生成器和向量存储管理器）。"""
        print("--- 初始化搜索服务 ---")
        # 实例化依赖的组件
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStoreManager()
        # 标记组件是否初始化成功
        self.model_loaded = False
        self.db_connected = False
        # 调用内部方法进行初始化
        self._initialize_components()

    def _initialize_components(self):
        """加载嵌入模型并连接到向量数据库。"""
        print("正在加载嵌入模型...")
        # 加载模型并更新状态
        self.model_loaded = self.embedding_generator.load_model()
        if not self.model_loaded:
            print("错误: 嵌入模型加载失败，搜索服务可能无法正常工作。")
            # 根据需求，这里可以引发异常或允许服务在部分不可用的状态下运行

        print("\n正在连接向量数据库...")
        # 连接数据库并更新状态
        self.db_connected = self.vector_store.connect()
        if not self.db_connected:
            print("错误: 向量数据库连接失败，搜索服务可能无法正常工作。")
            # 同上，根据需求处理连接失败的情况

    def is_ready(self) -> bool:
        """
        检查搜索服务是否准备就绪。

        Returns:
            bool: 如果模型已加载且数据库已连接，则返回 True。
        """
        return self.model_loaded and self.db_connected

    def _encode_query(self, query_text: str) -> Optional[np.ndarray]:
        """
        使用加载的模型对单个查询文本进行编码（生成嵌入向量）。

        Args:
            query_text (str): 要编码的查询文本。

        Returns:
            Optional[np.ndarray]: 查询文本的嵌入向量 NumPy 数组，如果模型未加载或编码出错则返回 None。
        """
        # 检查模型是否可用
        if not self.embedding_generator.model:
            print("错误: 嵌入模型不可用，无法编码查询。")
            return None
        try:
            # 调用模型的 encode 方法，注意输入需要是列表形式
            # 返回的是一个包含单个嵌入向量的 NumPy 数组
            return self.embedding_generator.model.encode([query_text])
        except Exception as e:
            # 捕获编码过程中可能发生的错误
            print(f"查询编码时出错: {e}")
            return None

    def search(self, parameter_type: str, parameter_value: str,
               n_results: int = config.DEFAULT_N_RESULTS) -> Optional[List[Dict[str, Any]]]:
        """
        执行参数搜索的核心方法。

        Args:
            parameter_type (str): 用户输入的参数类型。
            parameter_value (str): 用户输入的参数值。
            n_results (int): 希望最终返回的结果数量。

        Returns:
            Optional[List[Dict[str, Any]]]: 包含搜索结果字典的列表，按相关性（距离）排序。
                                            如果服务未就绪、输入无效或查询出错，返回 None。
                                            如果未找到匹配结果，返回空列表 []。
        """
        print(f"\n--- 开始搜索 ---")
        print(f"查询类型: '{parameter_type}', 查询值: '{parameter_value}'")

        # 检查服务是否准备就绪
        if not self.is_ready():
            print("错误: 搜索服务未就绪 (模型或数据库问题)。")
            return None
        # 检查输入是否有效
        if not parameter_type or not parameter_value:
             print("错误: 参数类型和参数值不能为空。")
             return None

        # 1. 构造查询文本并编码
        # 将参数类型和参数值结合，形成更具上下文的查询文本
        query_text_with_context = f"{parameter_type}: {parameter_value}"
        print(f"构造查询嵌入文本: '{query_text_with_context}'")
        # 获取查询文本的嵌入向量
        query_embedding = self._encode_query(query_text_with_context)

        # 检查编码是否成功
        if query_embedding is None:
            print("查询编码失败。")
            return None

        # 2. 在向量数据库中执行初步查询
        # 为了提高过滤后的结果质量，初始查询时获取更多结果
        initial_n_results = n_results * config.INITIAL_QUERY_MULTIPLIER
        print(f"向向量数据库查询 {initial_n_results} 个初始结果...")

        # !! 关键优化: 使用 'where' 子句在向量搜索时直接过滤参数类型 !!
        # 这比获取大量结果后再在 Python 中过滤要高效得多。
        # 过滤器使用 config 中定义的元数据字段名。
        where_filter = {config.META_FIELD_PARAM_TYPE: parameter_type}
        print(f"应用元数据过滤: {where_filter}")

        search_start_time = time.time()
        # 调用 VectorStoreManager 的查询方法
        initial_vector_results = self.vector_store.query_collection(
            query_embeddings=query_embedding.tolist(), # 查询嵌入需要是 list of lists
            n_results=initial_n_results,
            where_filter=where_filter, # 应用类型过滤器
            include_fields=['metadatas', 'distances'] # 指定需要返回的字段
        )
        search_end_time = time.time()
        print(f"向量数据库查询耗时: {search_end_time - search_start_time:.4f} 秒")

        # 检查查询是否成功
        if initial_vector_results is None:
            print("向量数据库查询失败。")
            return None

        # 3. 处理和格式化结果
        print("处理和格式化查询结果...")
        results_data = []
        # 从查询结果字典中提取所需信息
        # 使用 .get() 并提供默认空列表，以防某些键不存在
        ids_list = initial_vector_results.get('ids', [[]])[0]
        distances_list = initial_vector_results.get('distances', [[]])[0]
        metadatas_list = initial_vector_results.get('metadatas', [[]])[0]

        # 如果查询没有返回任何 ID，表示未找到匹配项
        if not ids_list:
            print(f"对于类型 '{parameter_type}' 未找到语义匹配的结果。")
            return [] # 返回空列表表示未找到

        # 遍历查询返回的每个结果
        for i in range(len(ids_list)):
            meta = metadatas_list[i]
            distance = distances_list[i] # 距离越小表示越相似

            # 构建格式化的结果字典，方便后续使用
            formatted_result = {
                "distance": round(distance, 4), # 保留4位小数
                # 使用 .get() 并提供默认值 'N/A'，以防元数据中缺少某个字段
                "standard_value": meta.get(config.META_FIELD_STANDARD_VALUE, 'N/A'),
                "standard_code": meta.get(config.META_FIELD_STANDARD_CODE, 'N/A'),
                "actual_value": meta.get(config.META_FIELD_ACTUAL_VALUE, 'N/A'), # 匹配的具体写法
                "parameter_type": meta.get(config.META_FIELD_PARAM_TYPE, 'N/A'),
                "component_part": meta.get(config.META_FIELD_COMPONENT, 'N/A'),
                # "raw_metadata": meta # (可选) 保留原始元数据以供调试
            }
            results_data.append(formatted_result)

        # 4. 排序和截断
        # ChromaDB 返回的结果已经按距离（相似度）排序，所以通常不需要再次排序
        # 如果需要实现更复杂的重排序逻辑（例如结合关键词匹配分数），可以在这里添加
        # reranked_results = self._rerank_results(results_data, parameter_value) # 示例调用
        # final_results = reranked_results[:n_results] # 截取重排序后的结果

        # 直接截取按距离排序的前 n_results 个结果
        final_results = results_data[:n_results]

        # 5. 添加排名信息
        for rank, item in enumerate(final_results):
            item['rank'] = rank + 1

        print(f"搜索完成，找到 {len(final_results)} 条相关结果。")
        return final_results

    # --- (可选) 示例：实现更复杂的重排序逻辑 ---
    # def _rerank_results(self, results: List[Dict[str, Any]], query_value: str) -> List[Dict[str, Any]]:
    #     """
    #     (示例) 对初步结果进行重排序，可以结合语义相似度和关键词匹配等。
    #     例如，可以计算 query_value 和 actual_value 之间的编辑距离或 Jaccard 相似度，
    #     然后结合向量距离给出一个综合分数进行排序。
    #     或者使用更高级的 Cross-Encoder 模型进行重排序。
    #     """
    #     print("  (可选) 正在执行重排序...")
    #     # 这里仅作演示，返回按原始距离排序的结果
    #     # 实际实现需要根据具体需求编写重排序算法
    #     return sorted(results, key=lambda x: x['distance'])


# --- 可选的直接执行入口 (用于测试服务) ---
if __name__ == "__main__":
    print("正在测试搜索服务...")
    search_service = SearchService()

    if search_service.is_ready():
        print("\n搜索服务已准备就绪。")
        # --- 定义测试查询 ---
        test_queries = [
            {"type": "输出信号", "value": "4-20mA hart"},
            {"type": "过程连接（法兰等级）", "value": "PN16 RF"},
            {"type": "铠套材质", "value": "316"},
            {"type": "不存在的类型", "value": "随便什么值"}, # 测试未找到类型
            {"type": "输出信号", "value": "不存在的值"},     # 测试未找到值
        ]

        for query in test_queries:
            test_type = query["type"]
            test_value = query["value"]
            print(f"\n--- 正在执行测试查询: 类型='{test_type}', 值='{test_value}' ---")

            results = search_service.search(test_type, test_value, n_results=3) # 获取前3个结果

            if results is not None:
                if results: # 如果结果列表不为空
                    print(f"--- 测试查询结果 ---")
                    for res in results:
                        print(f"\n  排名 {res['rank']} (距离: {res['distance']})")
                        print(f"    标准值: {res['standard_value']}")
                        print(f"    代码: {res['standard_code']}")
                        print(f"    匹配值: {res['actual_value']}")
                        print(f"    类型: {res['parameter_type']}")
                        print(f"    部件: {res['component_part']}")
                else: # 如果结果列表为空
                     print(f"未找到关于 类型='{test_type}', 值='{test_value}' 的结果。")
            else: # 如果 search 方法返回 None (表示查询出错)
                print("测试查询执行失败。")
            print("-" * 20) # 分隔符

    else:
        print("\n搜索服务初始化失败，无法执行测试查询。")
