# src/vector_store_manager.py
import chromadb
import numpy as np
import pandas as pd
import time
import os
from typing import Optional, List, Dict, Any, Tuple

# 从 config 模块导入配置
try:
    from . import config
except ImportError:
    import config

class VectorStoreManager:
    """
    管理与 ChromaDB 向量数据库的交互，包括连接、创建/获取集合、添加数据和查询。
    """
    def __init__(self, db_path: str = config.CHROMA_DB_PATH,
                 collection_name: str = config.COLLECTION_NAME):
        """
        初始化 VectorStoreManager。

        Args:
            db_path (str): ChromaDB 持久化存储的目录路径。
            collection_name (str): 要操作的集合名称。
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client: Optional[chromadb.PersistentClient] = None # ChromaDB 客户端对象
        self.collection: Optional[chromadb.Collection] = None # ChromaDB 集合对象

    def connect(self) -> bool:
        """
        连接到 ChromaDB 并获取/创建指定的集合。

        Returns:
            bool: 如果连接和获取集合成功返回 True，否则返回 False。
        """
        print(f"--- 正在连接到向量数据库 ---")
        print(f"数据库路径: {self.db_path}")
        print(f"集合名称: {self.collection_name}")
        # 如果已经连接，则直接返回 True
        if self.collection is not None:
            print("已连接到数据库集合。")
            return True
        try:
            # 确保数据库目录存在，如果不存在则创建
            os.makedirs(self.db_path, exist_ok=True)
            # 创建持久化客户端，数据将保存在指定的 db_path
            self.client = chromadb.PersistentClient(path=self.db_path)
            # 获取或创建集合
            # 如果集合已存在，则获取它；如果不存在，则创建它
            # 可以指定 embedding_function，但我们选择在外部生成嵌入
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            print(f"成功连接/创建集合 '{self.collection_name}'。")
            # 打印当前集合中的数据条数
            print(f"当前集合包含 {self.collection.count()} 条数据。")
            return True
        except Exception as e:
            # 捕获连接或获取集合时可能发生的错误
            print(f"连接或获取 ChromaDB 集合时出错: {e}")
            self.client = None
            self.collection = None
            return False

    def load_data_for_indexing(self, embeddings_path: str = config.EMBEDDINGS_PATH,
                               metadata_path: str = config.METADATA_PATH) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
        """
        从文件加载嵌入向量和元数据，用于索引构建。

        Args:
            embeddings_path (str): .npy 嵌入向量文件路径。
            metadata_path (str): .csv 元数据文件路径。

        Returns:
            Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
                包含加载的嵌入向量和元数据 DataFrame 的元组。如果出错则返回 (None, None)。
        """
        print("\n--- 正在加载嵌入向量和元数据以供索引 ---")
        embeddings = None
        df_metadata = None

        # 加载嵌入向量 .npy 文件
        if not os.path.exists(embeddings_path):
            print(f"错误: 嵌入向量文件未找到于 '{embeddings_path}'")
            return None, None
        try:
            embeddings = np.load(embeddings_path)
            print(f"已加载嵌入向量，形状: {embeddings.shape}")
        except Exception as e:
            print(f"加载嵌入向量 '{embeddings_path}' 时出错: {e}")
            return None, None

        # 加载元数据 .csv 文件
        if not os.path.exists(metadata_path):
            print(f"错误: 元数据文件未找到于 '{metadata_path}'")
            return None, None
        try:
            # 读取为字符串并填充 NaN
            df_metadata = pd.read_csv(metadata_path, dtype=str).fillna('')
            print(f"已加载元数据，形状: {df_metadata.shape}")
        except Exception as e:
            print(f"加载元数据 '{metadata_path}' 时出错: {e}")
            return None, None

        # 检查加载的数据是否有效且数量匹配
        if embeddings is not None and df_metadata is not None:
            if len(embeddings) != len(df_metadata):
                print("错误: 嵌入向量数量与元数据行数不匹配！")
                print(f"  嵌入数量: {len(embeddings)}, 元数据行数: {len(df_metadata)}")
                return None, None # 数量不匹配，返回 None

        return embeddings, df_metadata

    def build_index(self, embeddings: np.ndarray, df_metadata: pd.DataFrame, batch_size: int = 1000) -> bool:
        """
        使用提供的嵌入向量和元数据填充 ChromaDB 集合（构建索引）。

        Args:
            embeddings (np.ndarray): 嵌入向量数组。
            df_metadata (pd.DataFrame): 对应的元数据 DataFrame。
            batch_size (int): 单次添加到数据库的批次大小，用于处理大数据集。

        Returns:
            bool: 如果索引构建（数据添加）成功返回 True，否则返回 False。
        """
        print("\n--- 正在构建/填充向量索引 ---")
        # 检查数据库连接和输入数据
        if self.collection is None:
            print("错误: 未连接到数据库集合。请先调用 connect()。")
            return False
        if embeddings is None or df_metadata is None:
            print("错误: 缺少嵌入向量或元数据，无法构建索引。")
            return False
        if len(embeddings) != len(df_metadata):
             print("错误: 嵌入向量数量与元数据行数不匹配！")
             return False

        target_count = len(df_metadata) # 目标要添加的数据量
        current_count = self.collection.count() # 当前集合中的数据量
        print(f"目标索引大小: {target_count} 条，当前集合大小: {current_count} 条。")

        # 可选：如果集合非空，可以给出提示或执行清空操作
        # if current_count > 0:
        #     print("警告: 集合非空，可能包含旧数据。如果需要全新索引，请考虑清空集合。")
            # self.clear_collection() # 调用清空方法

        # 准备数据以符合 ChromaDB 的格式要求
        print("准备数据以添加到 ChromaDB...")
        embeddings_list = embeddings.tolist() # ChromaDB 需要 list 格式的嵌入

        # 准备元数据列表，每个元素是一个字典
        metadatas = []
        try:
            # 遍历元数据 DataFrame 的每一行
            for index, row in df_metadata.iterrows():
                # 构建元数据字典，key 使用 config 中定义的字段名
                metadata_item = {
                    config.META_FIELD_STANDARD_VALUE: row[config.COL_STANDARD_VALUE_DESC],
                    config.META_FIELD_STANDARD_CODE: row[config.COL_STANDARD_CODE],
                    config.META_FIELD_ACTUAL_VALUE: row[config.PREPROCESSED_COL_ACTUAL_VARIATION],
                    config.META_FIELD_PARAM_TYPE: row[config.COL_STANDARD_PARAM],
                    config.META_FIELD_COMPONENT: row[config.COL_COMPONENT_PART]
                }
                metadatas.append(metadata_item)
        except KeyError as e:
             # 如果 DataFrame 中缺少配置的列名，捕获错误
             print(f"错误: 准备元数据时找不到列: {e}。请检查 CSV 文件和 config.py 中的列名配置。")
             return False

        # 准备 ID 列表，确保每个 ID 是唯一的字符串
        # 使用 DataFrame 的索引作为 ID 是一个简单可靠的方法
        ids = [str(i) for i in df_metadata.index]

        print(f"开始向集合 '{self.collection_name}' 添加 {len(ids)} 条数据 (批大小: {batch_size})...")
        start_time = time.time()
        added_count = 0
        try:
            # 分批次添加数据到 ChromaDB，避免一次性加载过多数据到内存
            for i in range(0, len(ids), batch_size):
                # 获取当前批次的数据
                batch_ids = ids[i : i + batch_size]
                batch_embeddings = embeddings_list[i : i + batch_size]
                batch_metadatas = metadatas[i : i + batch_size]

                # 如果批次为空（可能发生在最后），则跳过
                if not batch_ids:
                    continue

                # 使用 upsert 方法添加或更新数据
                # upsert: 如果 ID 已存在，则更新；如果不存在，则插入。比 add 更安全。
                self.collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
                added_count += len(batch_ids)
                # 打印进度
                print(f"  已添加 {added_count}/{len(ids)} 条数据...")

            end_time = time.time()
            print(f"数据添加/更新完成，耗时 {end_time - start_time:.2f} 秒。")
            final_count = self.collection.count() # 获取添加后的总数
            print(f"添加后集合总数: {final_count}")
            # 简单检查最终数量是否符合预期
            if final_count < target_count:
                 print(f"警告: 最终集合大小 ({final_count}) 小于目标大小 ({target_count})。可能存在问题。")
            return True
        except Exception as e:
            # 捕获添加数据时可能发生的错误
            print(f"向 ChromaDB 集合添加数据时出错: {e}")
            return False

    def query_collection(self, query_embeddings: List[List[float]],
                         n_results: int = config.DEFAULT_N_RESULTS,
                         where_filter: Optional[Dict[str, Any]] = None,
                         include_fields: List[str] = ['metadatas', 'distances']) -> Optional[Dict]:
        """
        在 ChromaDB 集合中执行向量相似度查询。

        Args:
            query_embeddings (List[List[float]]): 查询文本的嵌入向量列表 (即使只有一个查询，也需要是列表的列表)。
            n_results (int): 每个查询希望返回的最大结果数量。
            where_filter (Optional[Dict[str, Any]]): 用于在查询时过滤元数据的条件字典。
                                                    例如: {"parameter_type": "输出信号"}
                                                    支持的操作符: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
                                                    参考 ChromaDB 文档获取更多过滤选项。
            include_fields (List[str]): 查询结果中希望包含的字段列表 ('metadatas', 'documents', 'distances', 'embeddings')。

        Returns:
            Optional[Dict]: ChromaDB 的查询结果字典，如果出错则返回 None。
                           结果字典结构通常包含 'ids', 'distances', 'metadatas' 等键。
        """
        # print("\n--- 正在执行向量查询 ---") # 调用频率可能较高，移到调用方打印
        # 检查数据库连接和查询输入
        if self.collection is None:
            print("错误: 未连接到数据库集合。请先调用 connect()。")
            return None
        if not query_embeddings:
             print("错误: 查询嵌入向量列表不能为空。")
             return None

        try:
            # print(f"查询参数: n_results={n_results}, filter={where_filter}, include={include_fields}")
            start_time = time.time()
            # 调用集合的 query 方法执行查询
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where_filter, # 应用元数据过滤器
                include=include_fields # 指定返回字段
            )
            end_time = time.time()
            # print(f"查询耗时: {end_time - start_time:.4f} 秒。")
            return results
        except Exception as e:
            # 捕获查询时可能发生的错误
            print(f"查询 ChromaDB 集合时出错: {e}")
            return None

    def get_all_metadata_values(self, metadata_field: str) -> List[str]:
        """
        获取集合中指定元数据字段的所有唯一值。
        注意：对于非常大的集合，此操作可能非常耗时和消耗内存。

        Args:
            metadata_field (str): 要获取唯一值的元数据字段名称 (例如: config.META_FIELD_PARAM_TYPE)。

        Returns:
            List[str]: 包含所有唯一值的列表，按字母顺序排序。如果出错或无数据则返回空列表。
        """
        print(f"\n--- 正在获取元数据字段 '{metadata_field}' 的唯一值 ---")
        if self.collection is None:
            print("错误: 未连接到数据库集合。")
            return []
        try:
            # 使用 get() 获取集合中的所有条目，只包含元数据部分
            # 警告：如果集合非常大，这可能会加载大量数据到内存中！
            print("  正在从数据库获取所有元数据 (大集合可能耗时)...")
            results = self.collection.get(include=['metadatas'])
            print("  元数据获取完成。")

            if results and results.get('metadatas'):
                # 使用集合推导式提取指定字段的唯一值
                # 处理 meta 为 None 或 meta.get(metadata_field) 为 None 的情况
                unique_values = {
                    meta.get(metadata_field)
                    for meta in results['metadatas']
                    if meta and meta.get(metadata_field) is not None
                }
                print(f"找到 {len(unique_values)} 个唯一值。")
                # 返回排序后的列表
                return sorted(list(unique_values))
            else:
                print("  未找到元数据。")
                return []
        except Exception as e:
            # 捕获获取元数据时可能发生的错误
            print(f"获取元数据字段 '{metadata_field}' 的值时出错: {e}")
            return []

    def clear_collection(self) -> bool:
        """
        (危险操作!) 清空当前集合中的所有数据。
        这将删除集合并重新创建一个同名的空集合。

        Returns:
            bool: 如果清空成功返回 True，否则返回 False。
        """
        print(f"\n--- 警告: 即将清空集合 '{self.collection_name}' ---")
        # 检查客户端和集合对象是否存在
        if self.collection is None or self.client is None:
            print("错误: 未连接到数据库或客户端，无法清空。")
            return False
        try:
            # 添加二次确认步骤，防止误操作
            confirm = input(f"这是一个危险操作！确定要删除集合 '{self.collection_name}' 中的所有数据吗? (输入 'yes' 确认): ")
            if confirm.lower() == 'yes':
                print(f"正在删除集合 '{self.collection_name}'...")
                start_time = time.time()
                # 调用客户端删除集合
                self.client.delete_collection(name=self.collection_name)
                # 重新创建同名的空集合
                self.collection = self.client.create_collection(name=self.collection_name)
                end_time = time.time()
                print(f"集合已清空并重新创建，耗时 {end_time - start_time:.2f} 秒。")
                return True
            else:
                print("操作已取消。")
                return False
        except Exception as e:
            # 捕获清空集合时可能发生的错误
            print(f"清空集合时出错: {e}")
            # 清空失败后，集合和客户端状态可能未知，设为 None
            self.collection = None
            self.client = None
            return False

# --- 可选的直接执行入口 (用于测试连接和基本功能) ---
if __name__ == "__main__":
    print("正在测试向量数据库管理器...")
    manager = VectorStoreManager()
    if manager.connect():
        print("\n数据库连接测试成功。")
        count = manager.collection.count()
        print(f"集合 '{manager.collection_name}' 当前包含 {count} 条数据。")

        # --- 可选测试：获取唯一参数类型 ---
        # print("\n测试获取唯一参数类型...")
        # param_types = manager.get_all_metadata_values(config.META_FIELD_PARAM_TYPE)
        # if param_types:
        #     print("\n找到的唯一参数类型示例 (最多显示10个):")
        #     print(param_types[:10])
        # else:
        #      print("\n未能获取唯一参数类型，可能是集合为空或字段名错误。")

        # --- 可选测试：查询 (需要先构建索引) ---
        # print("\n测试查询 (需要先构建索引)...")
        # # 假设我们有一个查询嵌入 (实际应用中由 EmbeddingGenerator 生成)
        # # 这里的维度需要与你的模型一致 (例如 paraphrase-multilingual-mpnet-base-v2 是 768 维)
        # dummy_embedding = [[0.1] * 768] # 创建一个假的 768 维嵌入向量
        # results = manager.query_collection(query_embeddings=dummy_embedding, n_results=3)
        # if results:
        #     print("\n测试查询结果:")
        #     print(results)
        # else:
        #     print("\n测试查询失败或未返回结果。")

    else:
        print("\n数据库连接测试失败。")
