# new_sensor_project/src/parameter_standardizer/vector_store_manager.py
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import chromadb
import numpy as np
import pandas as pd

# 导入项目配置
try:
    from ..config import settings
except ImportError:
    # 适应不同的导入上下文
    from config import settings

# 获取日志记录器实例
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    管理与 ChromaDB 向量数据库的交互。
    包括连接、构建索引（由脚本调用）和查询（由 SearchService 调用）。
    """
    def __init__(self, db_path: Optional[Path] = None, collection_name: Optional[str] = None):
        """
        初始化 VectorStoreManager。

        Args:
            db_path (Optional[Path]): ChromaDB 数据库的路径。如果为 None，则从配置加载。
            collection_name (Optional[str]): 要使用的集合名称。如果为 None，则从配置加载。
        """
        self.db_path = db_path or settings.KB_VECTOR_STORE_DIR
        self.collection_name = collection_name or settings.VECTOR_DB_COLLECTION_NAME
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[chromadb.Collection] = None
        logger.info(f"VectorStoreManager 初始化。DB路径: {self.db_path}, 集合: {self.collection_name}")

    def connect(self) -> bool:
        """
        连接到 ChromaDB 并获取/创建指定的集合。

        Returns:
            bool: 如果连接成功则返回 True，否则返回 False。
        """
        if self.collection is not None:
            logger.debug("已连接到向量数据库集合。")
            return True

        logger.info(f"--- 正在连接到向量数据库 ---")
        logger.info(f"数据库路径: {self.db_path}")
        logger.info(f"集合名称: {self.collection_name}")
        try:
            # 确保数据库目录存在
            self.db_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.db_path)) # PersistentClient 需要字符串路径
            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            logger.info(f"成功连接/创建集合 '{self.collection_name}'。")
            # 记录当前集合大小
            count = self.collection.count()
            logger.info(f"当前集合 '{self.collection_name}' 包含 {count} 条数据。")
            return True
        except Exception as e:
            logger.error(f"连接或获取 ChromaDB 集合 '{self.collection_name}' 时出错: {e}", exc_info=True)
            self.client = None
            self.collection = None
            return False

    def load_data_for_indexing(self, embeddings_path: Path, metadata_path: Path) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
        """
        从文件加载嵌入向量和元数据以供索引构建。
        (主要由知识库构建脚本调用)

        Args:
            embeddings_path (Path): 嵌入向量 .npy 文件路径。
            metadata_path (Path): 元数据 .csv 文件路径。

        Returns:
            Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]: 包含嵌入向量和元数据 DataFrame 的元组，
                                                               如果加载失败则对应项为 None。
        """
        logger.info("\n--- 正在加载嵌入向量和元数据以供索引 ---")
        embeddings: Optional[np.ndarray] = None
        df_metadata: Optional[pd.DataFrame] = None

        # 加载嵌入向量
        if not embeddings_path.is_file():
            logger.error(f"嵌入向量文件未找到: {embeddings_path}")
            return None, None
        try:
            embeddings = np.load(str(embeddings_path)) # np.load 需要字符串路径
            logger.info(f"已加载嵌入向量，形状: {embeddings.shape} (来自 {embeddings_path.name})")
        except Exception as e:
            logger.error(f"加载嵌入向量 '{embeddings_path.name}' 时出错: {e}", exc_info=True)
            return None, None

        # 加载元数据
        if not metadata_path.is_file():
            logger.error(f"元数据文件未找到: {metadata_path}")
            return embeddings, None # 可能只加载了嵌入
        try:
            # 读取时确保所有列为字符串，并填充 NaN
            df_metadata = pd.read_csv(metadata_path, dtype=str).fillna('')
            logger.info(f"已加载元数据，形状: {df_metadata.shape} (来自 {metadata_path.name})")
            # 可选：在这里添加列名验证，确保包含所有需要的元数据字段
            # required_meta_cols = [settings.SOURCE_COL_*, ...] # 定义需要的列
            # missing_cols = [col for col in required_meta_cols if col not in df_metadata.columns]
            # if missing_cols: ... log error and return

        except Exception as e:
            logger.error(f"加载元数据 '{metadata_path.name}' 时出错: {e}", exc_info=True)
            return embeddings, None # 可能只加载了嵌入

        # 验证嵌入和元数据数量是否匹配
        if embeddings is not None and df_metadata is not None:
            if len(embeddings) != len(df_metadata):
                logger.error(f"嵌入向量数量 ({len(embeddings)}) 与元数据行数 ({len(df_metadata)}) 不匹配！")
                return None, None # 数据不一致，返回 None

        return embeddings, df_metadata

    def build_index(self, embeddings: np.ndarray, df_metadata: pd.DataFrame, batch_size: int = 1000) -> bool:
        """
        使用提供的嵌入向量和元数据填充 ChromaDB 集合。
        (主要由知识库构建脚本调用)

        Args:
            embeddings (np.ndarray): 嵌入向量数组。
            df_metadata (pd.DataFrame): 包含元数据的 DataFrame。
            batch_size (int): 添加到数据库的批处理大小。

        Returns:
            bool: 如果索引构建成功则返回 True，否则返回 False。
        """
        logger.info("\n--- 正在构建/填充向量索引 ---")
        if self.collection is None:
            logger.error("无法构建索引：未连接到数据库集合。请先调用 connect()。")
            return False
        if embeddings is None or df_metadata is None:
            logger.error("无法构建索引：缺少嵌入向量或元数据。")
            return False
        if len(embeddings) != len(df_metadata):
            logger.error(f"无法构建索引：嵌入向量数量 ({len(embeddings)}) 与元数据行数 ({len(df_metadata)}) 不匹配！")
            return False

        target_count = len(df_metadata)
        current_count = self.collection.count()
        logger.info(f"目标索引大小: {target_count} 条，当前集合大小: {current_count} 条。")
        if target_count == 0:
             logger.warning("输入数据为空，无需构建索引。")
             return True # 认为空索引构建成功

        logger.info("准备元数据以添加到 ChromaDB...")
        embeddings_list = embeddings.tolist() # ChromaDB 需要列表
        metadatas: List[Dict[str, Any]] = []

        # 定义源列名到 ChromaDB 元数据字段名的映射
        metadata_mapping = {
            settings.META_FIELD_COMPONENT: settings.SOURCE_COL_COMPONENT_PART,
            settings.META_FIELD_PARAM_TYPE: settings.SOURCE_COL_STANDARD_PARAM,
            settings.META_FIELD_ACTUAL_PARAM_DESC: settings.SOURCE_COL_ACTUAL_PARAM_DESC,
            settings.META_FIELD_STANDARD_VALUE: settings.SOURCE_COL_STANDARD_VALUE_DESC,
            settings.META_FIELD_ACTUAL_VALUE: settings.SOURCE_COL_ACTUAL_VALUE_VARIATIONS,
            settings.META_FIELD_DEFAULT: settings.SOURCE_COL_DEFAULT_VALUE,
            settings.META_FIELD_CODE: settings.SOURCE_COL_STANDARD_CODE,
            settings.META_FIELD_FIELD_DESC: settings.SOURCE_COL_FIELD_DESC,
            settings.META_FIELD_REMARK: settings.SOURCE_COL_REMARK
            # 可以根据需要添加更多字段，例如来源文件名
            # settings.META_FIELD_SOURCE: 'source_column_name' # 如果有的话
        }

        # 检查 DataFrame 是否包含所有需要的源列
        missing_source_cols = [src_col for src_col in metadata_mapping.values() if src_col not in df_metadata.columns]
        if missing_source_cols:
            logger.error(f"元数据 DataFrame 中缺少以下必需的源列: {missing_source_cols}。无法构建索引。")
            return False

        # 迭代 DataFrame 行来构建元数据列表
        try:
            for index, row in df_metadata.iterrows():
                metadata_item = {
                    meta_field: row[source_col]
                    for meta_field, source_col in metadata_mapping.items()
                }
                metadatas.append(metadata_item)
        except KeyError as e:
             logger.error(f"准备元数据时找不到列 '{e}'。请检查元数据 CSV 文件列名是否与 config/settings.py 中的 SOURCE_COL_* 匹配。")
             return False
        except Exception as e:
             logger.error(f"准备元数据时发生未知错误: {e}", exc_info=True)
             return False

        # 生成 ID 列表 (使用 DataFrame 索引作为 ID)
        ids = [str(i) for i in df_metadata.index]

        logger.info(f"开始向集合 '{self.collection_name}' 添加/更新 {len(ids)} 条数据 (批大小: {batch_size})...")
        start_time = time.time()
        added_count = 0
        try:
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_embeddings = embeddings_list[i : i + batch_size]
                batch_metadatas = metadatas[i : i + batch_size]

                if not batch_ids: continue # 如果批次为空则跳过

                # 使用 upsert 来添加或更新数据
                self.collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
                added_count += len(batch_ids)
                logger.debug(f"  已处理 {added_count}/{len(ids)} 条数据...")

            end_time = time.time()
            logger.info(f"数据添加/更新完成。耗时: {end_time - start_time:.2f} 秒。")
            final_count = self.collection.count()
            logger.info(f"添加/更新后集合 '{self.collection_name}' 总数: {final_count}")
            # 验证最终数量是否符合预期
            if final_count < target_count:
                logger.warning(f"最终集合大小 ({final_count}) 小于目标大小 ({target_count})。可能存在未添加的数据或重复 ID。")
            elif final_count > target_count and current_count == 0 : # 如果是从空集合开始构建
                 logger.warning(f"最终集合大小 ({final_count}) 大于目标大小 ({target_count})。可能存在重复 ID 或数据源问题。")

            return True
        except Exception as e:
            logger.error(f"向 ChromaDB 集合 '{self.collection_name}' 添加数据时出错: {e}", exc_info=True)
            return False

    def query_collection(self, query_embeddings: List[List[float]],
                         n_results: int = settings.DEFAULT_N_RESULTS,
                         where_filter: Optional[Dict[str, Any]] = None,
                         include_fields: List[str] = ['metadatas', 'distances']) -> Optional[Dict]:
        """
        在 ChromaDB 集合中执行向量相似度查询。
        (主要由 SearchService 调用)

        Args:
            query_embeddings (List[List[float]]): 查询嵌入向量列表 (每个查询一个列表)。
            n_results (int): 每个查询希望返回的最大结果数量。
            where_filter (Optional[Dict[str, Any]]): 用于过滤元数据的条件。
            include_fields (List[str]): 查询结果中要包含的字段 ('metadatas', 'distances', 'documents', 'embeddings')。

        Returns:
            Optional[Dict]: 包含查询结果的字典，如果查询失败则返回 None。
        """
        if self.collection is None:
            logger.error("无法查询：未连接到数据库集合。")
            return None
        if not query_embeddings:
            logger.error("无法查询：查询嵌入向量列表不能为空。")
            return None

        logger.debug(f"开始查询集合 '{self.collection_name}'。查询数量: {len(query_embeddings)}, 返回结果数: {n_results}, 过滤条件: {where_filter}")
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where_filter,
                include=include_fields
            )
            logger.debug(f"查询成功。返回结果包含键: {results.keys() if results else 'None'}")
            return results
        except Exception as e:
            logger.error(f"查询 ChromaDB 集合 '{self.collection_name}' 时出错: {e}", exc_info=True)
            return None

    def get_all_metadata_values(self, metadata_field: str) -> List[str]:
        """
        获取集合中指定元数据字段的所有唯一值。
        注意：对于非常大的集合，此操作可能非常耗时和消耗内存。

        Args:
            metadata_field (str): 要获取唯一值的元数据字段名称 (例如 settings.META_FIELD_PARAM_TYPE)。

        Returns:
            List[str]: 包含唯一值的排序列表。
        """
        logger.info(f"\n--- 正在获取元数据字段 '{metadata_field}' 的唯一值 ---")
        if self.collection is None:
            logger.error("无法获取元数据：未连接到数据库集合。")
            return []
        try:
            logger.debug("正在从数据库获取所有元数据 (大集合可能耗时)...")
            # 只获取元数据。如果集合非常大，可能需要分页或限制数量。
            # 注意：ChromaDB 的 get() 可能有限制，或在非常大的集合上效率不高。
            # 考虑是否有更优化的方式获取唯一值，例如通过特定的数据库功能（如果支持）。
            results = self.collection.get(include=['metadatas'])
            logger.debug("元数据获取完成。")

            if results and results.get('metadatas'):
                unique_values = set()
                count = 0
                for meta in results['metadatas']:
                    if meta and metadata_field in meta:
                        value = meta[metadata_field]
                        # 过滤掉 None 或空字符串
                        if value is not None and value != '':
                            unique_values.add(value)
                    count += 1
                logger.debug(f"从 {count} 条记录中提取了字段 '{metadata_field}' 的值。")
                sorted_unique_values = sorted(list(unique_values))
                logger.info(f"找到 {len(sorted_unique_values)} 个唯一值。")
                return sorted_unique_values
            else:
                logger.warning("未在集合中找到任何元数据。")
                return []
        except Exception as e:
            logger.error(f"获取元数据字段 '{metadata_field}' 的值时出错: {e}", exc_info=True)
            return []

    def clear_collection(self) -> bool:
        """
        清空（删除并重建）当前集合。**警告：此操作不可逆！**
        (主要由知识库构建脚本在重建索引前调用)

        Returns:
            bool: 如果操作成功则返回 True，否则返回 False。
        """
        logger.warning(f"--- 准备清空集合: {self.collection_name} ---")
        if self.client is None:
            logger.error("无法清空集合：未连接到数据库客户端。")
            return False
        if not self.collection_name:
             logger.error("无法清空集合：集合名称无效。")
             return False

        try:
            logger.warning(f"正在删除集合 '{self.collection_name}'...")
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"集合 '{self.collection_name}' 已删除。")
            # 删除后需要重新获取或创建
            self.collection = None # 重置 collection 引用
            logger.info(f"正在重新创建集合 '{self.collection_name}'...")
            return self.connect() # 重新连接并创建
        except Exception as e:
            # 如果删除失败（例如集合不存在），尝试直接创建
            if "does not exist" in str(e).lower():
                 logger.warning(f"尝试删除不存在的集合 '{self.collection_name}'。将尝试直接创建。")
                 self.collection = None
                 return self.connect()
            else:
                 logger.error(f"清空集合 '{self.collection_name}' 时出错: {e}", exc_info=True)
                 self.collection = None # 出错时也重置
                 return False

# 注意：原 __main__ 部分已移除。
