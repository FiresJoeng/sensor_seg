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
    管理与 ChromaDB 向量数据库的交互。
    ** V3 Logic: 更新 build_index 以保存所有原始列对应的元数据字段。**
    """
    def __init__(self, db_path: str = config.CHROMA_DB_PATH,
                 collection_name: str = config.COLLECTION_NAME):
        """初始化 VectorStoreManager。"""
        self.db_path = db_path
        self.collection_name = collection_name
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[chromadb.Collection] = None

    def connect(self) -> bool:
        """连接到 ChromaDB 并获取/创建指定的集合。"""
        print(f"--- 正在连接到向量数据库 ---")
        print(f"数据库路径: {self.db_path}")
        print(f"集合名称: {self.collection_name}")
        if self.collection is not None: return True
        try:
            os.makedirs(self.db_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            print(f"成功连接/创建集合 '{self.collection_name}'。")
            print(f"当前集合包含 {self.collection.count()} 条数据。")
            return True
        except Exception as e:
            print(f"连接或获取 ChromaDB 集合时出错: {e}")
            self.client = None; self.collection = None
            return False

    def load_data_for_indexing(self, embeddings_path: str = config.EMBEDDINGS_PATH,
                               metadata_path: str = config.METADATA_PATH) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
        """从文件加载嵌入向量和元数据 (包含所有原始列)。"""
        print("\n--- 正在加载嵌入向量和元数据以供索引 ---")
        embeddings = None; df_metadata = None
        if not os.path.exists(embeddings_path): print(f"错误: 嵌入向量文件未找到于 '{embeddings_path}'"); return None, None
        try:
            embeddings = np.load(embeddings_path)
            print(f"已加载嵌入向量，形状: {embeddings.shape}")
        except Exception as e: print(f"加载嵌入向量 '{embeddings_path}' 时出错: {e}"); return None, None

        if not os.path.exists(metadata_path): print(f"错误: 元数据文件未找到于 '{metadata_path}'"); return None, None
        try:
            # 读取时确保所有列为字符串
            df_metadata = pd.read_csv(metadata_path, dtype=str).fillna('')
            print(f"已加载元数据，形状: {df_metadata.shape}")
            # 检查是否包含所有原始列
            missing_cols = [col for col in config.ALL_ORIGINAL_COLS if col not in df_metadata.columns]
            if missing_cols:
                print(f"错误: 元数据文件 '{metadata_path}' 中缺少以下列: {missing_cols}")
                print(f"请确保 embedding_generator.py 已正确运行并保存了元数据。")
                return None, None
        except Exception as e: print(f"加载元数据 '{metadata_path}' 时出错: {e}"); return None, None

        if embeddings is not None and df_metadata is not None:
            if len(embeddings) != len(df_metadata): print("错误: 嵌入向量数量与元数据行数不匹配！"); return None, None

        return embeddings, df_metadata

    def build_index(self, embeddings: np.ndarray, df_metadata: pd.DataFrame, batch_size: int = 1000) -> bool:
        """
        使用提供的嵌入向量和元数据填充 ChromaDB 集合。
        **V3: 保存所有原始列对应的元数据字段。**
        """
        print("\n--- 正在构建/填充向量索引 (V3 Logic) ---")
        if self.collection is None: print("错误: 未连接到数据库集合。"); return False
        if embeddings is None or df_metadata is None: print("错误: 缺少嵌入向量或元数据。"); return False
        if len(embeddings) != len(df_metadata): print("错误: 嵌入向量数量与元数据行数不匹配！"); return False

        target_count = len(df_metadata); current_count = self.collection.count()
        print(f"目标索引大小: {target_count} 条，当前集合大小: {current_count} 条。")

        print("准备数据以添加到 ChromaDB...")
        embeddings_list = embeddings.tolist()
        metadatas = []

        # 构建元数据字典，使用 config 中定义的映射关系
        try:
            for index, row in df_metadata.iterrows():
                metadata_item = {
                    # ChromaDB 字段名 : DataFrame 列名 (来自 config)
                    config.META_FIELD_COMPONENT: row[config.COL_COMPONENT_PART],
                    config.META_FIELD_PARAM_TYPE: row[config.COL_STANDARD_PARAM],
                    config.META_FIELD_ACTUAL_PARAM_DESC: row[config.COL_ACTUAL_PARAM_DESC],
                    config.META_FIELD_STANDARD_VALUE: row[config.COL_STANDARD_VALUE_DESC],
                    config.META_FIELD_ACTUAL_VALUE: row[config.COL_ACTUAL_VALUE_VARIATIONS],
                    config.META_FIELD_DEFAULT: row[config.COL_DEFAULT_VALUE],
                    config.META_FIELD_CODE: row[config.COL_STANDARD_CODE],
                    config.META_FIELD_FIELD_DESC: row[config.COL_FIELD_DESC],
                    config.META_FIELD_REMARK: row[config.COL_REMARK]
                }
                metadatas.append(metadata_item)
        except KeyError as e:
             print(f"错误: 准备元数据时找不到列 '{e}'。请检查元数据 CSV 文件 ('{config.METADATA_PATH}') 的列名是否与 config.py 中的 ALL_ORIGINAL_COLS 匹配。")
             return False
        except Exception as e:
             print(f"准备元数据时发生未知错误: {e}")
             return False


        ids = [str(i) for i in df_metadata.index]

        print(f"开始向集合 '{self.collection_name}' 添加 {len(ids)} 条数据...")
        start_time = time.time()
        added_count = 0
        try:
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_embeddings = embeddings_list[i : i + batch_size]
                batch_metadatas = metadatas[i : i + batch_size]
                if not batch_ids: continue
                self.collection.upsert(ids=batch_ids, embeddings=batch_embeddings, metadatas=batch_metadatas)
                added_count += len(batch_ids)
                print(f"  已添加 {added_count}/{len(ids)} 条数据...")

            end_time = time.time()
            print(f"数据添加/更新完成，耗时 {time.time() - start_time:.2f} 秒。")
            final_count = self.collection.count()
            print(f"添加后集合总数: {final_count}")
            if final_count < target_count: print(f"警告: 最终集合大小 ({final_count}) 小于目标大小 ({target_count})。")
            return True
        except Exception as e:
            print(f"向 ChromaDB 集合添加数据时出错: {e}")
            return False

    def query_collection(self, query_embeddings: List[List[float]],
                         n_results: int = config.DEFAULT_N_RESULTS,
                         where_filter: Optional[Dict[str, Any]] = None,
                         include_fields: List[str] = ['metadatas', 'distances']) -> Optional[Dict]:
        """在 ChromaDB 集合中执行向量相似度查询。"""
        if self.collection is None: print("错误: 未连接到数据库集合。"); return None
        if not query_embeddings: print("错误: 查询嵌入向量列表不能为空。"); return None
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where_filter,
                include=include_fields
            )
            return results
        except Exception as e:
            print(f"查询 ChromaDB 集合时出错: {e}")
            return None

    # get_all_metadata_values 和 clear_collection 方法保持不变
    # 在 VectorStoreManager 类中添加或确认此方法存在
    def get_all_metadata_values(self, metadata_field: str) -> List[str]:
        """
        获取集合中指定元数据字段的所有唯一值。
        注意：对于非常大的集合，此操作可能非常耗时和消耗内存。
        """
        print(f"\n--- 正在获取元数据字段 '{metadata_field}' 的唯一值 ---")
        if self.collection is None:
            print("错误: 未连接到数据库集合。")
            return []
        try:
            print("  正在从数据库获取所有元数据 (大集合可能耗时)...")
            # 只获取元数据，如果数据量巨大，考虑增加 limit 参数
            results = self.collection.get(include=['metadatas'])
            print("  元数据获取完成。")

            if results and results.get('metadatas'):
                unique_values = {
                    meta.get(metadata_field)
                    for meta in results['metadatas']
                    if meta and meta.get(metadata_field) is not None and meta.get(metadata_field) != '' # 过滤 None 和空字符串
                }
                print(f"找到 {len(unique_values)} 个唯一值。")
                return sorted(list(unique_values))
            else:
                print("  未找到元数据。")
                return []
        except Exception as e:
            print(f"获取元数据字段 '{metadata_field}' 的值时出错: {e}")
            return []

# --- 可选的直接执行入口 ---
if __name__ == "__main__":
    print("正在测试向量数据库管理器 (V3 Logic)...")
    manager = VectorStoreManager()
    if manager.connect():
        print("\n数据库连接测试成功。")
        count = manager.collection.count()
        print(f"集合 '{manager.collection_name}' 当前包含 {count} 条数据。")
    else:
        print("\n数据库连接测试失败。")

