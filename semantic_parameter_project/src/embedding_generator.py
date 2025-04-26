# src/embedding_generator.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import os
from typing import Optional

# 从 config 模块导入配置
try:
    from . import config
except ImportError:
    import config

class EmbeddingGenerator:
    """
    负责加载嵌入模型、为处理后的数据生成文本嵌入，并保存嵌入向量和元数据。
    ** V4 Logic: 将 '实际参数' 和 '实际参数值' 组合后进行嵌入。**
    """
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME,
                 input_csv_path: str = config.PREPROCESSED_DATA_PATH,
                 embeddings_output_path: str = config.EMBEDDINGS_PATH,
                 metadata_output_path: str = config.METADATA_PATH):
        """
        初始化 EmbeddingGenerator。
        """
        self.model_name = model_name
        self.input_csv_path = input_csv_path
        self.embeddings_output_path = embeddings_output_path
        self.metadata_output_path = metadata_output_path
        self.model: Optional[SentenceTransformer] = None
        # 需要从 CSV 读取的列 (所有原始列)
        self.required_input_cols = config.ALL_ORIGINAL_COLS
        # 需要保存到元数据文件的列 (所有原始列)
        self.metadata_cols_to_save = config.ALL_ORIGINAL_COLS

    def load_model(self) -> bool:
        """加载 Sentence Transformer 模型。"""
        print(f"--- 正在加载嵌入模型 ---")
        print(f"模型: {self.model_name}")
        if self.model is not None: return True
        start_time = time.time()
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"模型加载成功，耗时 {time.time() - start_time:.2f} 秒。")
            return True
        except Exception as e:
            print(f"加载模型 '{self.model_name}' 时出错: {e}")
            self.model = None
            return False

    def load_data(self) -> Optional[pd.DataFrame]:
        """加载预处理后的数据 CSV 文件 (包含原始列名)。"""
        print(f"\n--- 正在加载预处理数据 ---")
        if not os.path.exists(self.input_csv_path):
            print(f"错误: 未找到预处理数据文件 '{self.input_csv_path}'")
            return None
        try:
            df = pd.read_csv(self.input_csv_path, dtype=str).fillna('')
            print(f"已加载数据 '{self.input_csv_path}'，形状: {df.shape}")
            missing_cols = [col for col in self.required_input_cols if col not in df.columns]
            if missing_cols:
                 print(f"错误: 数据文件中缺少以下必需列: {missing_cols}")
                 print(f"请确保已使用最新的 data_processor.py (V6) 重新生成了 {self.input_csv_path}")
                 return None
            return df
        except Exception as e:
            print(f"加载数据 '{self.input_csv_path}' 时出错: {e}")
            return None

    def generate_embeddings(self, df: pd.DataFrame, batch_size: int = 64) -> Optional[np.ndarray]:
        """
        为 DataFrame 中的文本生成嵌入向量。
        **V4: 组合 '实际参数' (COL_ACTUAL_PARAM_DESC) 和 '实际参数值' (COL_ACTUAL_VALUE_VARIATIONS) 进行嵌入。**
        """
        print(f"\n--- 正在生成嵌入向量 (V4 Logic: 嵌入参数描述+值) ---")
        if self.model is None: print("错误: 嵌入模型未加载。"); return None
        if df is None or df.empty: print("错误: 输入的 DataFrame 为空。"); return None

        # 获取实际参数描述和实际参数值列名 (从 config 读取)
        param_desc_col = config.COL_ACTUAL_PARAM_DESC
        value_variation_col = config.COL_ACTUAL_VALUE_VARIATIONS

        # 检查所需列是否存在
        if param_desc_col not in df.columns or value_variation_col not in df.columns:
            print(f"错误: DataFrame 中缺少列 '{param_desc_col}' 或 '{value_variation_col}' 用于生成嵌入。")
            return None

        print(f"准备文本进行嵌入 (格式: '{param_desc_col}: {value_variation_col}')...")
        # 构造用于嵌入的文本列表
        try:
            texts_to_embed = [
                # 使用 f-string 格式化组合文本，处理可能的空字符串
                f"{row[param_desc_col]}: {row[value_variation_col]}"
                for index, row in df.iterrows()
            ]
        except KeyError as e:
             print(f"错误: 构造嵌入文本时找不到列: {e}。")
             return None
        except Exception as e:
             print(f"构造嵌入文本时发生未知错误: {e}")
             return None


        print(f"共准备了 {len(texts_to_embed)} 条组合文本。")

        print(f"开始生成嵌入向量 (批大小: {batch_size})...")
        start_time = time.time()
        try:
            embeddings = self.model.encode(texts_to_embed, show_progress_bar=True, batch_size=batch_size)
            print(f"嵌入向量生成完毕。形状: {embeddings.shape}。耗时: {time.time() - start_time:.2f} 秒。")
            return embeddings
        except Exception as e:
            print(f"生成嵌入向量时发生错误: {e}")
            return None

    def save_results(self, embeddings: np.ndarray, df_metadata_source: pd.DataFrame) -> bool:
        """
        保存生成的嵌入向量和对应的元数据 (包含所有原始列)。
        """
        print("\n--- 正在保存嵌入向量和元数据 ---")
        if embeddings is None or df_metadata_source is None: print("错误: 缺少嵌入向量或元数据 DataFrame。"); return False
        if len(embeddings) != len(df_metadata_source): print("错误: 嵌入向量数量与元数据行数不匹配！"); return False

        os.makedirs(os.path.dirname(self.embeddings_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_output_path), exist_ok=True)

        embeddings_saved = False
        try:
            np.save(self.embeddings_output_path, embeddings)
            print(f"嵌入向量成功保存到: {self.embeddings_output_path}")
            embeddings_saved = True
        except Exception as e:
            print(f"保存嵌入向量到 '{self.embeddings_output_path}' 时出错: {e}")

        metadata_saved = False
        try:
            # 检查元数据 DataFrame 是否包含所有需要的列
            missing_meta_cols = [col for col in self.metadata_cols_to_save if col not in df_metadata_source.columns]
            if missing_meta_cols:
                print(f"错误: 元数据 DataFrame 中缺少以下列，无法保存: {missing_meta_cols}")
                return False
            # 直接使用原始 DataFrame 保存，按配置中的顺序保存列
            metadata_df_to_save = df_metadata_source[self.metadata_cols_to_save].copy()
            metadata_df_to_save.to_csv(self.metadata_output_path, index=False, encoding='utf-8-sig')
            print(f"元数据成功保存到: {self.metadata_output_path}")
            metadata_saved = True
        except Exception as e:
            print(f"保存元数据到 '{self.metadata_output_path}' 时出错: {e}")

        return embeddings_saved and metadata_saved

    def run_pipeline(self) -> bool:
        """执行完整的嵌入生成流水线。"""
        if not self.load_model(): return False
        df = self.load_data()
        if df is None: return False
        embeddings = self.generate_embeddings(df)
        if embeddings is None: return False
        if self.save_results(embeddings, df):
             return True
        else:
             print("警告: 嵌入生成完成，但结果未能完全保存。")
             return False

# --- 可选的直接执行入口 ---
if __name__ == "__main__":
    print("正在执行嵌入向量生成流程 (V4 Logic: 嵌入参数描述+值)...")
    generator = EmbeddingGenerator()
    success = generator.run_pipeline()
    if success: print("\n嵌入向量生成流程成功完成。")
    else: print("\n嵌入向量生成流程失败或未完全成功。")

