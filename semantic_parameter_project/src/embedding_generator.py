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
    """
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME,
                 input_csv_path: str = config.PREPROCESSED_DATA_PATH,
                 embeddings_output_path: str = config.EMBEDDINGS_PATH,
                 metadata_output_path: str = config.METADATA_PATH):
        """
        初始化 EmbeddingGenerator。

        Args:
            model_name (str): 要使用的 Sentence Transformer 模型名称。
            input_csv_path (str): 预处理后的数据 CSV 文件路径。
            embeddings_output_path (str): 保存嵌入向量的 .npy 文件路径。
            metadata_output_path (str): 保存对应元数据的 .csv 文件路径。
        """
        self.model_name = model_name
        self.input_csv_path = input_csv_path
        self.embeddings_output_path = embeddings_output_path
        self.metadata_output_path = metadata_output_path
        self.model: Optional[SentenceTransformer] = None # 用于存储加载后的模型对象
        # 定义从输入 CSV 读取以及需要保存到元数据文件的列
        self.required_input_cols = [
            config.PREPROCESSED_COL_ACTUAL_VARIATION, config.COL_STANDARD_PARAM,
            config.COL_STANDARD_VALUE_DESC, config.COL_STANDARD_CODE, config.COL_COMPONENT_PART
        ]
        self.metadata_cols_to_save = [
            config.COL_STANDARD_VALUE_DESC, config.COL_STANDARD_CODE,
            config.PREPROCESSED_COL_ACTUAL_VARIATION, config.COL_STANDARD_PARAM,
            config.COL_COMPONENT_PART
        ]

    def load_model(self) -> bool:
        """
        加载 Sentence Transformer 模型。

        Returns:
            bool: 如果模型加载成功返回 True，否则返回 False。
        """
        print(f"--- 正在加载嵌入模型 ---")
        print(f"模型: {self.model_name}")
        # 如果模型已经加载，则直接返回 True
        if self.model is not None:
             print("模型已加载。")
             return True
        start_time = time.time()
        try:
            # 初始化 SentenceTransformer 模型
            # device 参数可以指定 'cuda' (如果 GPU 可用) 或 'cpu'
            self.model = SentenceTransformer(self.model_name) #, device='cuda'
            end_time = time.time()
            print(f"模型加载成功，耗时 {end_time - start_time:.2f} 秒。")
            return True
        except Exception as e:
            # 捕获加载模型时可能发生的错误
            print(f"加载模型 '{self.model_name}' 时出错: {e}")
            self.model = None # 加载失败，将模型对象设为 None
            return False

    def load_data(self) -> Optional[pd.DataFrame]:
        """
        加载预处理后的数据 CSV 文件。

        Returns:
            Optional[pd.DataFrame]: 加载的 DataFrame，如果出错则返回 None。
        """
        print(f"\n--- 正在加载预处理数据 ---")
        # 检查输入文件是否存在
        if not os.path.exists(self.input_csv_path):
            print(f"错误: 未找到预处理数据文件 '{self.input_csv_path}'")
            print("请先运行数据处理脚本生成该文件，或检查路径是否正确。")
            return None
        try:
            # 读取 CSV 文件，指定 dtype=str 并填充 NaN
            df = pd.read_csv(self.input_csv_path, dtype=str).fillna('')
            print(f"已加载数据 '{self.input_csv_path}'，形状: {df.shape}")

            # 检查所需的输入列是否存在
            missing_cols = [col for col in self.required_input_cols if col not in df.columns]
            if missing_cols:
                 print(f"错误: 数据文件中缺少以下必需列: {missing_cols}")
                 return None
            return df
        except Exception as e:
            # 捕获读取 CSV 时可能发生的错误
            print(f"加载数据 '{self.input_csv_path}' 时出错: {e}")
            return None

    def generate_embeddings(self, df: pd.DataFrame, batch_size: int = 64) -> Optional[np.ndarray]:
        """
        为 DataFrame 中的文本生成嵌入向量。

        Args:
            df (pd.DataFrame): 包含待嵌入文本的 DataFrame。
            batch_size (int): 嵌入生成的批处理大小，影响速度和内存使用。

        Returns:
            Optional[np.ndarray]: 生成的嵌入向量 NumPy 数组，如果出错则返回 None。
        """
        print(f"\n--- 正在生成嵌入向量 ---")
        # 检查模型是否已加载
        if self.model is None:
            print("错误: 嵌入模型未加载。请先调用 load_model()。")
            return None
        # 检查输入 DataFrame 是否有效
        if df is None or df.empty:
            print("错误: 输入的 DataFrame 为空，无法生成嵌入。")
            return None

        # 准备待嵌入的文本列表
        # 结合上下文信息（参数类型）和实际参数值变体，生成更有意义的嵌入
        text_col = config.PREPROCESSED_COL_ACTUAL_VARIATION
        context_col = config.COL_STANDARD_PARAM
        print(f"准备文本进行嵌入 (格式: '{context_col}: {text_col}')...")
        # 列表推导式，高效构建文本列表
        texts_to_embed = [
            f"{row[context_col]}: {row[text_col]}"
            for index, row in df.iterrows()
        ]
        print(f"共准备了 {len(texts_to_embed)} 条文本。")

        print(f"开始生成嵌入向量 (批大小: {batch_size})... 这可能需要一些时间。")
        start_time = time.time()
        try:
            # 调用模型的 encode 方法生成嵌入向量
            # show_progress_bar=True 会显示进度条
            embeddings = self.model.encode(
                texts_to_embed,
                show_progress_bar=True,
                batch_size=batch_size
            )
            end_time = time.time()
            print(f"嵌入向量生成完毕。形状: {embeddings.shape}。耗时: {end_time - start_time:.2f} 秒。")
            return embeddings
        except Exception as e:
            # 捕获嵌入生成过程中可能发生的错误
            print(f"生成嵌入向量时发生错误: {e}")
            return None

    def save_results(self, embeddings: np.ndarray, df_metadata_source: pd.DataFrame) -> bool:
        """
        保存生成的嵌入向量和对应的元数据。

        Args:
            embeddings (np.ndarray): 要保存的嵌入向量数组。
            df_metadata_source (pd.DataFrame): 包含元数据源的 DataFrame (应与嵌入一一对应)。

        Returns:
            bool: 如果嵌入和元数据都保存成功返回 True，否则返回 False。
        """
        print("\n--- 正在保存嵌入向量和元数据 ---")
        # 检查输入是否有效
        if embeddings is None or df_metadata_source is None:
            print("错误: 缺少嵌入向量或元数据 DataFrame，无法保存。")
            return False
        # 检查嵌入向量数量和元数据行数是否一致
        if len(embeddings) != len(df_metadata_source):
            print("错误: 嵌入向量数量与元数据行数不匹配！")
            print(f"  嵌入数量: {len(embeddings)}, 元数据行数: {len(df_metadata_source)}")
            return False

        # 创建嵌入向量和元数据文件的输出目录 (如果不存在)
        os.makedirs(os.path.dirname(self.embeddings_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_output_path), exist_ok=True)

        # 保存嵌入向量为 .npy 文件
        embeddings_saved = False
        try:
            np.save(self.embeddings_output_path, embeddings)
            print(f"嵌入向量成功保存到: {self.embeddings_output_path}")
            embeddings_saved = True
        except Exception as e:
            print(f"保存嵌入向量到 '{self.embeddings_output_path}' 时出错: {e}")

        # 保存元数据为 .csv 文件
        metadata_saved = False
        try:
            # 从源 DataFrame 中选择需要保存的元数据列
            metadata_df_to_save = df_metadata_source[self.metadata_cols_to_save].copy()
            # 保存为 CSV，使用 utf-8-sig 编码
            metadata_df_to_save.to_csv(self.metadata_output_path, index=False, encoding='utf-8-sig')
            print(f"元数据成功保存到: {self.metadata_output_path}")
            metadata_saved = True
        except KeyError as e:
             # 如果配置的列名在 DataFrame 中找不到，捕获 KeyError
             print(f"错误: 保存元数据时找不到列: {e}。请检查 config.py 中的列名配置。")
        except Exception as e:
            # 捕获其他保存元数据时的错误
            print(f"保存元数据到 '{self.metadata_output_path}' 时出错: {e}")

        # 只有当嵌入和元数据都成功保存时，才返回 True
        return embeddings_saved and metadata_saved

    def run_pipeline(self) -> bool:
        """
        执行完整的嵌入生成流水线：加载模型 -> 加载数据 -> 生成嵌入 -> 保存结果。

        Returns:
            bool: 如果整个流程成功完成返回 True，否则返回 False。
        """
        # 依次执行各个步骤，并在失败时提前返回 False
        if not self.load_model():
            return False
        df = self.load_data()
        if df is None:
            return False
        embeddings = self.generate_embeddings(df)
        if embeddings is None:
            return False
        if self.save_results(embeddings, df):
             return True # 所有步骤成功
        else:
             # 保存失败，打印警告并返回 False
             print("警告: 嵌入生成完成，但结果未能完全保存。")
             return False

# --- 可选的直接执行入口 ---
# 当直接运行 embedding_generator.py 时，执行嵌入生成流程
if __name__ == "__main__":
    print("正在执行嵌入向量生成流程...")
    generator = EmbeddingGenerator()
    success = generator.run_pipeline()

    if success:
        print("\n嵌入向量生成流程成功完成。")
    else:
        print("\n嵌入向量生成流程失败或未完全成功。")

