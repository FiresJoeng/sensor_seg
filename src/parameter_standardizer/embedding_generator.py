# new_sensor_project/src/parameter_standardizer/embedding_generator.py
import logging
import time
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# 导入项目配置
try:
    from config import settings
except ImportError:
    # 适应不同的导入上下文
    from config import settings

# 获取日志记录器实例
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    负责加载嵌入模型、为处理后的数据生成文本嵌入。
    主要供知识库构建脚本 (如 scripts/build_kb.py) 使用。
    """
    def __init__(self, model_name: Optional[str] = None):
        """
        初始化 EmbeddingGenerator。

        Args:
            model_name (Optional[str]): 要加载的 Sentence Transformer 模型名称。
                                        如果为 None，则从 config/settings.py 加载。
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.model: Optional[SentenceTransformer] = None
        logger.info(f"EmbeddingGenerator 初始化。模型: {self.model_name}")

    def load_model(self) -> bool:
        """
        加载 Sentence Transformer 模型。

        Returns:
            bool: 如果模型加载成功则返回 True，否则返回 False。
        """
        if self.model is not None:
            logger.debug("嵌入模型已加载。")
            return True

        logger.info(f"--- 正在加载嵌入模型: {self.model_name} ---")
        start_time = time.time()
        try:
            # 可以在这里添加设备选择逻辑 (例如 'cuda' 或 'cpu')
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # self.model = SentenceTransformer(self.model_name, device=device)
            self.model = SentenceTransformer(self.model_name)
            end_time = time.time()
            logger.info(f"模型加载成功。耗时: {end_time - start_time:.2f} 秒。")
            return True
        except Exception as e:
            logger.error(f"加载嵌入模型 '{self.model_name}' 时出错: {e}", exc_info=True)
            self.model = None
            return False

    def encode(self, texts: List[str], batch_size: int = 64, show_progress_bar: bool = True) -> Optional[np.ndarray]:
        """
        使用加载的模型为文本列表生成嵌入向量。

        Args:
            texts (List[str]): 需要编码的文本字符串列表。
            batch_size (int): 编码时使用的批处理大小。
            show_progress_bar (bool): 是否显示编码进度条。

        Returns:
            Optional[np.ndarray]: 包含嵌入向量的 NumPy 数组，如果失败则返回 None。
        """
        if self.model is None:
            logger.error("无法生成嵌入：模型未加载。请先调用 load_model()。")
            return None
        if not texts:
            logger.warning("输入文本列表为空，无需生成嵌入。")
            return np.array([]) # 返回空数组

        logger.info(f"开始为 {len(texts)} 条文本生成嵌入向量 (批大小: {batch_size})...")
        start_time = time.time()
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                # normalize_embeddings=True # 根据模型和下游任务决定是否需要归一化
            )
            end_time = time.time()
            logger.info(f"嵌入向量生成完毕。形状: {embeddings.shape}。耗时: {end_time - start_time:.2f} 秒。")
            return embeddings
        except Exception as e:
            logger.error(f"生成嵌入向量时发生错误: {e}", exc_info=True)
            return None

    def generate_embeddings_from_df(
        self,
        df: pd.DataFrame,
        text_col1: str = settings.SOURCE_COL_ACTUAL_PARAM_DESC,
        text_col2: str = settings.SOURCE_COL_ACTUAL_VALUE_VARIATIONS,
        batch_size: int = 64
    ) -> Optional[np.ndarray]:
        """
        从 DataFrame 的指定列组合文本并生成嵌入向量。
        这是知识库构建流程中常用的方法。

        Args:
            df (pd.DataFrame): 包含源数据的 DataFrame。
            text_col1 (str): 第一个文本列名 (例如实际参数描述)。
            text_col2 (str): 第二个文本列名 (例如实际参数值)。
            batch_size (int): 编码时使用的批处理大小。

        Returns:
            Optional[np.ndarray]: 包含嵌入向量的 NumPy 数组，如果失败则返回 None。
        """
        logger.info(f"--- 准备从 DataFrame 生成嵌入 (组合列: '{text_col1}', '{text_col2}') ---")
        if df is None or df.empty:
            logger.error("输入的 DataFrame 为空，无法生成嵌入。")
            return None

        # 检查所需列是否存在
        if text_col1 not in df.columns or text_col2 not in df.columns:
            logger.error(f"DataFrame 中缺少必需的列: '{text_col1}' 或 '{text_col2}'。")
            return None

        # 构造用于嵌入的文本列表
        logger.debug("正在构造用于嵌入的文本列表...")
        try:
            # 确保列是字符串类型并填充 NaN
            df[text_col1] = df[text_col1].astype(str).fillna('')
            df[text_col2] = df[text_col2].astype(str).fillna('')
            # 组合文本
            texts_to_embed = [
                f"{row[text_col1]}: {row[text_col2]}"
                for _, row in df.iterrows()
            ]
            logger.debug(f"成功构造了 {len(texts_to_embed)} 条组合文本。")
        except KeyError as e:
             logger.error(f"构造嵌入文本时找不到列: {e}。请检查 DataFrame 列名和配置。")
             return None
        except Exception as e:
             logger.error(f"构造嵌入文本时发生未知错误: {e}", exc_info=True)
             return None

        # 调用核心编码方法
        return self.encode(texts_to_embed, batch_size=batch_size)

# 注意：原 __main__ 和 run_pipeline 部分已移除。
# load_data 和 save_results 的逻辑现在应该由调用者（例如 build_kb.py 脚本）处理，
# 因为它们涉及到具体的文件路径和数据处理流程，不属于 EmbeddingGenerator 的核心职责。
# EmbeddingGenerator 的核心职责是加载模型和执行编码。
