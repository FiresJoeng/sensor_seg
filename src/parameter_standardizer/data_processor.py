# new_sensor_project/src/parameter_standardizer/data_processor.py
import logging
import re
import itertools
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np

# 导入项目配置
try:
    from ..config import settings
except ImportError:
    # 适应不同的导入上下文
    from config import settings

# 获取日志记录器实例
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    负责从 Excel 加载数据、进行预处理（合并、清理、拆分、展开）并返回处理后的 DataFrame。
    主要供知识库构建脚本 (如 scripts/build_kb.py) 使用。
    """
    def __init__(self,
                 excel_path: Optional[Path] = None,
                 sheet_names: Optional[List[str]] = None,
                 output_prepared_data_path: Optional[Path] = None,
                 output_metadata_path: Optional[Path] = None): # 添加 metadata path 参数
        """
        初始化 DataProcessor。

        Args:
            excel_path (Optional[Path]): 源 Excel 文件路径。如果为 None，从配置加载。
            sheet_names (Optional[List[str]]): 要处理的 Excel 工作表名称列表。如果为 None，从配置加载。
            output_prepared_data_path (Optional[Path]): 保存预处理数据的 CSV 路径 (用于嵌入)。如果为 None，从配置生成。
            output_metadata_path (Optional[Path]): 保存元数据的 CSV 路径 (用于索引)。如果为 None，从配置生成。
        """
        self.excel_path = excel_path or settings.KB_SOURCE_FILE
        # TODO: sheet_names is not defined in settings.py, using a default or requiring it
        self.sheet_names = sheet_names or ['变送器部分', '传感器部分', '保护管部分'] # 假设默认值
        # 定义输出路径，如果未提供
        self.output_prepared_data_path = output_prepared_data_path or (settings.OUTPUT_DIR / "prepared_data_for_kb.csv")
        self.output_metadata_path = output_metadata_path or (settings.OUTPUT_DIR / "metadata_for_kb.csv")

        # 定义需要进行拆分和笛卡尔积的列 (从 settings 读取)
        self.split_cols = [
            settings.SOURCE_COL_ACTUAL_PARAM_DESC,
            settings.SOURCE_COL_ACTUAL_VALUE_VARIATIONS
        ]
        # 定义所有需要从原始 Excel 保留到最终输出的列 (从 settings 读取)
        # 创建一个包含所有源列的列表，用于确保 DataFrame 包含所有必需列并按此顺序输出
        self.all_source_cols = [
            settings.SOURCE_COL_COMPONENT_PART,
            settings.SOURCE_COL_STANDARD_PARAM,
            settings.SOURCE_COL_ACTUAL_PARAM_DESC,
            settings.SOURCE_COL_STANDARD_VALUE_DESC,
            settings.SOURCE_COL_ACTUAL_VALUE_VARIATIONS,
            settings.SOURCE_COL_DEFAULT_VALUE,
            settings.SOURCE_COL_STANDARD_CODE,
            settings.SOURCE_COL_FIELD_DESC,
            settings.SOURCE_COL_REMARK
        ]
        # 定义用于拆分的分隔符 (仅分号)
        self.inner_delimiters_regex = r'\s*[;；]\s*'
        logger.info("DataProcessor 初始化完成。")
        logger.info(f"  源 Excel: {self.excel_path}")
        logger.info(f"  处理的工作表: {self.sheet_names}")
        logger.info(f"  预处理数据输出路径: {self.output_prepared_data_path}")
        logger.info(f"  元数据输出路径: {self.output_metadata_path}")
        logger.info(f"  拆分列: {self.split_cols}")

    def load_and_combine_sheets(self) -> Optional[pd.DataFrame]:
        """
        从 Excel 文件加载指定的工作表并将它们合并。
        确保包含所有在 self.all_source_cols 中定义的列。

        Returns:
            Optional[pd.DataFrame]: 合并后的 DataFrame，如果失败则返回 None。
        """
        logger.info(f"--- 正在加载并合并 Excel 工作表 ---")
        logger.info(f"文件: {self.excel_path}")
        if not self.excel_path.is_file():
            logger.error(f"Excel 文件未找到: {self.excel_path}")
            return None

        all_dfs: List[pd.DataFrame] = []
        try:
            excel_file = pd.ExcelFile(self.excel_path)
            available_sheets = excel_file.sheet_names
            sheets_to_process = [s for s in self.sheet_names if s in available_sheets]
            if not sheets_to_process:
                 logger.error(f"在 Excel 文件中未找到任何指定的工作表: {self.sheet_names}")
                 return None
            logger.info(f"将在以下工作表中查找数据: {sheets_to_process}")

            for sheet in sheets_to_process:
                logger.debug(f"  正在读取工作表: '{sheet}'...")
                try:
                    # 读取时确保所有列都作为字符串读取，避免类型推断问题
                    df_sheet = excel_file.parse(sheet, dtype=str)
                    # 填充 NaN 为空字符串
                    df_sheet.fillna('', inplace=True)

                    # 检查并添加缺失的必需列
                    missing_in_sheet = [col for col in self.all_source_cols if col not in df_sheet.columns]
                    if missing_in_sheet:
                        logger.warning(f"  工作表 '{sheet}' 缺少列: {missing_in_sheet}。这些列将填充为空字符串。")
                        for col in missing_in_sheet:
                            df_sheet[col] = '' # 添加空列

                    # 只选择在 all_source_cols 中定义的列，并按该顺序排列
                    # 这确保了所有 DataFrame 具有相同的结构以便合并
                    all_dfs.append(df_sheet[self.all_source_cols])
                    logger.debug(f"    '{sheet}' 读取成功，形状: {df_sheet.shape}")
                except Exception as sheet_error:
                    logger.warning(f"  无法读取工作表 '{sheet}'。错误: {sheet_error}。将跳过此表。", exc_info=True)

            if not all_dfs:
                logger.error("未能从 Excel 文件成功读取任何有效的工作表数据。")
                return None

            # 合并所有读取到的 DataFrame
            df_combined = pd.concat(all_dfs, ignore_index=True)
            logger.info(f"已合并来自 {len(all_dfs)} 个工作表的数据。合并后形状: {df_combined.shape}")
            # 再次确保所有列都存在且填充了 NaN
            df_combined = df_combined.fillna('')
            return df_combined
        except Exception as e:
            logger.error(f"读取或合并 Excel 工作表时出错: {e}", exc_info=True)
            return None

    def _split_and_clean(self, text: Optional[str]) -> List[str]:
        """
        辅助函数：按预定义的分隔符 (分号) 拆分文本并清理每个部分。
        空输入或仅包含空白的输入返回一个包含空字符串的列表 ['']。

        Args:
            text (Optional[str]): 要拆分的文本。

        Returns:
            List[str]: 清理后的文本片段列表。
        """
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            return [''] # 对于空或无效输入，返回包含空字符串的列表
        # 使用正则表达式拆分
        items = re.split(self.inner_delimiters_regex, text)
        # 清理每个部分（去除首尾空格），并过滤掉完全是空的部分
        cleaned_items = [item.strip() for item in items if item and item.strip()]
        # 如果清理后列表为空（例如原字符串是 "; ;"），则返回包含空字符串的列表
        return cleaned_items if cleaned_items else ['']

    def preprocess_and_expand(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        预处理 DataFrame：对指定列进行拆分和笛卡尔积展开。
        保持原始列名和顺序。

        Args:
            df (pd.DataFrame): 从 Excel 加载并合并的 DataFrame。

        Returns:
            Optional[pd.DataFrame]: 经过拆分和展开处理后的 DataFrame，如果失败则返回 None。
        """
        logger.info("\n--- 正在进行数据预处理 (拆分与笛卡尔积展开) ---")
        if df is None:
            logger.error("输入的 DataFrame 为空，无法进行预处理。")
            return None

        # 复制 DataFrame 以避免修改原始数据
        df_prepared = df.copy()

        # 检查拆分列是否存在
        for col in self.split_cols:
            if col not in df_prepared.columns:
                logger.error(f"DataFrame 中缺少用于拆分的列 '{col}'。请检查配置和 Excel 文件。")
                return None

        # 1. 拆分需要展开的列，并将结果存储在临时列中
        temp_split_col_names = []
        try:
            for i, col_name in enumerate(self.split_cols):
                temp_col_name = f"_temp_split_{i}"
                logger.debug(f"使用分号拆分列 '{col_name}' -> '{temp_col_name}'...")
                df_prepared[temp_col_name] = df_prepared[col_name].apply(self._split_and_clean)
                temp_split_col_names.append(temp_col_name)
        except Exception as e:
            logger.error(f"拆分列时出错: {e}", exc_info=True)
            return None

        # 2. 计算笛卡尔积并构建新数据
        logger.info("计算笛卡尔积并展开数据...")
        expanded_data: List[Dict[str, Any]] = []
        total_rows = len(df_prepared)
        processed_rows = 0

        for index, row in df_prepared.iterrows():
            # 获取当前行的所有原始列的值
            original_row_data = row[self.all_source_cols].to_dict()

            # 获取已拆分的列表
            split_lists = [row[temp_col] for temp_col in temp_split_col_names]

            # 计算这些列表的笛卡尔积
            # 例如: product(['A', 'B'], ['1', '2']) -> [('A', '1'), ('A', '2'), ('B', '1'), ('B', '2')]
            cartesian_product = list(itertools.product(*split_lists))

            # 为笛卡尔积中的每个组合创建新记录
            for combination in cartesian_product:
                new_record = original_row_data.copy() # 复制原始行数据
                # 用组合中的值更新对应的原始列
                for i, col_name in enumerate(self.split_cols):
                    new_record[col_name] = combination[i]
                expanded_data.append(new_record)

            processed_rows += 1
            if processed_rows % 100 == 0 or processed_rows == total_rows:
                 logger.debug(f"  已处理 {processed_rows}/{total_rows} 行...")

        # 3. 将展开后的数据列表转换为 DataFrame
        # 指定列顺序与原始 Excel 一致 (使用 self.all_source_cols)
        if not expanded_data:
             logger.warning("经过拆分和笛卡尔积处理后，没有生成任何数据。")
             # 返回一个具有正确列的空 DataFrame
             df_final = pd.DataFrame(columns=self.all_source_cols)
        else:
             df_final = pd.DataFrame(expanded_data, columns=self.all_source_cols)

        # 4. 重置索引
        df_final.reset_index(drop=True, inplace=True)
        logger.info(f"预处理完成。最终 DataFrame 形状: {df_final.shape}")
        logger.debug(f"最终列名: {df_final.columns.tolist()}")

        return df_final

    def save_dataframe(self, df: pd.DataFrame, output_path: Path) -> bool:
        """
        将 DataFrame 保存到指定的 CSV 文件路径。

        Args:
            df (pd.DataFrame): 要保存的 DataFrame。
            output_path (Path): 输出 CSV 文件的路径。

        Returns:
            bool: 如果保存成功则返回 True，否则返回 False。
        """
        if df is None:
            logger.error(f"没有数据可保存到 {output_path.name}。")
            # 决定是否应该创建一个空的带表头的文件
            try:
                 logger.warning(f"输入 DataFrame 为 None，将尝试创建空的带表头文件: {output_path}")
                 output_path.parent.mkdir(parents=True, exist_ok=True)
                 pd.DataFrame(columns=self.all_source_cols).to_csv(
                      output_path, index=False, encoding='utf-8-sig'
                 )
                 return True # 认为创建空文件是成功的
            except Exception as e:
                 logger.error(f"创建空的带表头文件 {output_path.name} 时失败: {e}", exc_info=True)
                 return False

        logger.info(f"--- 正在保存 DataFrame 到 {output_path.name} ---")
        try:
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # 保存时确保列顺序与 all_source_cols 一致
            df.to_csv(output_path, index=False, encoding='utf-8-sig', columns=self.all_source_cols)
            logger.info(f"DataFrame ({df.shape}) 成功保存到: {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存 DataFrame 到 '{output_path.name}' 时出错: {e}", exc_info=True)
            return False

    def run_pipeline(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        执行完整的数据处理流水线：加载 -> 处理 -> 保存。

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
                返回两个 DataFrame：
                1. 第一个 DataFrame 是经过预处理和展开后的数据，用于后续生成嵌入。
                2. 第二个 DataFrame 也是同一个预处理和展开后的数据，用于后续构建元数据索引。
                如果过程中出错，则对应项为 None。
        """
        # 1. 加载和合并数据
        df_combined = self.load_and_combine_sheets()
        if df_combined is None:
            return None, None # 加载失败

        # 2. 预处理和展开数据
        df_processed = self.preprocess_and_expand(df_combined)
        if df_processed is None:
            logger.warning("预处理步骤未生成有效 DataFrame 或返回 None。")
            # 即使处理失败，也可能需要返回一个空的 DataFrame 结构
            df_processed = pd.DataFrame(columns=self.all_source_cols)

        # 3. 保存处理后的数据 (可选，但对调试有用)
        # 保存用于嵌入的数据（与元数据相同）
        save_prepared_ok = self.save_dataframe(df_processed, self.output_prepared_data_path)
        # 保存用于元数据的数据（与准备好的数据相同）
        save_metadata_ok = self.save_dataframe(df_processed, self.output_metadata_path)

        if not save_prepared_ok or not save_metadata_ok:
            logger.warning("数据处理完成，但未能成功保存所有输出文件。")
            # 即使保存失败，仍然返回内存中的 DataFrame

        # 返回处理后的 DataFrame 两次，因为它同时用于嵌入和元数据
        return df_processed, df_processed

# 注意：原 __main__ 部分已移除。
