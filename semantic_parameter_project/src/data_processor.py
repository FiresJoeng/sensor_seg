# src/data_processor.py
import pandas as pd
import numpy as np
import os
import re # 导入正则表达式模块
import itertools # 导入 itertools 用于笛卡尔积
from typing import List, Optional, Tuple

# 从 config 模块导入配置
try:
    from . import config
except ImportError:
    import config

class DataProcessor:
    """
    负责从 Excel 加载数据、进行预处理（合并、清理、拆分、展开）并保存。
    ** V6 Logic: 实现笛卡尔积处理，仅使用分号拆分，不移除空行，
           并保持输出文件的列名与原始 Excel 一致。**
    """
    def __init__(self, excel_path: str = config.EXCEL_FILE_PATH,
                 sheet_names: List[str] = config.SHEET_NAMES,
                 output_csv_path: str = config.PREPROCESSED_DATA_PATH):
        """
        初始化 DataProcessor。
        """
        self.excel_path = excel_path
        self.sheet_names = sheet_names
        self.output_csv_path = output_csv_path
        # 定义需要进行拆分和笛卡尔积的列 (从 config 读取)
        self.split_cols = [
            config.COL_ACTUAL_PARAM_DESC,
            config.COL_ACTUAL_VALUE_VARIATIONS
        ]
        # 定义所有需要从原始 Excel 保留到最终输出的列 (从 config 读取)
        self.all_original_cols = config.ALL_ORIGINAL_COLS
        # 定义用于拆分的分隔符 (仅分号)
        self.inner_delimiters_regex = r'\s*[;；]\s*'

    def load_and_combine_sheets(self) -> Optional[pd.DataFrame]:
        """
        从 Excel 文件加载指定的工作表并将它们合并。
        """
        print(f"--- 正在加载并合并 Excel 工作表 ---")
        print(f"文件: {self.excel_path}")
        if not os.path.exists(self.excel_path):
            print(f"错误: Excel 文件未找到于 '{self.excel_path}'")
            return None
        all_dfs = []
        try:
            for sheet in self.sheet_names:
                print(f"  正在读取工作表: '{sheet}'...")
                try:
                    # 读取时确保所有列都作为字符串读取
                    df_sheet = pd.read_excel(self.excel_path, sheet_name=sheet, dtype=str)
                    # 检查并处理缺失的列
                    missing_in_sheet = [col for col in self.all_original_cols if col not in df_sheet.columns]
                    if missing_in_sheet:
                        print(f"  警告: 工作表 '{sheet}' 缺少列: {missing_in_sheet}。这些列将填充为空字符串。")
                        for col in missing_in_sheet:
                            df_sheet[col] = '' # 添加空列
                    # 只选择在 all_original_cols 中定义的列，并按该顺序排列
                    all_dfs.append(df_sheet[self.all_original_cols])
                    print(f"    '{sheet}' 读取成功，形状: {df_sheet.shape}")
                except Exception as sheet_error:
                    print(f"  警告: 无法读取工作表 '{sheet}'。错误: {sheet_error}。将跳过此表。")

            if not all_dfs:
                print("错误: 未能从 Excel 文件成功读取任何工作表。")
                return None

            df_combined = pd.concat(all_dfs, ignore_index=True)
            print(f"\n已合并来自 {len(all_dfs)} 个工作表的数据。合并后形状: {df_combined.shape}")
            return df_combined
        except Exception as e:
            print(f"读取或合并 Excel 工作表时出错: {e}")
            return None

    def _split_and_clean(self, text: Optional[str]) -> List[str]:
        """
        辅助函数：按分号拆分并清理。空输入返回 ['']。
        """
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            return ['']
        items = re.split(self.inner_delimiters_regex, text)
        cleaned_items = [item.strip() for item in items if item and item.strip()]
        return cleaned_items if cleaned_items else ['']

    def preprocess(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        预处理 DataFrame：进行笛卡尔积展开，保持原始列名。
        """
        print("\n--- 正在进行数据预处理 (V6 Logic: 保留原始表头) ---")
        if df is None: print("错误: 输入的 DataFrame 为空。"); return None

        # 1. 填充 NaN 为空字符串
        df_prepared = df.fillna('').copy()

        # 2. 拆分需要展开的列
        split_col_param = config.COL_ACTUAL_PARAM_DESC
        split_col_value = config.COL_ACTUAL_VALUE_VARIATIONS
        print(f"使用分号拆分列 '{split_col_param}' 和 '{split_col_value}'...")
        try:
            df_prepared['split_param_desc_list'] = df_prepared[split_col_param].apply(self._split_and_clean)
            df_prepared['split_value_list'] = df_prepared[split_col_value].apply(self._split_and_clean)
        except KeyError as e:
            print(f"错误: 找不到列 '{e}' 进行拆分。请检查 config.py 中的 COL_ACTUAL_PARAM_DESC 和 COL_ACTUAL_VALUE_VARIATIONS 是否与 Excel 列名匹配。")
            return None

        # 3. 计算笛卡尔积并构建新数据
        print("计算笛卡尔积并展开...")
        expanded_data = []
        for index, row in df_prepared.iterrows():
            # 获取当前行的所有原始列的值 (除了我们拆分的那两列)
            # 使用 all_original_cols 来确定要保留的列
            other_cols_data = row[config.ALL_ORIGINAL_COLS].drop(self.split_cols).to_dict()

            param_list = row['split_param_desc_list']
            value_list = row['split_value_list']

            cartesian_product = list(itertools.product(param_list, value_list))

            for param_variation, value_variation in cartesian_product:
                new_record = other_cols_data.copy() # 复制其他列的数据
                # 将单一变体填充回原始列名
                new_record[split_col_param] = param_variation
                new_record[split_col_value] = value_variation
                expanded_data.append(new_record)

        # 将展开后的数据列表转换为 DataFrame
        # 指定列顺序与原始 Excel 一致 (使用 config.ALL_ORIGINAL_COLS)
        df_final = pd.DataFrame(expanded_data, columns=config.ALL_ORIGINAL_COLS)

        if df_final.empty:
            print("警告：经过拆分和笛卡尔积处理后，没有生成任何数据。")
            return df_final # 返回空 DataFrame

        # 4. 重置索引
        df_final.reset_index(drop=True, inplace=True)
        print(f"\n预处理完成。最终 DataFrame 形状: {df_final.shape}")
        print(f"最终列名: {df_final.columns.tolist()}") # 确认列名和顺序

        return df_final

    def save_data(self, df: pd.DataFrame) -> bool:
        """
        将处理后的 DataFrame 保存到 CSV 文件。
        """
        if df is None: print("错误: 没有数据可保存。"); return False
        print(f"\n--- 正在保存处理后的数据 ---")
        try:
            output_dir = os.path.dirname(self.output_csv_path)
            os.makedirs(output_dir, exist_ok=True)
            # 保存时确保列顺序与 all_original_cols 一致
            df.to_csv(self.output_csv_path, index=False, encoding='utf-8-sig', columns=config.ALL_ORIGINAL_COLS)
            print(f"数据成功保存到: {self.output_csv_path}")
            return True
        except Exception as e:
            print(f"保存处理后的数据到 CSV 时出错: {e}")
            return False

    def run_pipeline(self) -> Optional[pd.DataFrame]:
        """
        执行完整的数据处理流水线：加载 -> 处理 -> 保存。
        """
        df_combined = self.load_and_combine_sheets()
        if df_combined is None: return None
        df_processed = self.preprocess(df_combined)
        if df_processed is None: print("警告: 预处理步骤未生成有效 DataFrame 或返回 None。")
        # 即使 df_processed 为空，也尝试保存一个带表头的空文件
        if self.save_data(df_processed if df_processed is not None else pd.DataFrame(columns=config.ALL_ORIGINAL_COLS)):
            return df_processed
        else:
            print("警告: 数据处理完成（可能为空）但未能成功保存。")
            return df_processed

# --- 可选的直接执行入口 ---
if __name__ == "__main__":
    print("正在执行数据处理流程 (V6 Logic)...")
    processor = DataProcessor()
    processed_df = processor.run_pipeline()
    if processed_df is not None:
        print("\n数据处理流程执行完毕。")
        print(f"最终生成 {len(processed_df)} 行数据。")
        if not processed_df.empty:
            print("\n处理后数据的前 10 行:")
            try:
                print(processed_df.head(10).to_markdown(index=False))
            except ImportError:
                print(processed_df.head(10))
        else:
            print("\n处理后数据为空。")
    else:
        print("\n数据处理流程遇到错误。")
