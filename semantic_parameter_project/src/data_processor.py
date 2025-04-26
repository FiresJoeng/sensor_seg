# src/data_processor.py
import pandas as pd
import numpy as np
import os
from typing import List, Optional

# 从 config 模块导入配置
# 假设 config.py 与 data_processor.py 在同一 src 目录下
try:
    from . import config
except ImportError:
    # 如果直接运行此脚本，可能需要调整导入方式
    import config

class DataProcessor:
    """
    负责从 Excel 加载数据、进行预处理（合并、清理、拆分、展开）并保存。
    """
    def __init__(self, excel_path: str = config.EXCEL_FILE_PATH,
                 sheet_names: List[str] = config.SHEET_NAMES,
                 output_csv_path: str = config.PREPROCESSED_DATA_PATH):
        """
        初始化 DataProcessor。

        Args:
            excel_path (str): 输入 Excel 文件路径。
            sheet_names (List[str]): 需要读取的 Excel 工作表名称列表。
            output_csv_path (str): 处理后数据保存的 CSV 文件路径。
        """
        self.excel_path = excel_path
        self.sheet_names = sheet_names
        self.output_csv_path = output_csv_path
        # 定义预处理所需的关键列名
        self.required_cols = [
            config.COL_ACTUAL_VALUE_VARIATIONS, config.COL_STANDARD_VALUE_DESC,
            config.COL_STANDARD_CODE, config.COL_STANDARD_PARAM, config.COL_COMPONENT_PART
        ]

    def load_and_combine_sheets(self) -> Optional[pd.DataFrame]:
        """
        从 Excel 文件加载指定的工作表并将它们合并。

        Returns:
            Optional[pd.DataFrame]: 合并后的 DataFrame，如果出错则返回 None。
        """
        print(f"--- 正在加载并合并 Excel 工作表 ---")
        print(f"文件: {self.excel_path}")

        if not os.path.exists(self.excel_path):
            print(f"错误: Excel 文件未找到于 '{self.excel_path}'")
            return None

        all_dfs = []
        try:
            # 遍历需要读取的工作表名称
            for sheet in self.sheet_names:
                print(f"  正在读取工作表: '{sheet}'...")
                try:
                    # 读取 Excel 表格，指定 dtype=str 确保所有数据按字符串读取，避免格式问题
                    df_sheet = pd.read_excel(self.excel_path, sheet_name=sheet, dtype=str)
                    all_dfs.append(df_sheet)
                    print(f"    '{sheet}' 读取成功，形状: {df_sheet.shape}")
                except Exception as sheet_error:
                    # 如果某个工作表读取失败，打印警告并跳过
                    print(f"  警告: 无法读取工作表 '{sheet}'。错误: {sheet_error}。将跳过此表。")

            # 如果没有任何工作表成功读取，则返回错误
            if not all_dfs:
                print("错误: 未能从 Excel 文件成功读取任何工作表。")
                return None

            # 使用 concat 合并所有读取到的 DataFrame
            df_combined = pd.concat(all_dfs, ignore_index=True)
            print(f"\n已合并来自 {len(all_dfs)} 个工作表的数据。合并后形状: {df_combined.shape}")
            print(f"合并后列名: {df_combined.columns.tolist()}")
            return df_combined

        except Exception as e:
            # 捕获其他可能的读取或合并错误
            print(f"读取或合并 Excel 工作表时出错: {e}")
            return None

    def preprocess(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        对合并后的 DataFrame 进行预处理：选择列、处理缺失值、拆分和展开。

        Args:
            df (pd.DataFrame): 待处理的 DataFrame。

        Returns:
            Optional[pd.DataFrame]: 处理后的 DataFrame，如果出错则返回 None。
        """
        print("\n--- 正在进行数据预处理 ---")
        if df is None:
            print("错误: 输入的 DataFrame 为空，无法处理。")
            return None

        # 1. 检查并选择相关列
        print(f"选择相关列: {self.required_cols}")
        # 检查所有必需列是否存在于 DataFrame 中
        missing_cols = [col for col in self.required_cols if col not in df.columns]
        if missing_cols:
            # 如果缺少关键列，打印错误并返回 None
            print(f"错误: 合并后的数据中缺少以下关键列: {missing_cols}。无法继续处理。")
            return None

        # 仅保留需要的列
        df_prepared = df[self.required_cols].copy()

        # 2. 处理待拆分列的缺失值
        split_col = config.COL_ACTUAL_VALUE_VARIATIONS
        print(f"处理列 '{split_col}' 的缺失值 (填充为空字符串)...")
        # 使用空字符串填充 NaN 值，以便后续进行字符串操作
        df_prepared[split_col] = df_prepared[split_col].fillna('')

        # 3. 拆分和展开
        print(f"  按 '|' 拆分 '{split_col}' 中的值...")
        # 确保该列是字符串类型，然后使用 str.split 进行拆分
        df_prepared[split_col] = df_prepared[split_col].astype(str).str.split('|')

        print("  展开行 (基于拆分后的列表)...")
        # 使用 explode 方法将包含列表的行展开为多行
        df_exploded = df_prepared.explode(split_col)

        # 4. 清理和重命名
        exploded_col_name = config.PREPROCESSED_COL_ACTUAL_VARIATION
        # 重命名展开后的列，方便后续引用
        df_exploded.rename(columns={split_col: exploded_col_name}, inplace=True)
        # 去除拆分后可能产生的多余空格
        df_exploded[exploded_col_name] = df_exploded[exploded_col_name].str.strip()
        print(f"  重命名展开列为 '{exploded_col_name}' 并去除前后空格。")

        # 5. 过滤无效数据
        print("  过滤空值变体以及缺少标准值/代码的行...")
        original_rows = len(df_exploded)
        # 获取标准值和标准代码的列名
        standard_value_col = config.COL_STANDARD_VALUE_DESC
        standard_code_col = config.COL_STANDARD_CODE

        # 确保过滤所需的列存在
        if standard_value_col not in df_exploded.columns or standard_code_col not in df_exploded.columns:
             print(f"  错误: 标准列 ('{standard_value_col}', '{standard_code_col}') 在过滤前丢失。")
             return None

        # 定义过滤条件：
        # - 实际参数值变体 (展开后的列) 不为空且非 NaN
        # - 标准参数值描述不为空且非 NaN
        # - 标准代码不为空且非 NaN
        df_final = df_exploded[
            (df_exploded[exploded_col_name].notna()) & (df_exploded[exploded_col_name] != '') &
            (df_exploded[standard_value_col].notna()) & (df_exploded[standard_value_col] != '') &
            (df_exploded[standard_code_col].notna()) & (df_exploded[standard_code_col] != '')
        ].copy() # 使用 .copy() 避免 SettingWithCopyWarning

        removed_rows = original_rows - len(df_final)
        print(f"  过滤完成，移除了 {removed_rows} 行。")

        # 6. 重置索引
        # 重置 DataFrame 索引，使之连续
        df_final.reset_index(drop=True, inplace=True)
        print(f"\n预处理完成。最终 DataFrame 形状: {df_final.shape}")
        print(f"最终列名: {df_final.columns.tolist()}")

        return df_final

    def save_data(self, df: pd.DataFrame) -> bool:
        """
        将处理后的 DataFrame 保存到 CSV 文件。

        Args:
            df (pd.DataFrame): 要保存的 DataFrame。

        Returns:
            bool: 如果保存成功返回 True，否则返回 False。
        """
        if df is None:
            print("错误: 没有数据可保存。")
            return False
        print(f"\n--- 正在保存处理后的数据 ---")
        try:
            # 获取输出目录路径
            output_dir = os.path.dirname(self.output_csv_path)
            # 如果目录不存在，则创建目录
            os.makedirs(output_dir, exist_ok=True)
            # 将 DataFrame 保存为 CSV 文件，使用 utf-8-sig 编码确保中文正确显示
            df.to_csv(self.output_csv_path, index=False, encoding='utf-8-sig')
            print(f"数据成功保存到: {self.output_csv_path}")
            return True
        except Exception as e:
            # 捕获保存文件时可能发生的错误
            print(f"保存处理后的数据到 CSV 时出错: {e}")
            return False

    def run_pipeline(self) -> Optional[pd.DataFrame]:
        """
        执行完整的数据处理流水线：加载 -> 处理 -> 保存。

        Returns:
            Optional[pd.DataFrame]: 最终处理后的 DataFrame，如果中途出错则返回 None。
        """
        # 依次执行加载、处理和保存步骤
        df_combined = self.load_and_combine_sheets()
        if df_combined is None:
            return None # 加载失败则终止
        df_processed = self.preprocess(df_combined)
        if df_processed is None:
            return None # 处理失败则终止
        if self.save_data(df_processed):
            return df_processed # 保存成功，返回处理后的数据
        else:
            # 保存失败，打印警告但仍然返回处理后的数据
            print("警告: 数据处理完成但未能成功保存。")
            return df_processed

# --- 可选的直接执行入口 ---
# 当直接运行 data_processor.py 时，执行数据处理流程
if __name__ == "__main__":
    print("正在执行数据处理流程...")
    processor = DataProcessor()
    processed_df = processor.run_pipeline()

    if processed_df is not None:
        print("\n数据处理流程成功完成。")
        print("\n处理后数据的前5行:")
        # 使用 to_markdown 打印更美观的表格形式
        # 需要安装 tabulate 库: pip install tabulate
        try:
            print(processed_df.head().to_markdown(index=False))
        except ImportError:
            print(processed_df.head()) # 如果没装 tabulate，则直接打印
    else:
        print("\n数据处理流程失败。")
