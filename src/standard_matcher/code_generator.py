# -*- coding: utf-8 -*-
"""
模块：代码生成器 (Code Generator)
功能：负责根据选择的代码和模型顺序生成最终的产品型号代码。
"""

import json
import logging
import sys
import re
import time
import argparse
import os
import shutil # Added for temporary file cleanup
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Set

# 尝试导入模糊匹配库 (CodeGenerator 不直接使用 thefuzz)
# try:
#     from thefuzz import fuzz, process
#     THEFUZZ_AVAILABLE = True
# except ImportError:
#     THEFUZZ_AVAILABLE = False
#     # logger.warning("警告：'thefuzz' 库未安装。模糊匹配功能将不可用。请运行 'pip install thefuzz python-Levenshtein'")

# 确保项目根目录在 sys.path 中以便导入 config
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import settings
    # CodeGenerator 不直接使用 llm 或 json_processor
    # from src.standard_matcher.llm import call_llm_for_match
    # from src.standard_matcher.json_processor import AnalysisJsonProcessor
except ImportError as e:
    logging.getLogger(__name__).critical(
        f"错误：在 code_generator.py 中导入模块失败 - {e}。"
        f"请检查项目结构和 PYTHONPATH。\n"
        f"项目根目录尝试设置为: {project_root}", exc_info=True)
    raise

# --- 全局配置 ---
logger = logging.getLogger(__name__)

# 定义可跳过的 model 名称集合
SKIPPABLE_MODELS = {
    "变送器附加规格",
    "传感器附加规格",
    "套管附加规格"
}

# --- 文件路径定义 ---
# 这些路径在 standard_matcher.py 的主逻辑中使用，不移到这里
# DEFAULT_INDEX_JSON_PATH = project_root / "libs" / "standard" / "index.json"
# INDEX_JSON_PATH = Path(
#     getattr(settings, 'INDEX_JSON_PATH', DEFAULT_INDEX_JSON_PATH))
# TEMP_OUTPUT_DIR = project_root / "data" / "output" / "temp"


# ==============================================================================
# 4. Code Generator
# ==============================================================================

class CodeGenerator:
    """
    负责根据选择的代码和模型顺序生成最终的产品型号代码。
    """

    def get_model_order(self, csv_list_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        根据CSV列表映射和指定规则确定各产品类型下model的排序顺序。

        规则:
        1. 产品顺序: transmitter -> sensor -> tg
        2. 每个产品类型只读取其CSV列表中的第一个文件来确定该类型的model顺序。
        3. CSV内部按'model'列从上到下的首次出现顺序排序。

        Args:
            csv_list_map: 产品类型到CSV文件路径列表的映射字典。
                           例如: {"transmitter": ["path1.csv", "path2.csv"], ...}

        Returns:
            一个字典，键是产品类型 ("transmitter", "sensor", "tg")，
            值是该产品类型下按顺序排列的唯一 model 名称列表。
            如果某个产品类型的文件读取失败或列不存在，则其条目可能为空列表。
        """
        ordered_models_by_product: Dict[str, List[str]] = {
            "transmitter": [], "sensor": [], "tg": []}
        # 用于跟踪全局已添加的model，确保跨产品类型的唯一性（如果需要）
        # processed_models_globally: Set[str] = set() # 在当前逻辑下似乎不需要

        # 预定义的产品处理顺序
        product_order = ["transmitter", "sensor", "tg"]

        for product_type in product_order:
            # 确保字典中有该产品类型的键
            if product_type not in ordered_models_by_product:
                ordered_models_by_product[product_type] = []

            if product_type in csv_list_map and csv_list_map[product_type]:
                # 只取该产品类型的第一个CSV文件来确定顺序
                csv_path_str = csv_list_map[product_type][0]
                csv_path = project_root / csv_path_str  # FIX: Construct absolute path
                logger.info(f"正在处理产品 '{product_type}' 的CSV文件以确定顺序: {csv_path}")
                try:
                    # --- 重用 ModelMatcher 的 CSV 读取逻辑 ---
                    df = None
                    try:
                        df = pd.read_csv(csv_path, dtype=str)
                    except UnicodeDecodeError:
                        logger.warning(
                            f"Order: 使用 utf-8 读取 {csv_path} 失败，尝试 gbk...")
                        try:
                            df = pd.read_csv(
                                csv_path, encoding='gbk', dtype=str)
                        except Exception as e_gbk:
                            logger.warning(
                                f"Order: 使用 gbk 读取 {csv_path} 失败: {e_gbk}。尝试分号...")
                    except Exception as e_utf8_comma:
                        logger.warning(
                            f"Order: 使用 utf-8 和逗号读取 {csv_path} 失败: {e_utf8_comma}。尝试分号...")

                    if df is None or 'model' not in df.columns:
                        logger.warning(f"Order: 尝试使用分号分隔符读取 {csv_path}...")
                        try:
                            df_semi = pd.read_csv(csv_path, sep=';', dtype=str)
                            if 'model' in df_semi.columns:
                                df = df_semi
                            else:
                                logger.warning(
                                    f"Order: 使用 utf-8 和分号读取 {csv_path} 仍缺少 'model' 列。尝试 GBK...")
                        except UnicodeDecodeError:
                            logger.warning(
                                f"Order: 使用 utf-8 和分号读取 {csv_path} 失败，尝试 gbk...")
                        except Exception as e_utf8_semi:
                            logger.warning(
                                f"Order: 使用 utf-8 和分号读取 {csv_path} 失败: {e_utf8_semi}。尝试 GBK...")

                        if df is None or 'model' not in df.columns:
                            try:
                                df_semi_gbk = pd.read_csv(
                                    csv_path, sep=';', encoding='gbk', dtype=str)
                                if 'model' in df_semi_gbk.columns:
                                    df = df_semi_gbk
                                else:
                                    logger.warning(
                                        f"Order: 尝试多种方式后，CSV文件 '{csv_path}' 中仍未找到 'model' 列。")
                            except Exception as e_gbk_semi:
                                logger.warning(
                                    f"Order: 尝试多种方式读取 CSV 文件 {csv_path} 失败: {e_gbk_semi}")

                    # --- 提取顺序 ---
                    if df is not None and 'model' in df.columns:
                        # 提取model列，去除NaN/空值，并转换成字符串以防数字等类型
                        models_in_csv = df['model'].dropna().astype(
                            str).tolist()
                        # 获取当前CSV中唯一的model，并保持首次出现的顺序
                        models_in_this_csv_ordered = []
                        seen_in_this_csv = set()
                        for model in models_in_csv:
                            model_clean = model.strip()  # 去除首尾空格
                            if model_clean and model_clean not in seen_in_this_csv:
                                models_in_this_csv_ordered.append(model_clean)
                                seen_in_this_csv.add(model_clean)

                        # 将这些 model 添加到对应产品类型的列表中
                        ordered_models_by_product[product_type].extend(
                            models_in_this_csv_ordered)
                        logger.debug(
                            f"为产品 '{product_type}' 添加 models 顺序: {models_in_this_csv_ordered}")

                    # else: # 警告已在读取逻辑中发出

                except FileNotFoundError:
                    logger.error(f"错误：找不到用于确定顺序的CSV文件 '{csv_path}'。")
                except pd.errors.EmptyDataError:
                    logger.warning(f"警告：用于确定顺序的CSV文件 '{csv_path}' 为空。")
                except Exception as e:
                    logger.error(f"读取或处理用于确定顺序的CSV文件 '{csv_path}' 时出错: {e}")
            else:
                logger.warning(
                    f"在CSV列表映射中未找到产品类型 '{product_type}' 或其列表为空，无法确定其 model 顺序。")

        logger.info(f"最终确定的各产品类型 model 排序: {ordered_models_by_product}")
        return ordered_models_by_product

    def _preload_model_details(self, csv_list_map: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """
        预加载所有相关 CSV 文件中的模型详情，包括默认代码和所有代码。

        Args:
            csv_list_map: 产品类型到 CSV 文件路径列表的映射。

        Returns:
            一个字典，键是 model 名称，值是包含 "default_code", "all_codes", "product_type" 的字典。
        """
        model_details_map: Dict[str, Dict[str, Any]] = {}
        product_order_for_defaults = [
            "transmitter", "sensor", "tg"]  # 与 get_model_order 一致

        processed_files = set()  # 避免重复处理同一个文件

        for product_type_default in product_order_for_defaults:
            if product_type_default in csv_list_map and csv_list_map[product_type_default]:
                # 只处理第一个文件来确定顺序，但需要加载所有文件以获取所有可能的默认值？
                # 当前逻辑是只从第一个文件加载默认值，这可能不完整。
                # 改进：应该加载所有文件来查找默认值。
                for csv_path_str in csv_list_map[product_type_default]:
                    if csv_path_str in processed_files:
                        continue  # 跳过已处理的文件
                    processed_files.add(csv_path_str)

                    csv_path_default = project_root / csv_path_str # FIX: Construct absolute path
                    logger.info(
                        f"正在为产品 '{product_type_default}' 从 {csv_path_default} 预加载模型详情...")
                    try:
                        # --- 重用 ModelMatcher 的 CSV 读取逻辑 ---
                        df_default = None
                        try:
                            df_default = pd.read_csv(
                                csv_path_default, dtype=str).fillna('')
                        except UnicodeDecodeError:
                            try:
                                df_default = pd.read_csv(
                                    csv_path_default, encoding='gbk', dtype=str).fillna('')
                            except Exception:
                                pass
                        except Exception:
                            pass

                        if df_default is None or not {'model', 'code', 'is_default'}.issubset(df_default.columns):
                            try:
                                df_semi = pd.read_csv(
                                    csv_path_default, sep=';', dtype=str).fillna('')
                                if {'model', 'code', 'is_default'}.issubset(df_semi.columns):
                                    df_default = df_semi
                            except UnicodeDecodeError:
                                try:
                                    df_semi_gbk = pd.read_csv(
                                        csv_path_default, sep=';', encoding='gbk', dtype=str).fillna('')
                                    if {'model', 'code', 'is_default'}.issubset(df_semi_gbk.columns):
                                        df_default = df_semi_gbk
                                except Exception:
                                    pass
                            except Exception:
                                pass

                        # 确保 'model', 'code', 'is_default' 列存在
                        if df_default is not None and all(col in df_default.columns for col in ['model', 'code', 'is_default']):
                            # 清理数据：去除 model 或 code 为空的行，并将 model, code, is_default 转为字符串
                            required_cols = ['model', 'code', 'is_default']
                            # 先填充 NaN
                            df_default[required_cols] = df_default[required_cols].fillna(
                                '')
                            # 过滤 model 或 code 为空的行
                            df_cleaned_default = df_default[
                                (df_default['model'].astype(str).str.strip() != '') &
                                (df_default['code'].astype(
                                    str).str.strip() != '')
                            ].copy()  # 使用 .copy() 避免 SettingWithCopyWarning
                            # 全部转为字符串并去除空格
                            for col in required_cols:
                                df_cleaned_default[col] = df_cleaned_default[col].astype(
                                    str).str.strip()

                            # 获取每个 model 对应的所有 code 值
                            model_to_all_codes = df_cleaned_default.groupby(
                                'model')['code'].apply(list).to_dict()

                            # 查找每个 model 的默认 code (is_default == '1')
                            default_rows = df_cleaned_default[df_cleaned_default['is_default'] == '1']
                            # 使用 drop_duplicates 确保每个 model 只取一个默认值（如果CSV中有多个标记为1）
                            model_to_default_code = default_rows.drop_duplicates(
                                subset=['model']).set_index('model')['code'].to_dict()

                            # 遍历此CSV中的唯一 model
                            for model_name_default in df_cleaned_default['model'].unique():
                                if model_name_default not in model_details_map:  # 避免被后续产品类型覆盖
                                    all_codes_for_model = model_to_all_codes.get(
                                        model_name_default, [])
                                    default_code_for_model = model_to_default_code.get(
                                        model_name_default)  # 获取默认 code

                                    details = {
                                        "default_code": default_code_for_model,  # 可能为 None
                                        "all_codes": all_codes_for_model,
                                        "product_type": product_type_default  # 记录来源产品类型
                                    }
                                    model_details_map[model_name_default] = details

                                    # if default_code_for_model:
                                    #     logger.debug(f"为 model '{model_name_default}' 找到默认代码: {default_code_for_model}")
                                    # else:
                                    #     logger.debug(f"Model '{model_name_default}' 未找到标记为默认 (is_default='1') 的代码。所有代码: {all_codes_for_model}")

                        else:
                            missing_cols = {'model', 'code', 'is_default'} - \
                                set(df_default.columns if df_default is not None else [])
                            logger.warning(
                                f"预加载: CSV文件 '{csv_path_default}' 缺少必需列: {missing_cols}，无法为 {product_type_default} 生成模型详情。")

                    except FileNotFoundError:
                        logger.error(
                            f"预加载: 找不到CSV文件 '{csv_path_default}'，无法为 {product_type_default} 生成模型详情。")
                    except pd.errors.EmptyDataError:
                        logger.warning(
                            f"预加载: CSV文件 '{csv_path_default}' 为空，无法为 {product_type_default} 生成模型详情。")
                    except Exception as e:
                        logger.error(
                            f"预加载: 处理CSV文件 '{csv_path_default}' 以生成模型详情时出错: {e}")

        logger.info(f"预加载的模型详情映射完成: {len(model_details_map)} 个条目")
        # logger.debug(f"模型详情映射内容: {model_details_map}") # 内容可能过多，谨慎开启
        return model_details_map

    def generate_final_code(self, csv_list_map: Dict[str, List[str]], selected_codes_data: Dict[str, Dict[str, Any]]) -> str:
        """
        根据确定的各产品类型model顺序，从selected_codes_data中提取代码，
        将同一产品类型的代码连接，不同产品类型的代码块用空格分隔。

        Args:
            csv_list_map: 产品类型到CSV文件路径列表的映射字典。
            selected_codes_data: 从 CodeSelector 输出的 JSON 结构，
                                 格式为 {"'key': 'value'": {"model": "模型名", "code": "代码", ...}}。

        Returns:
            最终拼接成的产品型号代码字符串，格式如 "产品型号生成：transmitter_code_block sensor_code_block tg_code_block"。
            如果无法生成代码，则返回错误信息。
        """
        logger.info("开始生成最终产品代码...")
        logger.debug(f"接收到的CSV列表映射: {csv_list_map}")
        # logger.debug(f"接收到的已选代码数据: {selected_codes_data}") # 可能过长

        # 1. 获取按产品类型分组的 model 顺序
        model_order_by_product = self.get_model_order(csv_list_map)

        if not any(model_order_by_product.values()):  # 检查是否所有产品类型的 model 列表都为空
            logger.error("未能确定任何产品类型的 model 排序顺序，无法生成代码。")
            return "产品型号生成失败：无法确定任何排序。"

        # 2. 预加载模型详情 (默认代码等)
        model_details_map = self._preload_model_details(csv_list_map)

        # 3. 构建一个从 model 名称到 code 的快速查找字典 (来自 selected_codes_data)
        model_to_code_map = {}
        found_models_in_selection = set()
        for input_str, param_data in selected_codes_data.items():
            if isinstance(param_data, dict) and 'model' in param_data and 'code' in param_data:
                model_name = str(param_data['model']).strip()  # 确保是字符串并去空格
                code_value = str(param_data['code']).strip() if pd.notna(
                    param_data['code']) else ''  # 处理可能的 NaN 或 None 并去空格
                if model_name:  # 确保 model 名称不为空
                    model_to_code_map[model_name] = code_value
                    found_models_in_selection.add(model_name)
                else:
                    logger.warning(
                        f"selected_codes_data 中的条目 '{input_str}' 包含空的 model 名称，已跳过。")
            else:
                logger.warning(
                    f"selected_codes_data 中的条目 '{input_str}' 格式不符合预期或缺少 model/code: {param_data}")

        logger.debug(
            f"从 selected_codes_data 构建的 model->code 映射: {len(model_to_code_map)} 个条目")
        # logger.debug(f"在 selected_codes_data 中找到的 models: {found_models_in_selection}") # 可能过长

        # --- 1. 预先计算条件 ---
        logger.info("--- CodeGenerator: 进入预计算条件部分 ---")
        logger.debug(f"预计算前 CSV列表映射 (完整): {json.dumps(csv_list_map, indent=2, ensure_ascii=False)}")
        logger.debug(f"预计算前 CSV列表映射的类型: {type(csv_list_map)}")

        value_for_tg_debug = csv_list_map.get('tg', [None])
        logger.debug(f"  csv_list_map.get('tg', [None]) 的结果 (value_for_tg_debug): {value_for_tg_debug}")
        logger.debug(f"  value_for_tg_debug 的类型: {type(value_for_tg_debug)}")

        if isinstance(value_for_tg_debug, list):
            logger.debug(f"  value_for_tg_debug 是列表，长度: {len(value_for_tg_debug)}")
            if len(value_for_tg_debug) > 0:
                logger.debug(f"  value_for_tg_debug[0] 的值: {value_for_tg_debug[0]}")
                logger.debug(f"  value_for_tg_debug[0] 的类型: {type(value_for_tg_debug[0])}")
        elif isinstance(value_for_tg_debug, str): # 理论上不应是字符串，但以防万一
            logger.debug(f"  value_for_tg_debug 是字符串，内容: {value_for_tg_debug}")
            if len(value_for_tg_debug) > 0:
                logger.debug(f"  value_for_tg_debug[0] 的值: {value_for_tg_debug[0]}")
                logger.debug(f"  value_for_tg_debug[0] 的类型: {type(value_for_tg_debug[0])}")

        has_tg_product = 'tg' in csv_list_map and bool(csv_list_map.get('tg'))
        has_sensor_product = 'sensor' in csv_list_map and bool(
            csv_list_map.get('sensor'))

        tg_csv_path = None # 初始化
        logger.info("准备执行 tg_csv_path 的获取逻辑...")
        try:
            intermediate_val = csv_list_map.get('tg', [None])
            logger.info(f"  intermediate_val (csv_list_map.get('tg', [None])): {intermediate_val}")
            logger.info(f"  intermediate_val 类型: {type(intermediate_val)}")
            if intermediate_val is not None and isinstance(intermediate_val, list) and len(intermediate_val) > 0:
                tg_csv_path = intermediate_val[0]
                logger.info(f"  成功获取 tg_csv_path: {tg_csv_path} (类型: {type(tg_csv_path)})")
            elif intermediate_val is not None and isinstance(intermediate_val, list) and len(intermediate_val) == 0:
                logger.info("  intermediate_val 是一个空列表，tg_csv_path 将保持为 None (或根据逻辑处理为 IndexError)。")
            else: # intermediate_val is None or not a list or an empty list (already handled)
                logger.info(f"  intermediate_val 不是预期的非空列表 (可能是 [None] 或其他)，tg_csv_path 将为 None。")
                # 如果 intermediate_val 是 [None]，那么 [None][0] 是 None。
                if intermediate_val == [None]: # 特殊处理默认情况
                    tg_csv_path = intermediate_val[0] # 这会是 None
                    logger.info(f"  intermediate_val 是 [None]，tg_csv_path 设为: {tg_csv_path}")


        except TypeError as te:
            logger.error(f"  在获取 tg_csv_path 时发生 TypeError: {te}", exc_info=True)
            raise # 重新抛出原始的 TypeError
        except IndexError as ie:
            logger.error(f"  在获取 tg_csv_path 时发生 IndexError: {ie}", exc_info=True)
            # 根据原始代码，如果列表为空，这里应该发生 IndexError
            # 如果原始代码不应该在这里处理 IndexError，而是依赖后续的 Path(None) TypeError，那么这里可以不 raise
            # 但为了调试，我们先记录并重新抛出
            raise
        except Exception as e:
            logger.error(f"  在获取 tg_csv_path 时发生其他未知错误: {e}", exc_info=True)
            raise

        # 确保比较时路径格式一致 (例如，都使用 posix 风格)
        specific_tg_csvs = {'libs/standard/tg/TG_PT-1.csv',
                            'libs/standard/tg/TG_PT-2.csv', 'libs/standard/tg/TG_PT-3.csv'}
        # 注意：如果 tg_csv_path 为 None，Path(tg_csv_path) 会在 Python 3.9+ 产生 TypeError
        # 我们需要确保 is_specific_tg_csv 的计算考虑到 tg_csv_path 可能为 None
        is_specific_tg_csv = False # 默认值
        if tg_csv_path is not None:
            try:
                is_specific_tg_csv = Path(tg_csv_path).as_posix() in specific_tg_csvs
            except TypeError as e_path: # 捕获 Path(None) 可能的 TypeError
                logger.error(f"  创建 Path 对象时出错 (tg_csv_path: {tg_csv_path}): {e_path}")
                # is_specific_tg_csv 保持 False
        else:
            logger.info("  tg_csv_path 为 None，is_specific_tg_csv 将为 False。")
        logger.info(
            f"规则条件检查: has_tg={has_tg_product}, has_sensor={has_sensor_product}, is_specific_tg_csv={is_specific_tg_csv} (path: {tg_csv_path})")

        # SKIPPABLE_MODELS 已在文件顶部定义

        # 4. 按产品顺序处理并拼接代码块
        product_code_strings = []
        input_requests = []  # 初始化输入请求列表
        product_order = ["transmitter", "sensor", "tg"]  # 预定义的产品处理顺序
        missing_models_log = {}  # 记录每个产品类型在 selected_codes_data 中缺失的 model

        for product_type in product_order:
            models_for_product = model_order_by_product.get(product_type, [])
            codes_for_this_product = []
            missing_models_for_product = []

            if not models_for_product:
                logger.info(f"产品类型 '{product_type}' 没有需要处理的 models。")
                continue  # 跳过这个产品类型

            logger.debug(
                f"处理产品类型 '{product_type}' 的 models: {models_for_product}")

            for target_model in models_for_product:
                target_model_str = str(target_model).strip()  # 确保比较时类型一致且无空格
                if not target_model_str:
                    continue  # 跳过空模型

                code_to_use = None
                source = "unknown"  # 初始化来源
                handled_by_rule = False  # 标记是否被新规则处理
                product_type_origin = product_type  # 用于日志和 %int% 提示

                # --- 2. 复杂规则处理块 ---
                if target_model_str == '插入长度（L）' and has_tg_product:
                    logger.info(
                        f"规则 1 触发：因存在 'tg' 产品，跳过模型 '{target_model_str}'。")
                    handled_by_rule = True
                    source = "rule_1_skip"

                elif target_model_str == '传感器连接螺纹（S）' and has_sensor_product:
                    logger.info(
                        f"规则 2 触发：因存在 'sensor' 产品，跳过模型 '{target_model_str}'。")
                    handled_by_rule = True
                    source = "rule_2_skip"

                elif target_model_str == '接头结构' and is_specific_tg_csv:
                    code_to_use = '2'
                    logger.info(
                        f"规则 3 触发：因 'tg' 产品使用特定 CSV ({tg_csv_path})，模型 '{target_model_str}' 代码强制为 '2'。")
                    handled_by_rule = True
                    source = "rule_3_override"

                elif target_model_str == "传感器输入":
                    element_quantity_code = model_to_code_map.get("元件数量")
                    if element_quantity_code == "-S":
                        code_to_use = "1"
                        logger.info(
                            f"规则 4 触发：model '元件数量' code 为 -S，强制 model '传感器输入' ({product_type}) code 为 '1'")
                        handled_by_rule = True
                        source = "rule_4_element_quantity_S"
                    elif element_quantity_code == "-D":
                        code_to_use = "2"
                        logger.info(
                            f"规则 4 触发：model '元件数量' code 为 -D，强制 model '传感器输入' ({product_type}) code 为 '2'")
                        handled_by_rule = True
                        source = "rule_4_element_quantity_D"

                elif target_model_str == "法兰材质" or target_model_str == "套管材质":
                    flange_material_code = model_to_code_map.get("法兰材质")
                    sleeve_material_code = model_to_code_map.get("套管材质")

                    # 检查 code 是否有效 (非 None 且非空字符串)
                    flange_code_specified = flange_material_code is not None and flange_material_code != ""
                    sleeve_code_specified = sleeve_material_code is not None and sleeve_material_code != ""

                    if target_model_str == "套管材质" and flange_code_specified and not sleeve_code_specified:
                        code_to_use = flange_material_code
                        logger.info(
                            f"规则 5 触发：model '法兰材质' code 为 '{flange_material_code}'，'套管材质' 未指定 code，"
                            f"强制 model '套管材质' ({product_type}) code 与 '法兰材质' 一致。")
                        handled_by_rule = True
                        source = "rule_5_sleeve_from_flange"
                    elif target_model_str == "法兰材质" and sleeve_code_specified and not flange_code_specified:
                        code_to_use = sleeve_material_code
                        logger.info(
                            f"规则 5 触发：model '套管材质' code 为 '{sleeve_material_code}'，'法兰材质' 未指定 code，"
                            f"强制 model '法兰材质' ({product_type}) code 与 '套管材质' 一致。")
                        handled_by_rule = True
                        source = "rule_5_flange_from_sleeve"

                elif target_model_str == "接线盒形式":
                    if not model_to_code_map.get("接线盒形式"):
                        wiring_port_code = model_to_code_map.get("接线口")
                        if wiring_port_code == "2":
                            code_to_use = "-2"
                            logger.info(
                                f"规则 6 触发：'接线盒形式' 缺失，'接线口' code 为 '2'，强制 model '接线盒形式' ({product_type}) code 为 '-2'")
                            handled_by_rule = True
                            source = "rule_6_missing_jxhxs_wp_2"
                        elif wiring_port_code == "4":
                            code_to_use = "-3"
                            logger.info(
                                f"规则 6 触发：'接线盒形式' 缺失，'接线口' code 为 '4'，强制 model '接线盒形式' ({product_type}) code 为 '-3'")
                            handled_by_rule = True
                            source = "rule_6_missing_jxhxs_wp_4"

                elif target_model_str == "铠套材质" or target_model_str == "套管材质":
                    armored_sheath_code_val = model_to_code_map.get("铠套材质")
                    thermowell_code_val = model_to_code_map.get("套管材质")
                    specific_thermowell_codes_for_rule7 = {"PN", "QN", "RN", "GH", "Z"}

                    # 检查 code 是否有效 (非 None 且非空字符串)
                    armored_sheath_code_is_specified = armored_sheath_code_val is not None and armored_sheath_code_val != ""
                    thermowell_code_is_specified = thermowell_code_val is not None and thermowell_code_val != ""

                    if target_model_str == "铠套材质" and not armored_sheath_code_is_specified: # 铠套材质缺失
                        if thermowell_code_is_specified and thermowell_code_val in specific_thermowell_codes_for_rule7:
                            code_to_use = thermowell_code_val
                            logger.info(
                                f"规则 7 触发：'铠套材质' ({product_type}) 缺失，'套管材质' code ('{thermowell_code_val}') 在特定集合中，"
                                f"强制 '铠套材质' code 与 '套管材质' 一致。")
                            handled_by_rule = True
                            source = "rule_7_armored_from_thermowell_specific"

                    elif target_model_str == "套管材质" and not thermowell_code_is_specified: # 套管材质缺失
                        if armored_sheath_code_is_specified: # 铠套材质存在且有值
                            code_to_use = armored_sheath_code_val
                            logger.info(
                                f"规则 7 触发：'套管材质' ({product_type}) 缺失，'铠套材质' code 为 '{armored_sheath_code_val}'，"
                                f"强制 '套管材质' code 与 '铠套材质' 一致。")
                            handled_by_rule = True
                            source = "rule_7_thermowell_from_armored"


                elif target_model_str == "传感器防爆规格":
                    transmitter_additional_spec_code = model_to_code_map.get("变送器附加规格")
                    if transmitter_additional_spec_code == "/NF2":
                        code_to_use = "/N1"
                        logger.info(
                            f"规则 8 触发：变送器附加规格 code 为 /NF2，强制传感器防爆规格 code 为 /N1")
                        handled_by_rule = True
                        source = "rule_8_override"
                    elif transmitter_additional_spec_code in ["/NS2", "/NS25"]:
                        code_to_use = "/N2"
                        logger.info(
                            f"规则 8 触发：变送器附加规格 code 为 /NS2 或 /NS25，强制传感器防爆规格 code 为 /N2")
                        handled_by_rule = True
                        source = "rule_8_override"

                # --- 3. 标准代码查找 (仅当未被规则处理时) ---
                if not handled_by_rule:
                    if target_model_str in model_to_code_map:
                        # 在 selected_codes_data 中找到
                        code_to_use = model_to_code_map[target_model_str]
                        source = "selected"
                        logger.debug(
                            f"找到产品 '{product_type}' 的 model '{target_model_str}' 对应的代码 (来自选择): '{code_to_use}'")
                    else:
                        # 在 selected_codes_data 中未找到
                        missing_models_for_product.append(target_model_str)

                        # --- 旧可跳过逻辑 ---
                        if target_model_str in SKIPPABLE_MODELS:
                            logger.info(
                                f"规则 通用 触发：产品 '{product_type}' 的 model '{target_model_str}' 在已选代码中缺失，且为可跳过项，将跳过。")
                            source = "rule_1_skip"
                            # code_to_use 保持 None
                        else:
                            # 非可跳过项，尝试默认值
                            model_details = model_details_map.get(
                                target_model_str)
                            if model_details:
                                default_code = model_details.get(
                                    "default_code")
                                product_type_origin = model_details.get(
                                    "product_type", product_type)  # 获取原始产品类型

                                if default_code is not None:  # 检查是否为 None
                                    logger.info(
                                        f"在已选代码中未找到产品 '{product_type_origin}' 的 model '{target_model_str}' (非可跳过项)，将使用其默认代码: '{default_code}'")
                                    code_to_use = str(default_code)  # 确保是字符串
                                    source = "default"
                                else:
                                    logger.warning(
                                        f"在已选代码中未找到产品 '{product_type_origin}' 的 model '{target_model_str}' (非可跳过项)，且该 model 在 CSV 中没有标记为默认 (is_default='1') 的代码，将使用 '?'。")
                                    code_to_use = "?"
                                    source = "missing_default"
                            else:
                                logger.error(
                                    f"严重警告：在 model_details_map 中未找到 model '{target_model_str}' 的详情，无法确定默认代码，使用 '?'。")
                                code_to_use = "?"
                                source = "missing_details"

                    # 收集需要用户输入的参数
                    if code_to_use is not None and "%int%" in str(code_to_use):
                        input_requests.append({
                            "product_type": product_type_origin if source == "default" else product_type,
                            "model_name": target_model_str,
                            "code_template": str(code_to_use),
                            "position": len(codes_for_this_product)  # 记录在代码块中的位置
                        })
                        # 使用临时占位符
                        codes_for_this_product.append("%INT_PLACEHOLDER%")
                        logger.debug(f"为 {target_model_str} 添加输入请求，使用临时占位符")
                    elif code_to_use is not None:
                        codes_for_this_product.append(str(code_to_use))
                        logger.debug(f"添加 {target_model_str} 的代码: {code_to_use}")
                    else:
                        logger.debug(f"Model '{target_model_str}' 被跳过，不生成代码。")

            if missing_models_for_product:
                missing_models_log[product_type] = missing_models_for_product

            if codes_for_this_product:  # 如果这个产品类型找到了任何代码
                # 将该产品类型的所有代码连接起来，中间不加空格
                product_string = "".join(codes_for_this_product)
                product_code_strings.append(product_string)
                logger.debug(f"生成产品 '{product_type}' 的代码块: {product_string}")
            else:
                logger.info(
                    f"产品类型 '{product_type}' 没有生成任何代码（可能所有 model 都被跳过或无对应 code）。")

        if missing_models_log:
            logger.warning(
                # 更新日志信息
                f"以下 models 根据排序规则需要，但在 selected_codes_data 中缺失（已使用默认值或 '?' 替代，或被规则跳过）: {missing_models_log}")

        # 5. 将不同产品的代码字符串用空格连接
        final_code = " ".join(product_code_strings)

        # 6. 返回结果和输入请求
        result = {
            "final_code": " ".join(product_code_strings),
            "input_requests": input_requests,
            "ordered_models": model_order_by_product
        }
        logger.info(f"代码生成完成，等待用户输入")
        return result
