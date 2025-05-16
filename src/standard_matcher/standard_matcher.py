# -*- coding: utf-8 -*-
"""
模块：标准匹配器 (Standard Matcher)
功能：整合了从获取CSV列表、模型匹配、代码选择到最终代码生成的完整流程。
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

# 尝试导入模糊匹配库
try:
    from thefuzz import fuzz, process
    THEFUZZ_AVAILABLE = True
except ImportError:
    THEFUZZ_AVAILABLE = False
    # 在模块级别打印警告，而不是在类初始化时，以避免在导入时就因 print 而产生副作用
    # logger.warning("警告：'thefuzz' 库未安装。模糊匹配功能将不可用。请运行 'pip install thefuzz python-Levenshtein'")

# 确保项目根目录在 sys.path 中以便导入 config, llm 和 json_processor
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import settings
    from src.standard_matcher.llm import call_llm_for_match
    # 导入 AnalysisJsonProcessor 用于文件拆分
    from src.standard_matcher.json_processor import AnalysisJsonProcessor
except ImportError as e:
    # 同样，在导入时记录错误，而不是打印
    logging.getLogger(__name__).critical(
        f"错误：在 standard_matcher.py 中导入模块失败 - {e}。"
        f"请检查项目结构和 PYTHONPATH。\n"
        f"项目根目录尝试设置为: {project_root}", exc_info=True)
    raise

# --- 全局配置 ---
# 配置日志记录器 (建议在项目入口统一配置)
# logging.basicConfig(level=settings.LOG_LEVEL, # settings 可能尚未完全加载
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # 获取 logger 实例

# 定义可跳过的 model 名称集合 (来自 code_generator.py)
SKIPPABLE_MODELS = {
    "变送器附加规格",
    "传感器附加规格",
    "套管附加规格"
}

# --- 文件路径定义 ---
DEFAULT_INDEX_JSON_PATH = project_root / "libs" / "standard" / "index.json"
INDEX_JSON_PATH = Path(
    getattr(settings, 'INDEX_JSON_PATH', DEFAULT_INDEX_JSON_PATH))

# TEMP_OUTPUT_DIR 将用于存放由 json_processor.py 生成的单个参数文件
TEMP_OUTPUT_DIR = project_root / "data" / "output" / "temp"


# ==============================================================================
# 1. Fetch CSV List (来自 fetch_csvlist.py)
# ==============================================================================

class FetchCsvlist:
    """
    负责根据输入的产品要求和索引文件，获取对应的 CSV 文件列表。
    """
    # --- LLM 相关定义 ---
    SYSTEM_PROMPT = """你是一个智能助手，负责根据提供的产品要求，为每个指定的产品类型从其对应的关键词列表中选择最匹配的一个关键词。
请严格按照以下 JSON 格式返回结果，其中键是产品类型，值是所选的最匹配关键词。如果某个产品类型无法找到合适的匹配，请在该产品类型的值中使用 "默认"。
确保 JSON 格式正确，不要包含任何额外的解释或文本。

输出格式示例:
{
  "product_type_1": "matched_keyword_1",
  "product_type_2": "默认",
  "product_type_3": "matched_keyword_3"
}
"""

    def fetch_csv_lists(self, input_json_path: Path, index_json_path: Path = INDEX_JSON_PATH) -> dict:
        """
        根据输入的产品要求 JSON 和索引文件，通过单次 LLM 调用为所有产品类型匹配最佳关键词，
        并返回包含对应 CSV 文件列表的字典。

        Args:
            input_json_path: 输入的产品要求 JSON 文件路径。
            index_json_path: 包含产品、关键词和 CSV 列表的索引 JSON 文件路径。

        Returns:
            一个字典，键是产品类型（如 'transmitter'），值是对应的 CSV 文件路径列表。
            如果发生错误，则返回空字典。
        """
        result_csv_lists = {}

        # 1. 加载输入 JSON 文件
        try:
            with open(input_json_path, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            logger.info(f"成功加载输入文件: {input_json_path}")
        except FileNotFoundError:
            logger.error(f"输入文件未找到: {input_json_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"解析输入 JSON 文件时出错: {input_json_path} - {e}")
            return {}
        except Exception as e:
            logger.error(f"加载输入文件时发生未知错误: {input_json_path} - {e}")
            return {}

        # 2. 加载索引 JSON 文件
        try:
            with open(index_json_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            logger.info(f"成功加载索引文件: {index_json_path}")
        except FileNotFoundError:
            logger.error(f"索引文件未找到: {index_json_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"解析索引 JSON 文件时出错: {index_json_path} - {e}")
            return {}
        except Exception as e:
            logger.error(f"加载索引文件时发生未知错误: {index_json_path} - {e}")
            return {}

        # 将输入数据格式化为字符串，供 LLM 使用
        try:
            requirements_str = json.dumps(
                input_data, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"将输入数据格式化为 JSON 字符串时出错: {e}")
            return {}

        # 3. 收集所有产品类型的可用关键词
        all_keywords_data = {}
        product_types_to_process = list(index_data.keys())  # 获取所有产品类型

        for product_type in list(product_types_to_process):  # Iterate over a copy
            keywords_dict = index_data.get(product_type, {})
            if not isinstance(keywords_dict, dict):
                logger.warning(f"产品类型 '{product_type}' 在索引文件中的值不是字典，跳过。")
                product_types_to_process.remove(product_type)  # 从待处理列表中移除
                continue

            available_keywords = [k for k in keywords_dict if k != "默认"]
            if available_keywords:
                all_keywords_data[product_type] = available_keywords
            else:
                logger.warning(f"产品类型 '{product_type}' 没有可用的非默认关键词。将直接使用默认值。")
                # 注意：即使没有可用关键词，我们仍然需要为它获取结果（默认值）

        # 4. 构建统一的 User Prompt
        user_prompt = f"""产品要求:
{requirements_str}

请根据以上“产品要求”，为下列每个产品类型选择最匹配的关键词。请从对应产品类型的“可用关键词”列表中选择。如果找不到合适的匹配，请为该产品类型选择 "默认"。

产品类型及其可用关键词:
{json.dumps(all_keywords_data, ensure_ascii=False, indent=2)}

请严格按照指定的 JSON 格式返回所有产品类型及其选定的关键词（匹配到的或 "默认"）。
"""

        # 5. 单次调用 LLM 获取所有匹配结果
        logger.info("调用 LLM 一次性获取所有产品类型的匹配关键词...")
        llm_response = call_llm_for_match(
            self.SYSTEM_PROMPT, user_prompt, expect_json=True)

        # 6. 处理 LLM 响应并确定最终关键词
        matched_keywords = {}
        if isinstance(llm_response, dict) and 'error' not in llm_response:
            logger.info(f"LLM 成功返回 JSON 响应: {llm_response}")
            matched_keywords = llm_response  # 直接使用返回的字典
        elif llm_response is None:
            logger.warning("LLM 调用因客户端未初始化或API密钥问题而跳过。所有产品将使用默认值。")
        elif isinstance(llm_response, dict) and 'error' in llm_response:
            logger.error(
                f"LLM 调用失败。所有产品将使用默认值。错误: {llm_response.get('details', llm_response['error'])}")
        else:
            logger.error(
                f"LLM 返回了非预期的响应类型 ({type(llm_response)}) 或无效的 JSON。所有产品将使用默认值。响应: {llm_response}")

        # 7. 遍历所有需要处理的产品类型，确定最终 CSV 列表
        for product_type in product_types_to_process:
            keywords_dict = index_data.get(product_type)  # Use .get for safety
            if not keywords_dict or not isinstance(keywords_dict, dict):
                logger.warning(f"在索引数据中未找到产品类型 '{product_type}' 或其格式不正确，跳过。")
                continue  # Skip if product_type is not found or not a dict

            available_keywords = all_keywords_data.get(
                product_type, [])  # 获取该类型的可用关键词

            # 从 LLM 响应中获取该产品类型的匹配关键词，如果响应无效或缺失则为 None
            llm_matched_keyword = matched_keywords.get(product_type)

            selected_keyword = "默认"  # 默认为 "默认"

            if isinstance(llm_matched_keyword, str) and llm_matched_keyword != "默认":
                # 验证 LLM 返回的非默认关键词是否有效
                if llm_matched_keyword in available_keywords:
                    selected_keyword = llm_matched_keyword
                    logger.info(
                        f"LLM 为产品 '{product_type}' 成功匹配到有效关键词: '{selected_keyword}'")
                else:
                    logger.warning(
                        f"LLM 为产品 '{product_type}' 返回的关键词 '{llm_matched_keyword}' 不在可用列表中。将使用默认值。")
            elif llm_matched_keyword == "默认":
                logger.info(f"LLM 为产品 '{product_type}' 明确选择了 '默认'。")
                selected_keyword = "默认"
            else:
                # 如果 LLM 响应中没有该产品类型，或值无效，则使用默认值
                if product_type not in matched_keywords:
                    logger.warning(f"LLM 响应中未包含产品类型 '{product_type}'。将使用默认值。")
                else:
                    logger.warning(
                        f"LLM 为产品 '{product_type}' 返回了无效值 '{llm_matched_keyword}'。将使用默认值。")
                selected_keyword = "默认"

            # 获取最终的 CSV 列表
            if selected_keyword in keywords_dict:
                csv_list = keywords_dict[selected_keyword]
                result_csv_lists[product_type] = csv_list
                logger.info(
                    f"产品 '{product_type}' 最终选择关键词 '{selected_keyword}'，对应 CSV 列表: {csv_list}")
            else:
                # 理论上 "默认" 应该总是在 keywords_dict 中
                logger.error(
                    f"严重错误：无法在索引中找到产品 '{product_type}' 的关键词 '{selected_keyword}' (即使是默认值)。跳过此产品。")

        logger.info("所有产品类型处理完毕。")
        return result_csv_lists


# ==============================================================================
# 2. Model Matcher (来自 model_matcher.py)
# ==============================================================================

class ModelMatcher:
    """
    负责将输入参数与标准 CSV 模型库进行匹配的核心类。
    """
    # --- LLM 提示词定义 ---
    SYSTEM_PROMPT = """
你是一个高度精确的智能匹配助手。你的核心任务是将下面列出的“待匹配输入参数的**键值对 (Key-Value Pair)**”与用户提供的“可用标准模型库条目”进行最合适的唯一匹配。

**待匹配输入参数 (Key-Value Pairs):**
{failed_inputs_kv_str}

**重要匹配规则:**
1.  **匹配基础**: 匹配决策必须基于“待匹配输入参数的**键值对**”与“可用标准模型库条目”的**整体语义相关性**。你需要理解输入键值对的含义，并找到在语义上最贴合的标准模型条目（由模型名称、描述和参数定义）。
2.  **优先最佳**: 必须优先匹配关联度最高的输入键值对和标准模型条目。
3.  **严格唯一**: 每个“待匹配输入参数的**键值对**”**必须**匹配到**一个且仅一个**“可用标准模型库条目”。反之，每个“可用标准模型库条目”也**必须**被**一个且仅一个**“待匹配输入参数的键值对”匹配。不允许遗漏任何输入键值对，也不允许重复使用任何标准模型条目。
4.  **完整匹配**: 尽可能为**每一个**“待匹配输入参数的**键值对**”找到一个唯一的“有关键词关联的”匹配模型条目，不能生搬硬套、强行匹配。对于毫无关联性的的输入参数键值对，可以放弃它的匹配。
5.  **输出格式**: **非常重要** - 返回结果**必须**严格按照以下 JSON 格式。JSON 的键**必须**是原始输入参数的**键 (Key)**，值是匹配到的标准模型库条目的**模型名称 (model)**。即使匹配是基于模型条目的详细信息（模型、描述、参数），最终返回的也**只是模型名称**。
    示例: `{{"输入参数键1": "匹配到的模型名称A", "输入参数键2": "匹配到的模型名称B", ...}}`
"""

    USER_PROMPT_TEMPLATE = """
这是可用的标准模型库条目列表（包含模型名称、描述和参数）。请根据你在系统提示中看到的“待匹配输入参数的键值对”列表和所有匹配规则，为系统提示中的**每一个**输入键值对，从下面的列表中选择最合适的、唯一的匹配标准模型条目。

**可用标准模型库条目 (模型名称、描述、参数):**
{available_models_details_str}

请严格按照系统提示中要求的 JSON 格式返回所有匹配结果。**再次强调**，JSON 的键必须是原始输入参数的**键 (Key)**，值是所选标准模型条目对应的**模型名称 (model)**。确保遵循所有规则，特别是优先匹配、唯一匹配和完整匹配的要求。
```json
{{{{
  "输入参数键1": "匹配到的模型名称A",
  "输入参数键2": "匹配到的模型名称B",
  ...
}}}}
```
"""

    def __init__(self, csv_list_map: Dict[str, List[str]], input_json_path: str):
        """
        初始化 ModelMatcher。

        Args:
            csv_list_map (Dict[str, List[str]]): CSV 文件路径列表的映射，键为类别，值为路径列表。
                                                 例如: {"transmitter": ["path/to/tx1.csv", ...]}
            input_json_path (str): 输入参数 JSON 文件的路径。
        """
        if not THEFUZZ_AVAILABLE:
            logger.error("模糊匹配库 'thefuzz' 不可用，无法继续。请安装。")
            raise ImportError("缺少 thefuzz 库")

        self.csv_list_map = csv_list_map
        self.input_json_path = Path(input_json_path)
        self.fuzzy_threshold = 0.9  # 模糊匹配相似度阈值

        self.input_data: Dict[str, str] = self._load_input_json()
        # 结构: {'model_name': {'rows': [row_dict, ...], 'used': False}}
        self.csv_data: Dict[str, Dict[str, Any]] = self._load_csv_data()

        # 最终匹配结果: {"input_key: value": [matched_rows]}
        self.matched_results: Dict[str, List[Dict[str, Any]]] = {}
        self.unmatched_inputs: Dict[str, str] = {}  # 记录完全无法匹配的输入

        logger.info(f"ModelMatcher 初始化完成。加载了 {len(self.input_data)} 个输入参数，"
                    f"{len(self.csv_data)} 个标准模型组。")

    def _load_input_json(self) -> Dict[str, str]:
        """加载输入 JSON 文件。"""
        data: Dict[str, str] = {}
        try:
            with open(self.input_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error(f"输入 JSON 文件未找到: {self.input_json_path}")
            raise  # 重新引发异常，让调用者处理
        except json.JSONDecodeError as e:
            logger.error(f"无法解析输入 JSON 文件: {self.input_json_path} - {e}")
            raise  # 重新引发异常
        except Exception as e:
            logger.error(f"加载输入 JSON 文件 {self.input_json_path} 时发生未知错误: {e}", exc_info=True)
            raise  # 重新引发异常

        # 在 try 块之外检查类型，因为此时文件已成功加载和解析
        if not isinstance(data, dict):
            logger.error(f"输入 JSON 文件 {self.input_json_path} 顶层不是一个字典。实际类型: {type(data)}")
            raise ValueError(f"输入 JSON 文件 {self.input_json_path} 顶层必须是一个字典。")

        logger.info(f"成功从 {self.input_json_path} 加载输入数据。")
        return data

    def _load_csv_data(self) -> Dict[str, Dict[str, Any]]:
        """加载并处理所有 CSV 文件，按 'model' 列分组。"""
        all_csv_data = {}
        # 更新所需的列
        required_columns = {'model', 'code',
                            'description', 'param', 'is_default'}

        for category, paths in self.csv_list_map.items():
            logger.debug(f"开始加载类别 '{category}' 的 CSV 文件: {paths}")
            for csv_path_str in paths:
                csv_path = Path(csv_path_str)
                if not csv_path.is_file():
                    logger.warning(f"CSV 文件未找到，跳过: {csv_path}")
                    continue
                try:
                    # 尝试不同的编码和分隔符读取 CSV
                    df = None
                    try:
                        # 尝试 UTF-8, 逗号分隔
                        df = pd.read_csv(csv_path, dtype=str).fillna('')
                    except UnicodeDecodeError:
                        logger.warning(f"使用 utf-8 读取 {csv_path} 失败，尝试 gbk...")
                        try:
                            # 尝试 GBK, 逗号分隔
                            df = pd.read_csv(
                                csv_path, encoding='gbk', dtype=str).fillna('')
                        except Exception as e_gbk:
                            logger.warning(
                                f"使用 gbk 读取 {csv_path} 失败: {e_gbk}。尝试分号分隔符...")
                    except Exception as e_utf8_comma:
                        logger.warning(
                            f"使用 utf-8 和逗号分隔符读取 {csv_path} 失败: {e_utf8_comma}。尝试分号分隔符...")

                    # 如果逗号分隔失败，尝试分号分隔符
                    if df is None or not required_columns.issubset(df.columns):
                        logger.warning(f"尝试使用分号分隔符读取 {csv_path}...")
                        try:
                            # 尝试 UTF-8, 分号分隔
                            df_semi = pd.read_csv(
                                csv_path, sep=';', dtype=str).fillna('')
                            if required_columns.issubset(df_semi.columns):
                                df = df_semi
                            else:
                                logger.warning(
                                    f"使用 utf-8 和分号分隔符读取 {csv_path} 仍缺少列。尝试 GBK...")
                        except UnicodeDecodeError:
                            logger.warning(
                                f"使用 utf-8 和分号分隔符读取 {csv_path} 失败，尝试 gbk...")
                        except Exception as e_utf8_semi:
                            logger.warning(
                                f"使用 utf-8 和分号分隔符读取 {csv_path} 失败: {e_utf8_semi}。尝试 GBK...")

                        if df is None or not required_columns.issubset(df.columns):
                            try:
                                # 尝试 GBK, 分号分隔
                                df_semi_gbk = pd.read_csv(
                                    csv_path, sep=';', encoding='gbk', dtype=str).fillna('')
                                if required_columns.issubset(df_semi_gbk.columns):
                                    df = df_semi_gbk
                                else:
                                    logger.error(
                                        f"尝试多种编码和分隔符后，CSV 文件 {csv_path} 仍缺少必需的列。")
                                    continue  # 跳过此文件
                            except Exception as e_gbk_semi:
                                logger.error(
                                    f"尝试多种编码和分隔符后，读取 CSV 文件 {csv_path} 失败: {e_gbk_semi}")
                                continue  # 跳过此文件

                    # 检查必需的列 (再次确认)
                    if df is None or not required_columns.issubset(df.columns):
                        logger.error(f"最终未能成功加载或文件 {csv_path} 缺少必需的列。"
                                     f"需要: {required_columns}, 实际: {set(df.columns) if df is not None else '加载失败'}")
                        continue  # 跳过这个文件

                    # 按 'model' 分组
                    grouped = df.groupby('model')
                    for model_name, group_df in grouped:
                        # 跳过 model 为空或 NaN 的情况
                        if not model_name or pd.isna(model_name):
                            logger.warning(
                                f"在 {csv_path} 中发现空的 model 名称，跳过该组。")
                            continue

                        model_name_str = str(
                            model_name).strip()  # 确保是字符串并去除首尾空格
                        if not model_name_str:
                            logger.warning(
                                f"在 {csv_path} 中发现处理后为空的 model 名称，跳过该组。")
                            continue

                        rows = group_df.to_dict('records')
                        if model_name_str in all_csv_data:
                            # 如果模型已存在（可能来自不同文件），合并行，但要小心重复
                            # 这里简单地追加，如果需要更复杂的去重逻辑可以在此添加
                            logger.debug(f"模型 '{model_name_str}' 在多个 CSV 文件中找到。"
                                         # 改为 debug 级别
                                         f"将合并来自 {csv_path} 的行。")
                            all_csv_data[model_name_str]['rows'].extend(rows)
                        else:
                            all_csv_data[model_name_str] = {
                                'rows': rows, 'used': False}
                    logger.debug(f"成功处理 CSV 文件: {csv_path}")

                except pd.errors.EmptyDataError:
                    logger.warning(f"CSV 文件为空，跳过: {csv_path}")
                except Exception as e:
                    logger.error(
                        f"处理 CSV 文件 {csv_path} 时出错: {e}", exc_info=True)

        logger.info(f"CSV 数据加载完成。共加载 {len(all_csv_data)} 个标准模型组。")
        return all_csv_data

    def _get_combined_string(self, data: Dict[str, Any] | Tuple[str, str]) -> str:
        """
        根据数据类型返回用于匹配的字符串。
        对于输入元组，返回键。
        对于 CSV 行字典，返回 'model' 值。
        """
        if isinstance(data, tuple):  # 输入键值对
            key, _ = data  # 只需要键
            return key.strip()  # 去除键的首尾空格
        elif isinstance(data, dict):  # CSV 行
            return data.get('model', '').strip()  # 去除模型名称的首尾空格
        return ""

    def _fuzzy_match(self) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Tuple[str, str]]]:
        """
        执行模糊匹配。

        Returns:
            Tuple[Dict[str, List[Dict[str, Any]]], List[Tuple[str, str]]]:
                - fuzzy_matches: 成功匹配的结果 {"'input_key': 'input_value'": [matched_rows]}
                - failed_fuzzy_matches: 匹配失败或分数过低的输入项 [(input_key, input_value), ...]
        """
        fuzzy_matches = {}
        failed_fuzzy_matches = []
        # 创建可用模型的副本，用于迭代和修改 'used' 状态
        available_models = {name: data for name,
                            data in self.csv_data.items() if not data['used']}
        # 将输入数据转换为元组列表 [(key, value), ...]
        input_items = list(self.input_data.items())

        logger.info("开始模糊匹配...")
        # 优先匹配高分项
        # 使用 process.extract 获取所有输入项相对于所有可用模型名称的匹配度
        # 注意：process.extract 返回 [(choice, score, key), ...]
        # 我们需要将输入项作为查询，模型名称作为选项

        # 构建选项列表（模型名称）
        model_name_choices = list(available_models.keys())
        if not model_name_choices:
            logger.warning("没有可用的模型名称进行模糊匹配。")
            failed_fuzzy_matches = input_items  # 所有输入都失败
            return fuzzy_matches, failed_fuzzy_matches

        # 存储每个输入项的最佳匹配及其分数
        # { (input_key, input_value): (best_model_name, best_score) }
        best_matches_for_inputs = {}

        for input_key, input_value in input_items:
            input_key_str = input_key.strip()  # 用于匹配的键
            # 使用 process.extractOne 找到最佳匹配
            # scorer 可以选择 fuzz.ratio, fuzz.partial_ratio, fuzz.token_sort_ratio 等
            best_match = process.extractOne(
                input_key_str, model_name_choices, scorer=fuzz.token_sort_ratio)

            if best_match:
                best_model_name, best_score = best_match
                best_matches_for_inputs[(input_key, input_value)] = (
                    best_model_name, best_score)
            else:
                best_matches_for_inputs[(input_key, input_value)] = (
                    None, -1)  # 没有找到匹配

        # 排序：优先处理分数高且模型唯一的匹配
        # 1. 按分数降序排序
        sorted_inputs = sorted(best_matches_for_inputs.items(
        ), key=lambda item: item[1][1], reverse=True)

        # 2. 迭代处理，标记已使用的模型和输入
        processed_inputs = set()
        used_models = set()

        for (input_key, input_value), (best_model_name, best_score) in sorted_inputs:
            input_tuple = (input_key, input_value)
            # 跳过已处理的输入或分数低于阈值的
            if input_tuple in processed_inputs or best_score < self.fuzzy_threshold * 100:
                continue

            # 检查模型是否已被使用
            if best_model_name and best_model_name not in used_models:
                # 成功匹配
                match_key = f"'{input_key}': '{input_value}'"  # 输出键格式
                matched_rows = self.csv_data[best_model_name]['rows']
                fuzzy_matches[match_key] = matched_rows
                self.csv_data[best_model_name]['used'] = True  # 更新原始数据
                used_models.add(best_model_name)
                processed_inputs.add(input_tuple)
                logger.info(
                    f"模糊匹配成功: 输入键 '{input_key}' -> 模型 '{best_model_name}' (分数: {best_score})")
            # else: 模型已被使用或无效，此输入项暂时无法通过模糊匹配解决

        # 收集所有未被处理的输入项
        failed_fuzzy_matches = [
            item for item in input_items if item not in processed_inputs]

        logger.info(
            f"模糊匹配完成。成功 {len(fuzzy_matches)} 项，失败 {len(failed_fuzzy_matches)} 项。")
        return fuzzy_matches, failed_fuzzy_matches

    def _llm_match(self, failed_inputs: List[Tuple[str, str]],
                   available_models: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        使用 LLM 对模糊匹配失败的项进行匹配。

        Args:
            failed_inputs: 模糊匹配失败的输入项列表 [(key, value), ...]。
            available_models: 仍然可用的模型组 {'model_name': {'rows': [...], 'used': False}, ...}。

        Returns:
            Dict[str, List[Dict[str, Any]]]: LLM 匹配成功的结果 {"'input_key': 'input_value'": [matched_rows]}。
        """
        llm_matches = {}
        if not failed_inputs:
            logger.info("没有模糊匹配失败的项，跳过 LLM 匹配。")
            return llm_matches
        if not available_models:
            logger.warning("没有可用的标准模型库条目，无法进行 LLM 匹配。")
            self.unmatched_inputs.update(
                {k: v for k, v in failed_inputs})  # 记录为无法匹配
            return llm_matches

        logger.info(f"开始 LLM 匹配，处理 {len(failed_inputs)} 个失败项...")

        # 准备提示词内容
        # 1. 准备待匹配输入键值对字符串
        failed_inputs_kv_str = "\n".join(
            # 使用 key: value 格式, 去除空格
            [f"- {k.strip()}: {v.strip()}" for k, v in failed_inputs])

        # 2. 准备可用模型详情字符串 (model, description, param)
        available_models_details_list = []
        model_name_to_details_map = {}  # 用于快速查找模型详情
        for model_name, model_data in available_models.items():
            model_name_clean = model_name.strip()
            # 为每个 model_name 创建一个条目，包含其下所有行的相关信息
            model_details_parts = [f"模型名称 (Model Name): {model_name_clean}"]
            row_details = []
            for i, row in enumerate(model_data.get('rows', [])):
                desc = row.get('description', 'N/A').strip()
                param = row.get('param', 'N/A').strip()
                # 附带行信息以帮助 LLM 理解，但匹配目标仍是 model_name
                row_details.append(f"  - 行 {i+1}: 描述='{desc}', 参数='{param}'")
            model_details_parts.extend(row_details)
            detail_str = "\n".join(model_details_parts)
            available_models_details_list.append(detail_str)
            model_name_to_details_map[model_name_clean] = model_data  # 存储原始数据

        available_models_details_str = "\n\n".join(
            available_models_details_list)
        # 注意：如果可用模型条目非常多，这个字符串可能会变得很长，可能需要考虑截断或分页策略

        # 格式化 System Prompt 和 User Prompt
        system_prompt_formatted = self.SYSTEM_PROMPT.format(
            failed_inputs_kv_str=failed_inputs_kv_str)
        user_prompt_formatted = self.USER_PROMPT_TEMPLATE.format(
            available_models_details_str=available_models_details_str)

        logger.debug(
            # 增加预览长度
            f"格式化后的 System Prompt (部分): {system_prompt_formatted[:500]}...")
        logger.debug(
            # 增加预览长度
            f"格式化后的 User Prompt (部分): {user_prompt_formatted[:500]}...")

        # 调用 LLM 前添加延时
        logger.info("模型匹配：等待 5 秒以避免 LLM 速率限制...")
        time.sleep(5)
        llm_response = call_llm_for_match(
            system_prompt_formatted, user_prompt_formatted, expect_json=True)

        # --- LLM 响应处理部分 ---
        if not llm_response or isinstance(llm_response, str) or llm_response.get("error"):
            logger.error(f"LLM 调用失败或返回错误: {llm_response}")
            self.unmatched_inputs.update({k: v for k, v in failed_inputs})
            return llm_matches

        try:
            if not isinstance(llm_response, dict):
                logger.error(f"LLM 响应不是预期的字典格式: {llm_response}")
                self.unmatched_inputs.update({k: v for k, v in failed_inputs})
                return llm_matches

            processed_inputs_tuples = set()  # 记录已处理的 (key, value) 元组
            used_models_by_llm = set()  # 记录 LLM 在此轮匹配中使用的模型

            # 验证 LLM 返回的匹配是否满足唯一性要求
            llm_model_usage_count = {}
            valid_llm_matches = {}  # 存储初步验证通过的匹配 {input_key: model_name}

            for input_key_from_llm, matched_model_name in llm_response.items():
                input_key_clean = input_key_from_llm.strip()
                model_name_clean = matched_model_name.strip()

                # 检查模型是否存在于可用模型中
                if model_name_clean in model_name_to_details_map:
                    valid_llm_matches[input_key_clean] = model_name_clean
                    llm_model_usage_count[model_name_clean] = llm_model_usage_count.get(
                        model_name_clean, 0) + 1
                else:
                    logger.warning(
                        f"LLM 为输入 '{input_key_clean}' 返回了无效或不可用的模型名称 '{model_name_clean}'，忽略此匹配。")

            # 检查是否有模型被重复使用
            duplicate_models = {model: count for model,
                                count in llm_model_usage_count.items() if count > 1}
            if duplicate_models:
                logger.error(f"LLM 违反唯一性规则，重复使用了以下模型: {duplicate_models}")
                # 如何处理？可以选择全部放弃，或尝试保留第一个？这里选择全部放弃重复使用的模型的匹配
                keys_to_remove = [
                    key for key, model in valid_llm_matches.items() if model in duplicate_models]
                for key in keys_to_remove:
                    logger.warning(
                        f"因模型 '{valid_llm_matches[key]}' 被重复使用，放弃输入 '{key}' 的 LLM 匹配。")
                    del valid_llm_matches[key]  # 从有效匹配中移除

            # 处理最终确认的有效且唯一的 LLM 匹配
            for input_key_clean, model_name_clean in valid_llm_matches.items():
                # 从原始 failed_inputs 中找到对应的 (key, value) 元组
                original_input_tuple = None
                for k, v in failed_inputs:
                    if k.strip() == input_key_clean:  # 匹配清理后的键
                        original_input_tuple = (k, v)
                        break

                if original_input_tuple:
                    # 使用找到的原始 key 和 value 构建最终输出的键
                    final_match_key = f"'{original_input_tuple[0]}': '{original_input_tuple[1]}'"
                    # 获取原始模型数据（注意：available_models 的键可能带空格）
                    # 需要找到原始带空格的 key
                    original_model_key = None
                    for m_key in available_models.keys():
                        if m_key.strip() == model_name_clean:
                            original_model_key = m_key
                            break

                    if original_model_key:
                        matched_rows = self.csv_data[original_model_key]['rows']
                        llm_matches[final_match_key] = matched_rows
                        # 标记整个模型为已使用
                        self.csv_data[original_model_key]['used'] = True
                        used_models_by_llm.add(
                            original_model_key)  # 使用原始 key 标记
                        processed_inputs_tuples.add(
                            original_input_tuple)  # 标记原始元组为已处理
                        logger.info(
                            f"LLM 匹配成功: {final_match_key} -> 模型 '{model_name_clean}'")
                    else:
                        logger.error(f"内部错误：无法找到模型 '{model_name_clean}' 的原始键。")

                else:
                    # 这种情况理论上不应发生
                    logger.warning(
                        f"LLM 返回了无法在失败列表中找到其原始元组的输入键: '{input_key_clean}'")

            # 将 LLM 未能成功匹配或处理的项记录为无法匹配
            remaining_failed = [
                item for item in failed_inputs if item not in processed_inputs_tuples]
            self.unmatched_inputs.update({k: v for k, v in remaining_failed})
            if remaining_failed:
                logger.warning(f"{len(remaining_failed)} 个输入项在 LLM 匹配后仍未匹配。")

        except Exception as e:
            logger.error(f"处理 LLM 响应时出错: {e}", exc_info=True)
            # 将所有失败的输入记录为无法匹配
            self.unmatched_inputs.update({k: v for k, v in failed_inputs})
            return {}  # 返回空字典表示处理失败

        logger.info(f"LLM 匹配完成。成功匹配 {len(llm_matches)} 项。")
        return llm_matches

    def match(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        执行完整的匹配流程：模糊匹配 -> LLM 匹配。

        Returns:
            Dict[str, List[Dict[str, Any]]]: 最终的匹配结果 {"'input_key': 'input_value'": [matched_rows]}。
        """
        # 重置状态以允许重新运行 match
        self.matched_results = {}
        self.unmatched_inputs = {}
        for model_data in self.csv_data.values():
            model_data['used'] = False

        # 1. 模糊匹配
        fuzzy_matches, failed_fuzzy_matches = self._fuzzy_match()
        self.matched_results.update(fuzzy_matches)

        # 2. LLM 匹配
        remaining_available_models = {
            name: data for name, data in self.csv_data.items() if not data['used']}
        llm_matches = self._llm_match(
            failed_fuzzy_matches, remaining_available_models)
        self.matched_results.update(llm_matches)

        logger.info(f"所有匹配流程完成。总共匹配 {len(self.matched_results)} 项。")
        if self.unmatched_inputs:
            # 修改日志记录，逐条输出未匹配的键
            logger.warning(f"有 {len(self.unmatched_inputs)} 个输入项最终未能匹配。")
            for unmatched_key, unmatched_value in self.unmatched_inputs.items():
                logger.warning(
                    f"  - 未匹配输入键: '{unmatched_key}' (值: '{unmatched_value}')")

        # 记录未被匹配的标准模型组
        unmatched_models = [name for name,
                            data in self.csv_data.items() if not data['used']]
        if unmatched_models:
            logger.warning(f"有 {len(unmatched_models)} 个标准模型组未被任何输入参数匹配:")
            # 只记录前 10 个未匹配的模型以避免日志过长
            for i, model_name in enumerate(unmatched_models):
                if i < 10:
                    logger.warning(f"  - 未匹配模型组: '{model_name}'")
                elif i == 10:
                    logger.warning(
                        f"  - ... (还有 {len(unmatched_models) - 10} 个未显示)")
                    break
        # else: # 如果不需要在所有模型都匹配时输出信息，可以注释掉这部分
        #     logger.info("所有标准模型组都已成功匹配或尝试匹配。")

        return self.matched_results


# ==============================================================================
# 3. Code Selector (来自 code_selector.py)
# ==============================================================================

class CodeSelector:
    """
    负责从候选 CSV 行列表中为每个输入参数选择最佳代码行。
    """
    # --- LLM 提示词定义 ---
    SELECTOR_SYSTEM_PROMPT = """
你是一个精确的代码选择  助手。你的任务是：对于用户提供的每个“输入参数”（键值对形式），从其对应的“候选标准行列表”中，选择**唯一**的最匹配的那一行。

**重要规则:**
1.  **选择基础**: 你的选择必须基于“输入参数的键和值整体”与“候选行提供的 **description 和 param 字段内容**”之间的**语义相似度**。你需要理解输入参数的含义，并找到语义上最贴合的那一行候选标准代码（基于其 description 和 param）。
2.  **唯一性**: 对于每个输入参数，你必须从其候选列表中选择**且仅选择一个**最佳匹配行。
3.  **输出格式**: 必须以 JSON 格式返回选择结果。JSON 的键是原始的输入参数字符串（格式："key: value"），值是你选择的最佳匹配行的**索引号 (从 0 开始)**。
    示例: `{"输入参数键1: 输入参数值1": 0, "输入参数键2: 输入参数值2": 2}`
4.  **完整性**: 一定要为每一个提供的“输入参数”都选择一个最优候选行索引，不允许不选择。
5.  **注意转义**: 请注意引号、反斜杠等特殊字符必须正确转义，如：“1/2" NPT (F)”应该正确转义成“1/2\" NPT (F)”，避免输出JSON时出现格式错误。返回参数时注意不要增加其他的符号，对于字符处理使用半角。
"""

    SELECTOR_USER_PROMPT_TEMPLATE = """
请为以下每个“输入参数”，从其对应的“候选标准行列表”中选择唯一最匹配的一行。请仔细考虑每个候选行的完整信息。

**待选择项:**
{items_to_select_str}

请严格按照以下 JSON 格式返回所有输入参数的选择结果（值为选中行的索引号）:
```json
{{
  "输入参数键1: 输入参数值1": <选中行的索引号0>,
  "输入参数键2: 输入参数值2": <选中行的索引号1>,
  ...
}}
```
确保每个输入参数都有一个对应的选中行索引。
"""

    def __init__(self, matched_models_dict: Dict[str, List[Dict[str, Any]]]):
        """
        初始化 CodeSelector。

        Args:
            matched_models_dict (Dict[str, List[Dict[str, Any]]]):
                模型匹配器输出的字典，格式为 {"'input_key': 'value'": [candidate_row1, ...]}。
        """
        if not THEFUZZ_AVAILABLE:
            logger.error("模糊匹配库 'thefuzz' 不可用，无法继续。请安装。")
            raise ImportError("缺少 thefuzz 库")

        self.matched_models_dict = matched_models_dict
        self.fuzzy_select_threshold = 0.8  # 模糊选择相似度阈值

        # 最终选择结果: {original_input_key_str: selected_row_dict}
        self.selected_codes: Dict[str, Dict[str, Any]] = {}
        # 模糊选择失败的项: [(input_str, candidate_rows), ...]
        self.failed_fuzzy_selection: List[Tuple[str,
                                                List[Dict[str, Any]]]] = []

        logger.info(
            f"CodeSelector 初始化完成。待处理 {len(self.matched_models_dict)} 个输入参数。")

    def _row_to_string(self, row_dict: Dict[str, Any]) -> str:
        """将 CSV 行字典的关键描述信息转换为用于匹配的单一字符串。"""
        # 只使用 description 和 param 进行匹配
        desc = row_dict.get('description', '')
        param = row_dict.get('param', '')
        # 确保两者之间有空格分隔，除非其中一个为空
        return f"{desc} {param}".strip()

    def _fuzzy_select(self):
        """执行模糊选择逻辑。"""
        logger.info("开始模糊选择...")
        self.selected_codes = {}
        self.failed_fuzzy_selection = []

        for input_str, candidate_rows in self.matched_models_dict.items():
            if not candidate_rows:
                logger.warning(f"输入 '{input_str}' 没有候选行，跳过选择。")
                continue

            # 如果只有一个候选行，直接选择
            if len(candidate_rows) == 1:
                selected_row_dict = candidate_rows[0].copy()  # 使用副本以防修改原始数据
                original_code = selected_row_dict.get('code', '')

                # 如果 code 包含 %int%，尝试从 input_str 提取数字并替换
                if isinstance(original_code, str) and '%int%' in original_code:
                    # 尝试从 input_str (格式 "'key': 'value'") 提取 value 中的内容
                    # 捕获引号内的所有内容 (包括空)
                    match = re.search(r":\s*'([^']*)'", input_str)
                    if match:
                        value_part = match.group(1).strip()  # 获取值并去除首尾空格
                        # 从捕获的内容中提取所有数字，去除空格和其他非数字字符
                        extracted_digits = re.sub(r'\D', '', value_part)
                        if extracted_digits:  # 确保提取到了数字
                            # 替换 %int% 部分
                            new_code = original_code.replace(
                                '%int%', extracted_digits)
                            selected_row_dict['code'] = new_code
                            logger.debug(
                                f"为输入 '{input_str}' 的唯一候选行替换 %int% 得到 code: {new_code}")
                        else:
                            logger.warning(
                                f"唯一候选行的 code '{original_code}' 包含 '%int%'，但在输入 '{input_str}' 的值 '{value_part}' 中未能提取到有效数字，保留原样。")
                    else:
                        logger.warning(
                            f"唯一候选行的 code '{original_code}' 包含 '%int%'，但在输入 '{input_str}' 中未能按预期格式提取到值，保留原样。")

                # 使用原始 input_str 作为 key 存储选择结果（可能是修改后的，也可能是原始的）
                self.selected_codes[input_str] = selected_row_dict
                logger.info(
                    f"模糊选择成功 (唯一候选): 键值对 '{input_str}' -> 最终匹配结果 {selected_row_dict}")
                continue

            best_match_row = None
            best_score = -1
            best_row_index = -1

            # 将输入字符串与每个候选行进行比较
            # 提取输入字符串中的 value 部分用于比较
            input_value_match = re.search(r":\s*'([^']*)'", input_str)
            input_value_for_match = input_value_match.group(1).strip(
            ) if input_value_match else input_str  # Fallback to full string

            for index, row in enumerate(candidate_rows):
                row_str = self._row_to_string(row)
                # 使用 token_sort_ratio 忽略词序，比较输入值与行的描述/参数
                score = fuzz.token_sort_ratio(input_value_for_match, row_str)

                if score > best_score:
                    best_score = score
                    best_match_row = row
                    best_row_index = index

            # 判断是否达到阈值
            if best_match_row and best_score >= self.fuzzy_select_threshold * 100:
                # 使用原始 input_str 作为 key
                selected_row_dict = best_match_row.copy()  # 使用副本
                # 检查是否需要处理 %int% (即使不是唯一候选)
                original_code = selected_row_dict.get('code', '')
                if isinstance(original_code, str) and '%int%' in original_code:
                    match = re.search(r":\s*'([^']*)'", input_str)
                    if match:
                        value_part = match.group(1).strip()
                        extracted_digits = re.sub(r'\D', '', value_part)
                        if extracted_digits:
                            new_code = original_code.replace(
                                '%int%', extracted_digits)
                            selected_row_dict['code'] = new_code
                            logger.debug(
                                f"为输入 '{input_str}' 的模糊匹配结果替换 %int% 得到 code: {new_code}")
                        # else: 保留原样，日志在唯一候选时已记录

                self.selected_codes[input_str] = selected_row_dict
                logger.info(
                    f"模糊选择成功: 键值对 '{input_str}' -> 匹配结果 {selected_row_dict} (分数: {best_score})")
            else:
                # 记录失败项以供 LLM 处理
                self.failed_fuzzy_selection.append((input_str, candidate_rows))
                logger.debug(
                    f"模糊选择失败或分数低: 输入 '{input_str}' (最高分: {best_score}, 最佳候选 code: {best_match_row.get('code', 'N/A') if best_match_row else 'N/A'})")

        logger.info(
            f"模糊选择完成。成功 {len(self.selected_codes)} 项，失败 {len(self.failed_fuzzy_selection)} 项。")

    def _llm_select(self) -> Dict[str, Dict[str, Any]]:
        """
        使用 LLM 对模糊选择失败的项进行选择。

        Returns:
            Dict[str, Dict[str, Any]]: LLM 选择成功的结果 {original_input_key_str: selected_row_dict}。
        """
        llm_selected_codes = {}
        if not self.failed_fuzzy_selection:
            logger.info("没有模糊选择失败的项，跳过 LLM 选择。")
            return llm_selected_codes

        total_failed_count = len(self.failed_fuzzy_selection)
        logger.info(f"开始 LLM 选择，处理 {total_failed_count} 个模糊选择失败项...")

        batch_size = 5  # 每批处理的最大数量 (可调整)
        processed_inputs_in_llm_responses = set()  # 记录 LLM 实际返回了结果的输入字符串

        for i in range(0, total_failed_count, batch_size):
            batch = self.failed_fuzzy_selection[i:i + batch_size]
            batch_number = (i // batch_size) + 1
            total_batches = (total_failed_count + batch_size - 1) // batch_size
            logger.info(
                f"处理 LLM 选择批次 {batch_number}/{total_batches} (共 {len(batch)} 项)...")

            # 准备当前批次的提示词内容
            items_to_select_str_parts = []
            batch_input_mapping = {}  # 存储当前批次的 input_str -> candidate_rows 映射
            for input_str, candidate_rows in batch:
                batch_input_mapping[input_str] = candidate_rows
                item_str = f"输入参数: \"{input_str}\"\n候选行列表:\n"
                for idx, row in enumerate(candidate_rows):
                    # 获取完整的 description 和 param
                    desc = row.get('description', '无描述')
                    param = row.get('param', '无参数')  # 或者根据实际情况决定默认值
                    # 更新 item_str 的格式，只包含索引、description 和 param
                    item_str += f"  {idx}: description='{desc}', param='{param}'\n"
                items_to_select_str_parts.append(item_str)

            items_to_select_str = "\n---\n".join(items_to_select_str_parts)
            user_prompt = self.SELECTOR_USER_PROMPT_TEMPLATE.format(
                items_to_select_str=items_to_select_str)

            # 调用 LLM 处理当前批次前添加延时
            logger.info(f"代码选择 (批次 {batch_number}/{total_batches})：等待 5 秒以避免 LLM 速率限制...")
            time.sleep(5)
            llm_response = call_llm_for_match(
                self.SELECTOR_SYSTEM_PROMPT, user_prompt, expect_json=True)

            # 处理当前批次的 LLM 响应
            if not llm_response or isinstance(llm_response, str) or llm_response.get("error"):
                error_msg = f"批次 {batch_number}/{total_batches} 的 LLM 调用失败或返回错误: {llm_response}"
                logger.error(error_msg)
                # 决定是否继续下一批？这里选择抛出异常，因为可能影响最终结果
                raise ValueError(error_msg + " - 无法继续选择。")

            # 成功获取响应，处理批内各项结果
            try:
                if not isinstance(llm_response, dict):
                    error_msg = f"批次 {batch_number}/{total_batches} 的 LLM 响应不是预期的字典格式: {llm_response}"
                    logger.error(error_msg)
                    raise ValueError(error_msg + " - 无法继续选择。")

                # 遍历 LLM 返回的当前批次的结果
                batch_processed_internal_keys_in_response = set()  # 用于存储当前批次中，LLM已响应并转换为内部格式的键
                for llm_key_str, selected_index in llm_response.items():  # llm_key_str 是 LLM 返回的 "key: value" 格式
                    # 解析 LLM 返回的键 ("key: value") 以便转换为内部格式 ("'key': 'value'")
                    match_llm_key = re.match(r"^(.*?):\s*(.*)$", llm_key_str)
                    if not match_llm_key:
                        logger.warning(
                            f"LLM 在批次 {batch_number} 返回了无法解析的键格式: '{llm_key_str}'，跳过此项。")
                        continue

                    parsed_llm_key_part_raw = match_llm_key.group(1)
                    parsed_llm_value_part_raw = match_llm_key.group(2)

                    # 清理从LLM键中解析出的键和值部分，去除可能存在的多余单引号
                    cleaned_key_part = parsed_llm_key_part_raw.strip("'")
                    cleaned_value_part = parsed_llm_value_part_raw.strip("'")

                    # 转换为内部使用的键格式："'key': 'value'"
                    internal_key_str = f"'{cleaned_key_part}': '{cleaned_value_part}'"

                    batch_processed_internal_keys_in_response.add(
                        internal_key_str)
                    processed_inputs_in_llm_responses.add(
                        internal_key_str)  # 加入全局已处理集合（使用内部键格式）

                    # 查找原始候选列表 (在当前批次的映射中查找，使用内部键格式)
                    original_candidates = batch_input_mapping.get(
                        internal_key_str)

                    if original_candidates is None:
                        # 此警告现在表示LLM返回了一个在当前批次中（即使转换格式后）也找不到的键
                        # 在日志中同时显示原始解析部分和清理后部分，方便调试
                        logger.warning(
                            f"LLM 在批次 {batch_number} 返回了与批次输入不符的键: "
                            f"LLM原始键='{llm_key_str}' -> "
                            f"解析前(key='{parsed_llm_key_part_raw}', value='{parsed_llm_value_part_raw}') -> "
                            f"解析后(key='{cleaned_key_part}', value='{cleaned_value_part}') -> "
                            f"最终内部键='{internal_key_str}'，跳过此项。"
                        )
                        continue

                    # 验证索引并选择
                    try:
                        selected_index = int(selected_index)
                        if 0 <= selected_index < len(original_candidates):
                            selected_row = original_candidates[selected_index]
                            selected_row_dict = selected_row.copy()  # 使用副本

                            # 检查并处理 %int%
                            original_code = selected_row_dict.get('code', '')
                            if isinstance(original_code, str) and '%int%' in original_code:
                                # 注意：这里的 input_str 用于提取 %int% 的值，应使用 internal_key_str
                                match_percent_int = re.search(
                                    r":\s*'([^']*)'", internal_key_str)
                                if match_percent_int:
                                    value_part = match_percent_int.group(
                                        1).strip()
                                    extracted_digits = re.sub(
                                        r'\D', '', value_part)
                                    if extracted_digits:
                                        new_code = original_code.replace(
                                            '%int%', extracted_digits)
                                        selected_row_dict['code'] = new_code
                                        logger.debug(
                                            f"为输入 '{internal_key_str}' 的 LLM 选择结果替换 %int% 得到 code: {new_code}")
                                    # else: 保留原样

                            # 使用内部键格式存储结果
                            llm_selected_codes[internal_key_str] = selected_row_dict
                            logger.info(
                                f"LLM 选择成功 (批次 {batch_number}): 键值对 '{internal_key_str}' -> 匹配结果 {selected_row_dict} (选中索引: {selected_index})")
                        else:
                            error_msg = f"LLM 在批次 {batch_number} 为输入 '{internal_key_str}' (LLM原始键: '{llm_key_str}') 返回了无效索引: {selected_index} (候选数量: {len(original_candidates)})。"
                            logger.error(error_msg)
                            raise ValueError(error_msg + " - 无法继续选择。")
                    except (ValueError, TypeError):
                        error_msg = f"LLM 在批次 {batch_number} 为输入 '{internal_key_str}' (LLM原始键: '{llm_key_str}') 返回了非整数索引: '{selected_index}'。"
                        logger.error(error_msg)
                        raise ValueError(error_msg + " - 无法继续选择。")

                # 检查当前批次中是否有 LLM 未返回结果的项
                # batch 中的 item[0] 是 internal_key_str ("'key': 'value'") 格式
                # batch_processed_internal_keys_in_response 也包含 internal_key_str 格式
                missing_in_batch_internal_keys = [
                    item_tuple[0] for item_tuple in batch if item_tuple[0] not in batch_processed_internal_keys_in_response
                ]
                if missing_in_batch_internal_keys:
                    # missing_in_batch_internal_keys 已经是内部格式的字符串列表
                    error_msg = f"LLM 在批次 {batch_number} 未对以下 {len(missing_in_batch_internal_keys)} 项返回结果: {', '.join([f'{k}' for k in missing_in_batch_internal_keys])}"
                    logger.error(error_msg)
                    # 决定是否继续？这里选择抛出异常
                    raise ValueError(error_msg + " - 无法继续选择。")

            except Exception as e:
                error_msg = f"处理批次 {batch_number}/{total_batches} 的 LLM 响应时出错: {e}"
                logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg + " - 无法继续选择。")

        # 检查是否有任何失败项未被 LLM 处理 (理论上不应发生，因为前面会报错)
        all_failed_input_strings = {item[0]
                                    for item in self.failed_fuzzy_selection}
        # 使用 processed_inputs_in_llm_responses 检查 LLM 是否处理了所有失败项
        final_unprocessed_strings = all_failed_input_strings - \
            processed_inputs_in_llm_responses

        if final_unprocessed_strings:
            # 如果执行到这里，说明之前的错误处理逻辑有遗漏
            error_msg = f"严重错误：在所有批次处理后，仍有 {len(final_unprocessed_strings)} 个模糊选择失败项未被 LLM 处理: {', '.join(final_unprocessed_strings)}"
            logger.error(error_msg)
            # 使用 RuntimeError 表示内部逻辑错误
            raise RuntimeError(error_msg + " - 代码选择逻辑存在问题。")

        logger.info(f"LLM 选择流程完成。成功处理/选择了 {len(llm_selected_codes)} 项。")
        return llm_selected_codes

    def select_codes(self) -> Dict[str, Dict[str, Any]]:
        """
        执行完整的代码选择流程：模糊选择 -> LLM 选择。

        Returns:
            Dict[str, Dict[str, Any]]: 最终的选择结果 {original_input_key_str: selected_row_dict}。
        """
        # 1. 模糊选择
        self._fuzzy_select()

        # 2. LLM 选择
        llm_selections = self._llm_select()

        # 3. 合并结果 (LLM 的结果优先级更高，如果 key 冲突)
        final_selection = self.selected_codes.copy()
        # 用 LLM 的结果覆盖模糊匹配的结果（现在基于 input_str，不应冲突）
        final_selection.update(llm_selections)

        # 检查是否有输入参数最终没有被选中
        # 使用原始的 input_str 进行比较
        all_input_strings = set(self.matched_models_dict.keys())
        selected_strings = set(final_selection.keys())
        unselected_strings = all_input_strings - selected_strings
        if unselected_strings:
            # 如果执行到这里，说明 LLM 选择步骤没有正确抛出错误
            error_msg = f"严重错误：有 {len(unselected_strings)} 个输入参数最终未能选定代码行: {', '.join(unselected_strings)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg + " - 代码选择逻辑存在问题，未能为所有输入选择代码。")

        logger.info(f"代码选择流程完成。最终为所有 {len(final_selection)} 个输入参数选定代码行。")
        return final_selection


# ==============================================================================
# 4. Code Generator (来自 code_generator.py)
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
                csv_path = Path(csv_path_str)  # 转换为 Path 对象
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

                    csv_path_default = Path(csv_path_str)
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

                # --- 统一处理代码添加和 %int% (无论代码来源) ---
                final_code_part = code_to_use  # 初始值

                # 只有当 code_to_use 不是 None (即没有被跳过) 时才处理
                if code_to_use is not None:
                    code_to_use_str = str(code_to_use)  # 确保是字符串
                    final_code_part = code_to_use_str  # 更新 final_code_part

                    if "%int%" in code_to_use_str:
                        # 确定提示信息中的产品类型
                        prompt_product_type = product_type_origin if source == "default" else product_type
                        prompt_model_name = target_model_str
                        # ... (现有 %int% 输入和替换逻辑) ...
                        while True:
                            try:
                                prompt_message = (
                                    f"请为 {prompt_product_type} - '{prompt_model_name}' 输入一个整数值 "
                                    f"(代码模板: {code_to_use_str}, 直接回车跳过使用 '?'): "
                                )
                                user_input_str = input(
                                    prompt_message).strip()  # 获取输入并去空格

                                if not user_input_str:  # 用户直接按回车
                                    final_code_part = "?"
                                    logger.info(
                                        f"用户跳过了为 {prompt_product_type} - '{prompt_model_name}' 输入整数，使用 '?' 占位。")
                                    break  # 跳出循环

                                # 用户有输入，尝试转换为整数
                                user_int = int(user_input_str)
                                # 替换占位符
                                final_code_part = code_to_use_str.replace(
                                    "%int%", str(user_int))
                                logger.info(
                                    f"用户为 {prompt_product_type} - '{prompt_model_name}' 输入了整数 {user_int}，替换占位符得到代码: '{final_code_part}'")
                                break  # 输入有效，跳出循环
                            except ValueError:
                                print("输入无效，请输入一个整数或直接回车跳过。")
                                logger.warning(
                                    f"用户为 {prompt_product_type} - '{prompt_model_name}' 输入了非整数值 '{user_input_str}'，要求重新输入。")

                    # 将最终处理后的代码部分添加到列表 (确保不为空)
                    if final_code_part:  # 只有非空代码才添加
                        codes_for_this_product.append(final_code_part)
                    else:
                        logger.debug(
                            f"Model '{target_model_str}' (来源: {source}) 生成了空代码部分，已跳过添加。")

                else:  # code_to_use is None (被跳过)
                    logger.debug(
                        f"Model '{target_model_str}' (来源: {source}) 被跳过，不生成代码。")

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

        # 6. 格式化输出
        output_string = f"{final_code}"
        logger.info(f"最终生成的产品代码字符串: {output_string}")

        return output_string


# ==============================================================================
# Main Execution Logic
# ==============================================================================

def execute_standard_matching(main_input_json_path: Path) -> Optional[Path]:
    """
    执行标准的匹配、选择和代码生成流程。

    Args:
        main_input_json_path: 主输入 JSON 文件的路径 (通常是 _standardized_all.json)。

    Returns:
        Optional[Path]: 成功时返回最终结果文件的 Path 对象，失败或无结果时返回 None。
    """
    if not main_input_json_path.is_file():
        logger.error(f"错误：指定的主输入文件不存在: {main_input_json_path}")
        return None

    logger.info(f"开始执行 Standard Matcher 完整流程，输入文件: {main_input_json_path.name}")

    # --- 步骤 0: 拆分主输入文件到 temp 目录 ---
    logger.info(f"\n--- 步骤 0: 拆分主输入文件 '{main_input_json_path.name}' 到 temp 目录 ---")
    try:
        processor = AnalysisJsonProcessor(analysis_json_path=main_input_json_path)
        extracted_data = processor.extract_tag_and_common_params()

        if not extracted_data:
            logger.error(f"从主输入文件 '{main_input_json_path.name}' 未提取到任何设备数据，无法继续。")
            return None

        # 确保 temp 目录存在
        TEMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"确保 temp 目录存在: {TEMP_OUTPUT_DIR}")

        # 清空 temp 目录中的旧 .json 文件
        logger.info(f"正在清空 temp 目录中的旧 .json 文件...")
        cleared_count = 0
        for old_file in TEMP_OUTPUT_DIR.glob("*.json"):
            try:
                old_file.unlink()
                cleared_count += 1
            except OSError as e_clear:
                logger.warning(f"无法删除旧文件 '{old_file}': {e_clear}")
        logger.info(f"已清空 {cleared_count} 个旧 .json 文件。")


        # 写入拆分后的文件
        split_files_count = 0
        for item in extracted_data:
            tag_numbers = item.get("位号", [])
            common_params = item.get("共用参数", {})

            if not tag_numbers or not common_params:
                logger.warning(f"跳过无效的数据项（缺少位号或共用参数）：{item}")
                continue

            # 生成文件名 (用下划线连接位号，并替换斜杠)
            # 替换文件名中的非法字符，例如 '/'
            safe_tag_str = "_".join(tag_numbers).replace("/", "_").replace("\\", "_")
            # 进一步清理，移除其他可能不安全的字符 (只保留字母、数字、下划线、连字符)
            safe_filename_base = re.sub(r'[^\w\-]+', '', safe_tag_str)
            if not safe_filename_base: # 如果清理后为空，给个默认名
                safe_filename_base = f"unknown_tag_{split_files_count + 1}"
            temp_filename = f"{safe_filename_base}.json"
            temp_file_path = TEMP_OUTPUT_DIR / temp_filename

            try:
                with open(temp_file_path, 'w', encoding='utf-8') as f_temp:
                    # 只写入共用参数部分
                    json.dump(common_params, f_temp, indent=4, ensure_ascii=False)
                logger.info(f"成功将位号 {tag_numbers} 的参数写入到: {temp_file_path}")
                split_files_count += 1
            except IOError as e_write:
                logger.error(f"无法写入临时文件 '{temp_file_path}': {e_write}")
                return None # 写入失败，中止

        if split_files_count == 0:
             logger.error("未能成功拆分任何文件到 temp 目录，无法继续。")
             return None

        logger.info(f"成功拆分主文件为 {split_files_count} 个文件到 {TEMP_OUTPUT_DIR}")

    except FileNotFoundError: # AnalysisJsonProcessor 初始化时可能抛出
         logger.error(f"初始化 AnalysisJsonProcessor 失败：文件未找到 {main_input_json_path}")
         return None
    except Exception as e_split:
        logger.error(f"拆分主输入文件时发生错误: {e_split}", exc_info=True)
        return None


    # --- 开始处理 temp 目录中的文件 ---
    index_json = INDEX_JSON_PATH # 索引文件路径是固定的
    logger.info(f"使用索引文件: {index_json}")

    json_files_in_temp = sorted(list(TEMP_OUTPUT_DIR.glob("*.json"))) # 按名称排序
    if not json_files_in_temp:
        logger.error(f"错误: temp 目录 '{TEMP_OUTPUT_DIR}' 中没有找到拆分后的 JSON 文件。")
        return None

    logger.info(f"开始处理 temp 目录中的 {len(json_files_in_temp)} 个 JSON 文件...")

    all_results = [] # 用于存储所有文件的处理结果
    any_file_processed_successfully = False # 跟踪是否有任何文件成功处理

    for i, temp_json_file_path in enumerate(json_files_in_temp):
        logger.info(f"\n{'='*20} 开始处理文件 {i+1}/{len(json_files_in_temp)}: {temp_json_file_path.name} {'='*20}")
        current_file_successful = True

        # --- 步骤 1: 获取 CSV 列表 ---
        print(f"\n--- 文件: {temp_json_file_path.name} - 步骤 1: 获取 CSV 列表 ---")
        fetcher = FetchCsvlist()
        logger.info(f"使用输入文件: {temp_json_file_path}")
        csv_list_map_result = fetcher.fetch_csv_lists(temp_json_file_path, index_json)

        if not csv_list_map_result:
            logger.error(f"文件 {temp_json_file_path.name}: 未能获取 CSV 列表，跳过此文件。")
            current_file_successful = False
            # 不立即将 all_successful 设为 False，允许其他文件继续处理
            # continue # 处理下一个文件 # 改为记录错误并继续，最后判断是否有成功项
        else:
            print(
                f"文件 {temp_json_file_path.name}: 获取到的 CSV 列表映射: {json.dumps(csv_list_map_result, indent=2, ensure_ascii=False)}")

        # --- 步骤 2: 模型匹配 ---
        if current_file_successful:
            print(f"\n--- 文件: {temp_json_file_path.name} - 步骤 2: 模型匹配 ---")
            model_matcher = ModelMatcher(
                csv_list_map=csv_list_map_result, input_json_path=str(temp_json_file_path))
            matched_models_result = model_matcher.match()

            if not matched_models_result:
                logger.warning(f"文件 {temp_json_file_path.name}: 模型匹配未产生任何结果。")
            else:
                print(
                    f"文件 {temp_json_file_path.name}: 模型匹配结果 (部分): {json.dumps(dict(list(matched_models_result.items())[:2]), indent=2, ensure_ascii=False)}...")
        else:
            matched_models_result = {} # 如果上一步失败，则无匹配结果

        # --- 步骤 3: 代码选择 ---
        selected_codes_result = {}
        if current_file_successful and matched_models_result:
            print(f"\n--- 文件: {temp_json_file_path.name} - 步骤 3: 代码选择 ---")
            code_selector = CodeSelector(matched_models_dict=matched_models_result)
            try:
                selected_codes_result = code_selector.select_codes()
                print(
                    f"文件 {temp_json_file_path.name}: 代码选择结果 (部分): {json.dumps(dict(list(selected_codes_result.items())[:2]), indent=2, ensure_ascii=False)}...")
            except (ValueError, RuntimeError) as e:
                logger.error(f"文件 {temp_json_file_path.name}: 代码选择过程中发生错误: {e}，跳过此文件。")
                current_file_successful = False
        elif not matched_models_result and current_file_successful:
             logger.warning(f"文件 {temp_json_file_path.name}: 没有模型匹配结果，跳过代码选择。")


        # --- 步骤 4: 代码生成 ---
        final_code_result = f"产品型号生成失败（文件: {temp_json_file_path.name}）：处理链早期步骤失败或无代码可选。"
        if current_file_successful and selected_codes_result:
            print(f"\n--- 文件: {temp_json_file_path.name} - 步骤 4: 代码生成 ---")
            code_generator = CodeGenerator()
            print(f"\n文件 {temp_json_file_path.name}: 代码生成过程中可能需要您输入整数值...")
            try:
                final_code_result = code_generator.generate_final_code(
                    csv_list_map=csv_list_map_result,
                    selected_codes_data=selected_codes_result
                )
            except Exception as e_gen:
                logger.error(f"文件 {temp_json_file_path.name}: 代码生成过程中发生错误: {e_gen}", exc_info=True)
                final_code_result = f"产品型号生成失败（文件: {temp_json_file_path.name}）：错误 - {e_gen}"
                current_file_successful = False
        elif not selected_codes_result and current_file_successful:
            logger.warning(f"文件 {temp_json_file_path.name}: 没有代码选择结果，跳过代码生成。")
            final_code_result = f"产品型号生成失败（文件: {temp_json_file_path.name}）：无代码可选。"


        # --- 单个文件最终结果 ---
        print(f"\n--- 文件: {temp_json_file_path.name} - 最终结果 ---")
        print(final_code_result)
        print(f"{'='*70}")

        # --- 提取位号并聚合结果 ---
        try:
            tag_number_str_from_filename = temp_json_file_path.stem.replace("_", "/")
            tag_numbers_list = [tag_number_str_from_filename]
        except Exception as e_tag:
            logger.error(f"从文件名 {temp_json_file_path.name} 提取位号时出错: {e_tag}")
            tag_numbers_list = [f"无法解析文件名: {temp_json_file_path.name}"]

        result_entry = {
            "位号": tag_numbers_list,
            "型号代码": final_code_result
        }
        all_results.append(result_entry)

        if current_file_successful:
            any_file_processed_successfully = True
        else:
            logger.error(f"文件 {temp_json_file_path.name} 处理失败。")

        if i < len(json_files_in_temp) - 1:
            logger.info(f"处理完文件 {temp_json_file_path.name}，等待 5 秒...")
            time.sleep(5)

    # --- 所有文件处理完毕，写入最终结果文件 ---
    if not any_file_processed_successfully and all_results: # 如果有结果但没有一个成功，说明都是错误信息
        logger.error("所有拆分出的文件均处理失败。最终结果文件将包含错误详情。")
    elif not all_results: # 如果根本没有结果（例如拆分后temp为空，或所有拆分项都跳过了）
        logger.error("未能生成任何结果。")
        return None

    logger.info("\n所有文件处理循环结束。准备写入最终结果文件...")

    input_file_stem = main_input_json_path.stem
    base_name_for_output = input_file_stem.replace('_analysis', '').replace('_standardized_all', '')
    output_filename = f"{base_name_for_output}_results.json"
    output_file_path = project_root / "data" / "output" / output_filename

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            json.dump(all_results, f_out, indent=4, ensure_ascii=False)
        logger.info(f"所有结果已成功写入到文件: {output_file_path}")
        # --- 清理临时文件 ---
        logger.info(f"正在清理临时目录: {TEMP_OUTPUT_DIR}")
        try:
            if TEMP_OUTPUT_DIR.is_dir():
                # 删除目录下的所有文件和子目录
                for item in TEMP_OUTPUT_DIR.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                logger.info(f"临时目录 '{TEMP_OUTPUT_DIR}' 清理完成。")
            else:
                logger.warning(f"临时目录 '{TEMP_OUTPUT_DIR}' 不存在，无需清理。")
        except Exception as e_cleanup:
            logger.error(f"清理临时目录 '{TEMP_OUTPUT_DIR}' 时出错: {e_cleanup}", exc_info=True)
        # --- 清理临时文件结束 ---
        return output_file_path
    except Exception as e_write:
        logger.error(f"将最终结果写入文件 {output_file_path} 时出错: {e_write}", exc_info=True)
        # --- 清理临时文件 (即使文件写入失败) ---
        logger.info(f"正在清理临时目录 (文件写入失败后): {TEMP_OUTPUT_DIR}")
        try:
            if TEMP_OUTPUT_DIR.is_dir():
                # 删除目录下的所有文件和子目录
                for item in TEMP_OUTPUT_DIR.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                logger.info(f"临时目录 '{TEMP_OUTPUT_DIR}' 清理完成 (文件写入失败后)。")
            else:
                logger.warning(f"临时目录 '{TEMP_OUTPUT_DIR}' 不存在，无需清理 (文件写入失败后)。")
        except Exception as e_cleanup:
            logger.error(f"清理临时目录 '{TEMP_OUTPUT_DIR}' 时出错 (文件写入失败后): {e_cleanup}", exc_info=True)
        # --- 清理临时文件结束 ---
        return None


if __name__ == "__main__":
    # --- 命令行参数解析 ---
    parser = argparse.ArgumentParser(description="Standard Matcher: 接收主输入文件，拆分，处理并生成型号代码。")
    parser.add_argument(
        "--input-file",
        required=True,
        help="主输入 JSON 文件的完整路径 (例如 'data/output/温变规格书_standardized_all.json')。"
    )
    args = parser.parse_args()
    input_file_path_cli = Path(args.input_file)

    final_output_file = execute_standard_matching(input_file_path_cli)

    if final_output_file:
        logger.info(f"\nStandard Matcher 完整流程执行完毕。最终输出: {final_output_file}")
        sys.exit(0)
    else:
        logger.error("\nStandard Matcher 流程执行失败或未生成输出文件。请检查日志。")
        sys.exit(1)
