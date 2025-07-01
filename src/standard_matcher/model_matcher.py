# -*- coding: utf-8 -*-
"""
模块：模型匹配器 (Model Matcher)
功能：负责将输入参数与标准 CSV 模型库进行匹配。
"""

import json
import logging
import sys
import re
import time
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

# 确保项目根目录在 sys.path 中以便导入 config, llm
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import settings
    from src.standard_matcher.llm import call_llm_for_match
except ImportError as e:
    logging.getLogger(__name__).critical(
        f"错误：在 model_matcher.py 中导入模块失败 - {e}。"
        f"请检查项目结构和 PYTHONPATH。\n"
        f"项目根目录尝试设置为: {project_root}", exc_info=True)
    raise

# --- 全局配置 ---
logger = logging.getLogger(__name__)

# ==============================================================================
# 2. Model Matcher
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
                # --- FIX: Construct absolute path from project root ---
                csv_path = project_root / csv_path_str
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
                            except Exception as e_gbk_semi:
                                logger.error(
                                    f"尝试多种编码和分隔符后，读取 CSV 文件 {csv_path} 失败: {e_gbk_semi}")

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
            key, _ = data  #只需要键
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
            # 只记录未匹配项，不提示用户输入
            sorted_unmatched = sorted(self.unmatched_inputs.items(),
                                     key=lambda x: x[0])  # 按键(位号)排序
            logger.warning(f"{len(sorted_unmatched)} 个输入项在 LLM 匹配后仍未匹配。")

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
