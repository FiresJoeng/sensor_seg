# -*- coding: utf-8 -*-
"""
模块：代码选择器 (Code Selector)
功能：负责从候选 CSV 行列表中为每个输入参数选择最佳代码行。
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
        f"错误：在 code_selector.py 中导入模块失败 - {e}。"
        f"请检查项目结构和 PYTHONPATH。\n"
        f"项目根目录尝试设置为: {project_root}", exc_info=True)
    raise

# --- 全局配置 ---
logger = logging.getLogger(__name__)

# ==============================================================================
# 3. Code Selector
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
5.  **注意转义**: 请注意反斜杠等特殊字符必须正确转义，如：“1/2" NPT (F)”应该正确转义成“1/2\" NPT (F)”，避免输出JSON时出现格式错误。只对原文的斜杠进行正确转义，请不要额外添加斜杠。返回参数时注意不要增加其他的符号，对于字符处理使用半角。
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
特别注意：所有的输入参数都必须选中一个最优索引！不允许不选择！返回的项一定要是完整的！！
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
        self.fuzzy_select_threshold = 0.6  # 模糊选择相似度阈值

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
            error_msg = f"注意：在所有批次处理后，仍有 {len(final_unprocessed_strings)} 个模糊选择失败项未被 LLM 处理: {', '.join(final_unprocessed_strings)}"
            logger.error(error_msg)

        logger.info(f"LLM 选择流程完成。成功处理/选择了 {len(llm_selected_codes)} 项。")
        return llm_selected_codes

    def select_codes(self) -> Dict[str, Dict[str, Any]]:
        """
        执行完整的代码选择流程：模糊选择 -> LLM 选择。

        Returns:
            Dict[str, Dict[str, Any]] : 最终的选择结果 {original_input_key_str: selected_row_dict}。
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
            error_msg = f"错误：有 {len(unselected_strings)} 个输入参数最终未能选定代码行: {', '.join(unselected_strings)}"
            logger.error(error_msg)

        logger.info(f"代码选择流程完成。最终为所有 {len(final_selection)} 个输入参数选定代码行。")
        return final_selection
