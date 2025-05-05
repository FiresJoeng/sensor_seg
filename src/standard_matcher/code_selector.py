# -*- coding: utf-8 -*-
"""
模块：代码选择器 (Code Selector)
功能：从模型匹配器提供的候选 CSV 行列表中，为每个输入参数选择唯一的最佳匹配行。
      采用模糊匹配优先，LLM 匹配补充的策略。
"""

import json
import logging
import sys
from pathlib import Path
import re
from typing import Dict, List, Any, Tuple, Optional

# 尝试导入模糊匹配库
try:
    from thefuzz import fuzz
    THEFUZZ_AVAILABLE = True
except ImportError:
    THEFUZZ_AVAILABLE = False
    print("警告：'thefuzz' 库未安装。模糊匹配功能将不可用。请运行 'pip install thefuzz python-Levenshtein'", file=sys.stderr)

# 确保项目根目录在 sys.path 中以便导入 config 和 llm
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import settings
    # 假设 llm.py 提供了调用 LLM 的接口
    from src.standard_matcher.llm import call_llm_for_match
except ImportError as e:
    print(
        f"错误：在 code_selector.py 中导入模块失败 - {e}。"
        f"请检查项目结构和 PYTHONPATH。\n"
        f"项目根目录尝试设置为: {project_root}", file=sys.stderr)
    raise

# 配置日志
logging.basicConfig(level=settings.LOG_LEVEL,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- LLM 提示词定义 ---
# 系统提示：定义 LLM 的角色和任务
SELECTOR_SYSTEM_PROMPT = """
你是一个精确的代码选择助手。你的任务是：对于用户提供的每个“输入参数”（键值对形式），从其对应的“候选标准行列表”中，选择**唯一**的最匹配的那一行。

**重要规则:**
1.  **选择基础**: 你的选择必须基于“输入参数的键和值整体”与“候选行提供的 **description 和 param 字段内容**”之间的**语义相似度**。你需要理解输入参数的含义，并找到语义上最贴合的那一行候选标准代码（基于其 description 和 param）。
2.  **唯一性**: 对于每个输入参数，你必须从其候选列表中选择**且仅选择一个**最佳匹配行。
3.  **输出格式**: 必须以 JSON 格式返回选择结果。JSON 的键是原始的输入参数字符串（格式："key: value"），值是你选择的最佳匹配行的**索引号 (从 0 开始)**。
    示例: `{"输入参数键1: 输入参数值1": 0, "输入参数键2: 输入参数值2": 2}`
4.  **完整性**: 确保为每一个提供的“输入参数”都选择一个候选行索引。
"""

# 用户提示模板：包含待选择项和候选行列表
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


class CodeSelector:
    """
    负责从候选 CSV 行列表中为每个输入参数选择最佳代码行。
    """

    def __init__(self, matched_models_dict: Dict[str, List[Dict[str, Any]]]):
        """
        初始化 CodeSelector。

        Args:
            matched_models_dict (Dict[str, List[Dict[str, Any]]]):
                模型匹配器输出的字典，格式为 {"input_key: value": [candidate_row1, ...]}。
        """
        if not THEFUZZ_AVAILABLE:
            logger.error("模糊匹配库 'thefuzz' 不可用，无法继续。请安装。")
            raise ImportError("缺少 thefuzz 库")

        self.matched_models_dict = matched_models_dict
        self.fuzzy_select_threshold = 0.6  # 模糊选择相似度阈值

        # 最终选择结果: {original_input_key: selected_row_dict}
        self.selected_codes: Dict[str, Dict[str, Any]] = {}
        # 模糊选择失败的项: [(input_str, candidate_rows), ...]
        self.failed_fuzzy_selection: List[Tuple[str,
                                                List[Dict[str, Any]]]] = []

        logger.info(
            f"CodeSelector 初始化完成。待处理 {len(self.matched_models_dict)} 个输入参数。")

    # _parse_input_string 函数不再需要，已移除

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

                # 新规则：如果 code 包含 %int%，尝试从 input_str 提取数字并替换
                if '%int%' in original_code:
                    # 尝试从 input_str (格式 "'key': 'value'") 提取 value 中的数字
                    match = re.search(r":\s*'(\d+)'", input_str)
                    if match:
                        extracted_value = match.group(1)
                        # 替换 %int% 部分
                        new_code = original_code.replace('%int%', extracted_value)
                        selected_row_dict['code'] = new_code
                    else:
                        logger.warning(
                            f"唯一候选行的 code '{original_code}' 包含 '%int%'，但在输入 '{input_str}' 中未能提取到数字值，保留原样。")

                # 使用原始 input_str 作为 key 存储选择结果（可能是修改后的，也可能是原始的）
                self.selected_codes[input_str] = selected_row_dict
                logger.info(
                    f"模糊选择成功 (唯一候选): 键值对 '{input_str}' -> 最终匹配结果 {selected_row_dict}")
                continue

            best_match_row = None
            best_score = -1
            best_row_index = -1

            # 将输入字符串与每个候选行进行比较
            for index, row in enumerate(candidate_rows):
                row_str = self._row_to_string(row)
                # 使用 token_sort_ratio 忽略词序
                score = fuzz.token_sort_ratio(input_str, row_str)

                if score > best_score:
                    best_score = score
                    best_match_row = row
                    best_row_index = index

            # 判断是否达到阈值
            if best_match_row and best_score >= self.fuzzy_select_threshold * 100:
                # 使用原始 input_str 作为 key
                selected_row_dict = best_match_row
                self.selected_codes[input_str] = selected_row_dict
                logger.info(
                    f"模糊选择成功: 键值对 '{input_str}' -> 匹配结果 {selected_row_dict} (分数: {best_score})")
            else:
                # 记录失败项以供 LLM 处理
                self.failed_fuzzy_selection.append((input_str, candidate_rows))
                logger.debug(
                    f"模糊选择失败或分数低: 输入 '{input_str}' (最高分: {best_score}, 最佳候选 code: {best_match_row.get('code', 'N/A') if best_match_row else 'N/A'})")
                # 注意：即使 original_key 解析失败，也应该记录失败项

        # 使用 set(self.selected_codes.keys()) 来获取唯一选中的 input_str 数量
        logger.info(
            f"模糊选择完成。成功 {len(self.selected_codes)} 项，失败 {len(self.failed_fuzzy_selection)} 项。")

    def _llm_select(self) -> Dict[str, Dict[str, Any]]:
        """
        使用 LLM 对模糊选择失败的项进行选择。

        Returns:
            Dict[str, Dict[str, Any]]: LLM 选择成功的结果 {original_input_key: selected_row_dict}。
        """
        llm_selected_codes = {}
        if not self.failed_fuzzy_selection:
            logger.info("没有模糊选择失败的项，跳过 LLM 选择。")
            return llm_selected_codes

        total_failed_count = len(self.failed_fuzzy_selection)
        logger.info(f"开始 LLM 选择，处理 {total_failed_count} 个模糊选择失败项...")

        batch_size = 5  # 每批处理的最大数量
        processed_inputs_in_llm_responses = set() # 记录 LLM 实际返回了结果的输入字符串

        for i in range(0, total_failed_count, batch_size):
            batch = self.failed_fuzzy_selection[i:i + batch_size]
            batch_number = (i // batch_size) + 1
            total_batches = (total_failed_count + batch_size - 1) // batch_size
            logger.info(f"处理 LLM 选择批次 {batch_number}/{total_batches} (共 {len(batch)} 项)...")

            # 准备当前批次的提示词内容
            items_to_select_str_parts = []
            batch_input_mapping = {} # 存储当前批次的 input_str -> candidate_rows 映射
            for input_str, candidate_rows in batch:
                batch_input_mapping[input_str] = candidate_rows
                item_str = f"输入参数: \"{input_str}\"\n候选行列表:\n"
                for idx, row in enumerate(candidate_rows):
                    # 获取完整的 description 和 param
                    desc = row.get('description', '无描述')
                    param = row.get('param', '无参数') # 或者根据实际情况决定默认值
                    # 更新 item_str 的格式，只包含索引、description 和 param
                    item_str += f"  {idx}: description='{desc}', param='{param}'\n"
                items_to_select_str_parts.append(item_str)

            items_to_select_str = "\n---\n".join(items_to_select_str_parts)
            user_prompt = SELECTOR_USER_PROMPT_TEMPLATE.format(
                items_to_select_str=items_to_select_str)

            # 调用 LLM 处理当前批次
            llm_response = call_llm_for_match(
                SELECTOR_SYSTEM_PROMPT, user_prompt, expect_json=True)

            # 处理当前批次的 LLM 响应
            if not llm_response or isinstance(llm_response, str) or llm_response.get("error"):
                logger.error(f"批次 {batch_number}/{total_batches} 的 LLM 调用失败或返回错误: {llm_response}")
                # 当前批次失败，可以选择跳过或应用备选策略给批内所有项
                logger.warning(f"批次 {batch_number}/{total_batches} 中的所有项将尝试使用备选策略（选择第一个）。")
                for input_str, original_candidates in batch:
                    if input_str not in llm_selected_codes and original_candidates: # 避免重复处理和空列表
                        selected_row_dict = original_candidates[0]
                        llm_selected_codes[input_str] = selected_row_dict
                        logger.warning(
                            f"  - LLM 批次失败，备选策略: 键值对 '{input_str}' -> 匹配结果 {selected_row_dict} (选中索引: 0)")
                continue # 继续处理下一批

            # 成功获取响应，处理批内各项结果
            try:
                if not isinstance(llm_response, dict):
                    logger.error(f"批次 {batch_number}/{total_batches} 的 LLM 响应不是预期的字典格式: {llm_response}")
                    # 格式错误，同样可以应用备选策略
                    logger.warning(f"批次 {batch_number}/{total_batches} 响应格式错误，将尝试对批内未处理项使用备选策略。")
                    for input_str, original_candidates in batch:
                         if input_str not in llm_selected_codes and original_candidates:
                            selected_row_dict = original_candidates[0]
                            llm_selected_codes[input_str] = selected_row_dict
                            logger.warning(
                                f"  - LLM 响应格式错误，备选策略: 键值对 '{input_str}' -> 匹配结果 {selected_row_dict} (选中索引: 0)")
                    continue # 继续处理下一批

                # 遍历 LLM 返回的当前批次的结果
                batch_processed_inputs_str_in_response = set()
                for input_str, selected_index in llm_response.items():
                    batch_processed_inputs_str_in_response.add(input_str)
                    processed_inputs_in_llm_responses.add(input_str) # 加入全局已处理集合

                    # 查找原始候选列表 (在当前批次的映射中查找)
                    original_candidates = batch_input_mapping.get(input_str)

                    if original_candidates is None:
                        logger.warning(f"LLM 在批次 {batch_number} 返回了未知的输入参数: '{input_str}' (可能来自其他批次或无效)，跳过此项。")
                        continue

                    # 验证索引并选择
                    try:
                        selected_index = int(selected_index)
                        if 0 <= selected_index < len(original_candidates):
                            selected_row = original_candidates[selected_index]
                            selected_row_dict = selected_row
                            llm_selected_codes[input_str] = selected_row_dict
                            logger.info(
                                f"LLM 选择成功 (批次 {batch_number}): 键值对 '{input_str}' -> 匹配结果 {selected_row_dict} (选中索引: {selected_index})")
                        else:
                            logger.warning(
                                f"LLM 在批次 {batch_number} 为输入 '{input_str}' 返回了无效索引: {selected_index} (候选数量: {len(original_candidates)})。将尝试选择第一个。")
                            if original_candidates:
                                selected_row_dict = original_candidates[0]
                                llm_selected_codes[input_str] = selected_row_dict
                                logger.info(
                                    f"LLM 选择失败 (无效索引，批次 {batch_number})，备选策略: 键值对 '{input_str}' -> 匹配结果 {selected_row_dict} (选中索引: 0)")
                    except (ValueError, TypeError):
                        logger.warning(
                            f"LLM 在批次 {batch_number} 为输入 '{input_str}' 返回了非整数索引: '{selected_index}'。将尝试选择第一个。")
                        if original_candidates:
                            selected_row_dict = original_candidates[0]
                            llm_selected_codes[input_str] = selected_row_dict
                            logger.info(
                                f"LLM 选择失败 (非整数索引，批次 {batch_number})，备选策略: 键值对 '{input_str}' -> 匹配结果 {selected_row_dict} (选中索引: 0)")

                # 检查当前批次中是否有 LLM 未返回结果的项
                missing_in_batch_tuples = [
                    (item[0], item[1]) for item in batch if item[0] not in batch_processed_inputs_str_in_response
                ]
                if missing_in_batch_tuples:
                    logger.warning(f"LLM 在批次 {batch_number} 未对以下 {len(missing_in_batch_tuples)} 项返回结果，将尝试选择第一个候选行:")
                    for missing_input_str, original_candidates in missing_in_batch_tuples:
                         if missing_input_str not in llm_selected_codes and original_candidates: # 避免重复处理和空列表
                            selected_row_dict = original_candidates[0]
                            llm_selected_codes[missing_input_str] = selected_row_dict
                            logger.warning(
                                f"  - LLM 未返回，备选策略 (批次 {batch_number}): 键值对 '{missing_input_str}' -> 匹配结果 {selected_row_dict} (选中索引: 0)")
                         elif not original_candidates:
                             logger.warning(f"  - 备选策略失败 (批次 {batch_number}): 输入 '{missing_input_str}' 没有候选行。")


            except Exception as e:
                logger.error(f"处理批次 {batch_number}/{total_batches} 的 LLM 响应时出错: {e}", exc_info=True)
                # 批次处理出错，可以选择对批内未处理项应用备选策略
                logger.warning(f"因处理批次 {batch_number} 时出错，将尝试对批内未成功处理项使用备选策略。")
                for input_str, original_candidates in batch:
                    if input_str not in llm_selected_codes and original_candidates:
                        selected_row_dict = original_candidates[0]
                        llm_selected_codes[input_str] = selected_row_dict
                        logger.warning(
                            f"  - LLM 批次处理异常，备选策略: 键值对 '{input_str}' -> 匹配结果 {selected_row_dict} (选中索引: 0)")

        # 所有批次处理完毕后，检查是否有任何失败项完全没有被 LLM 处理（即使是备选策略也没覆盖到）
        # 这通常不应该发生，因为上面的逻辑会尝试覆盖所有情况，但作为最后防线检查
        all_failed_input_strings = {item[0] for item in self.failed_fuzzy_selection}
        final_unprocessed_strings = all_failed_input_strings - set(llm_selected_codes.keys())

        if final_unprocessed_strings:
             logger.error(f"严重警告：在所有批次处理和备选策略后，仍有 {len(final_unprocessed_strings)} 个模糊选择失败项未被处理:")
             for unprocessed_str in final_unprocessed_strings:
                 logger.error(f"  - 最终未处理: '{unprocessed_str}'")
                 # 这里可以考虑是否强制选择第一个，或者保留为未选择状态
                 # 查找原始数据以强制选择第一个
                 candidates_for_unprocessed = None
                 for failed_input, candidates in self.failed_fuzzy_selection:
                     if failed_input == unprocessed_str:
                         candidates_for_unprocessed = candidates
                         break
                 if candidates_for_unprocessed:
                     selected_row_dict = candidates_for_unprocessed[0]
                     llm_selected_codes[unprocessed_str] = selected_row_dict
                     logger.warning(f"  - 强制最终备选策略: 键值对 '{unprocessed_str}' -> 匹配结果 {selected_row_dict} (选中索引: 0)")
                 else:
                      logger.error(f"  - 无法应用最终备选策略：找不到 '{unprocessed_str}' 的候选行。")


        logger.info(f"LLM 选择流程完成。通过 LLM (包括备选策略) 共处理/选择了 {len(llm_selected_codes)} 项。")
        return llm_selected_codes

    def select_codes(self) -> Dict[str, Dict[str, Any]]:
        """
        执行完整的代码选择流程：模糊选择 -> LLM 选择。

        Returns:
            Dict[str, Dict[str, Any]]: 最终的选择结果 {original_input_key: selected_row_dict}。
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
            logger.warning(f"有 {len(unselected_strings)} 个输入参数最终未能选定代码行:")
            for unselected_str in unselected_strings:
                logger.warning(f"  - 未选定: '{unselected_str}'")

        logger.info(f"代码选择流程完成。最终选定 {len(final_selection)} 项。")
        return final_selection


# --- 主执行逻辑 (示例) ---
if __name__ == "__main__":
    logger.info("开始执行 CodeSelector 示例...")

    # 1. 定义示例输入数据 (来自 ModelMatcher 的输出格式)
    example_matched_models = {
    "'元件数量': '单支'": [
        {
            "model": "元件数量",
            "code": "-S",
            "description": "单支式",
            "param": "1；单支；Simplex；单支单点；单元件；单只；Single，单支铠装；Single，，1支；one",
            "is_default": "1"
        },
        {
            "model": "元件数量",
            "code": "-D",
            "description": "双支式",
            "param": "2；双支；Double；Duplex；Two；双支式；双元件；",
            "is_default": "0"
        }
    ],
    "'分度号': 'IEC标准 Pt100'": [
        {
            "model": "分度号",
            "code": "3",
            "description": "PT100 三线",
            "param": "RTD Pt100 三线制；Pt100（三线制）；三线制；Pt100；Pt100（A级 三线制)；CLASS A/三线制；IEC60751 CLASS A / 三线制；RTD Pt100 三线制；三线制 Three wire；热电阻Pt100三线制；Pt100（A级 三线制）；3线制；3线制RTD；Pt100(3-wire) IEC60751 Class A；3 wires；3线RTD；PT100 3线； PT100 AT 0℃ 3WIRE (DIN TYPE)；3-wire；3线 IEC 60751；PT100-3WIRE；Pt100 ohm，3W，Class B；3 WIRE",
            "is_default": "0"
        },
        {
            "model": "分度号",
            "code": "4",
            "description": "PT100 四线",
            "param": "四线制；Pt100（四线制）；4WIRE；4线；",
            "is_default": "0"
        },
        {
            "model": "分度号",
            "code": "E",
            "description": "镍铬-铜镍",
            "param": "E；镍铬-铜镍",
            "is_default": "0"
        },
        {
            "model": "分度号",
            "code": "J",
            "description": "铁-铜镍",
            "param": "J；铁-铜镍",
            "is_default": "0"
        },
        {
            "model": "分度号",
            "code": "K",
            "description": "镍铬-镍铝",
            "param": "K；IEC标准 K；镍铬-镍硅",
            "is_default": "1"
        },
        {
            "model": "分度号",
            "code": "T",
            "description": "铜-铜镍",
            "param": "T；铜-铜镍",
            "is_default": "0"
        },
        {
            "model": "分度号",
            "code": "N",
            "description": "镍铬硅-镍硅镁",
            "param": "N；镍铬硅-镍硅镁",
            "is_default": "0"
        },
        {
            "model": "分度号",
            "code": "S",
            "description": "铂-铂铑10",
            "param": "S；铂-铂铑10",
            "is_default": "0"
        },
        {
            "model": "分度号",
            "code": "R",
            "description": "铂-铂铑13",
            "param": "R；铂-铂铑13",
            "is_default": "0"
        },
        {
            "model": "分度号",
            "code": "B",
            "description": "铂铑30-铂铑6",
            "param": "B；铂铑30-铂铑6",
            "is_default": "0"
        },
        {
            "model": "分度号",
            "code": "Z",
            "description": "其它",
            "param": "",
            "is_default": "0"
        }
    ],
    "'铠套材质': '316'": [
        {
            "model": "铠套材质",
            "code": "PN",
            "description": "304SS",
            "param": "304SS；AISI 304；304；30408；S30408；06Cr19Ni10",
            "is_default": "1"
        },
        {
            "model": "铠套材质",
            "code": "SN",
            "description": "321SS",
            "param": "321SS；1Cr18Ni9Ti；321；S32168；06Cr18Ni11Ti",
            "is_default": "0"
        },
        {
            "model": "铠套材质",
            "code": "RN",
            "description": "316SS",
            "param": "316SS；316；SS316；316S.S；316SST；S.S316；S31608；06Cr17Ni12Mo2",
            "is_default": "0"
        },
        {
            "model": "铠套材质",
            "code": "GH",
            "description": "GH3030",
            "param": "GH3030",
            "is_default": "0"
        },
        {
            "model": "铠套材质",
            "code": "Z",
            "description": "其它",
            "param": "其他",
            "is_default": "0"
        }
    ],
    "'铠套外径': 'φ6'": [
        {
            "model": "铠套外径(d)",
            "code": "3",
            "description": "Ø3mm（仅用于固定式接头结构）",
            "param": "3；3mm；φ3；φ3mm",
            "is_default": "0"
        },
        {
            "model": "铠套外径(d)",
            "code": "4",
            "description": "Ø4mm（仅用于固定式接头结构）",
            "param": "4；4mm；φ4；φ4mm",
            "is_default": "0"
        },
        {
            "model": "铠套外径(d)",
            "code": "5",
            "description": "Ø5mm",
            "param": "5mm；φ5mm；5；φ5",
            "is_default": "0"
        },
        {
            "model": "铠套外径(d)",
            "code": "6",
            "description": "Ø6mm",
            "param": "6；φ6；6mm；Φ6mm",
            "is_default": "1"
        },
        {
            "model": "铠套外径(d)",
            "code": "8",
            "description": "Ø8mm",
            "param": "8；8mm；φ8；φ8mm",
            "is_default": "0"
        },
        {
            "model": "铠套外径(d)",
            "code": "10",
            "description": "Ø10mm",
            "param": "10；10mm；φ10；φ10mm",
            "is_default": "0"
        },
        {
            "model": "铠套外径(d)",
            "code": "Z",
            "description": "其它",
            "param": "其他",
            "is_default": "0"
        }
    ],
    "'接线盒形式代码': '分体式'": [
        {
            "model": "接线盒形式",
            "code": "-1",
            "description": "YTA610、YTA710",
            "param": "一体式；一体化铠装热电阻-带外套管；一体化温度变送器；Head Mount；Integrate；一体式温度变送器；一体化安装；变送器一体式；一体化；Integral RTD Field Mounted Tran.；一体化温变；Integral；就地⼀体式；智能一体化；一体式温度变送器；一体化变送器；一体化温度计；integrated；就地一体式",
            "is_default": "1"
        },
        {
            "model": "接线盒形式",
            "code": "-2",
            "description": "接线盒、1/2NPT电气接口",
            "param": "分体式；精小型分体式温度变送器；分体式安装；分体式温度变送器",
            "is_default": "0"
        },
        {
            "model": "接线盒形式",
            "code": "-3",
            "description": "接线盒、M20*1.5电气接口",
            "param": "分体式；精小型分体式温度变送器；分体式安装；分体式温度变送器",
            "is_default": "0"
        },
        {
            "model": "接线盒形式",
            "code": "-4",
            "description": "不带接线盒、1/2NPT电气接口",
            "param": "",
            "is_default": "0"
        },
        {
            "model": "接线盒形式",
            "code": "-5",
            "description": "不带接线盒、M20*1.5电气接口",
            "param": "",
            "is_default": "0"
        }
    ],
    "'TG套管形式': '整体钻孔锥形保护管'": [
        {
            "model": "TG套管形式",
            "code": "-K",
            "description": "K型法兰安装 锥形保护套管",
            "param": "整体钻孔直形套管；直型；直形；整体钻孔直形保护管；直形Straight；法兰式整体钻孔式直形；固定直形整体钻孔法兰套管；法兰直形套管；整体钻孔保护管Tapered Type；Solid hole， tapered；Tapered from drilled barstock；整体钻孔锥型；Tapered；单端钻孔锥型套管；法兰连接整体锥型钻孔；法兰式锥形整体钻孔外套管；钢棒整体钻孔锥形套管；固定法兰式整体钻孔锥形保护套管；固定法兰锥形整体钻孔式；加强型锥型整体钻孔；锥形；一体化整体钻孔锥形法兰套管；整体锥形套管；整体锥形钻孔；整体钻孔式的锥形套管；整体钻孔锥形；整体钻孔锥形保护管；整体钻孔锥形管；整体钻孔锥形套管；整钻锥形；锥形整体钻孔；锥形整体钻孔式套管；整体钻孔保护管",
            "is_default": "0"
        },
        {
            "model": "TG套管形式",
            "code": "-L",
            "description": "L型法兰安装 直形保护套管",
            "param": "整体钻孔直形套管；直型；直形；整体钻孔直形保护管；直形Straight；法兰式整体钻孔式直形；固定直形整体钻孔法兰套管；法兰直形套管；整体钻孔保护管",
            "is_default": "0"
        },
        {
            "model": "TG套管形式",
            "code": "-M",
            "description": "M型法兰安装 台阶形保护套管",
            "param": "台阶形保护管；阶梯形保护管；Stepped；Stepped Type",
            "is_default": "0"
        }
    ],
    "'套管材质': '316'": [
        {
            "model": "套管材质",
            "code": "GH",
            "description": "GH3030",
            "param": "GH3030",
            "is_default": "0"
        },
        {
            "model": "套管材质",
            "code": "GN",
            "description": "GH3039",
            "param": "GH3039",
            "is_default": "0"
        },
        {
            "model": "套管材质",
            "code": "HC",
            "description": "Hastelloy C-276",
            "param": "Hastelloy C-276；哈氏C276；HC276",
            "is_default": "0"
        },
        {
            "model": "套管材质",
            "code": "JN",
            "description": "Inconel 625",
            "param": "Inconel 625",
            "is_default": "0"
        },
        {
            "model": "套管材质",
            "code": "N1",
            "description": "碳钢（SA-105）",
            "param": "SA-105",
            "is_default": "0"
        },
        {
            "model": "套管材质",
            "code": "N2",
            "description": "碳钢（15CrMo）",
            "param": "15CrMo",
            "is_default": "0"
        },
        {
            "model": "套管材质",
            "code": "PN",
            "description": "304不锈钢",
            "param": "304SS；AISI 304；304；30408；S30408；06Cr19Ni10;0Cr18Ni9",
            "is_default": "0"
        },
        {
            "model": "套管材质",
            "code": "QN",
            "description": "310不锈钢",
            "param": "S31008；310S；310；310SS；06Cr25Ni20",
            "is_default": "0"
        },
        {
            "model": "套管材质",
            "code": "RN",
            "description": "316不锈钢",
            "param": "316SS；316；SS316；316S.S；316SST；S.S316；S31608；06Cr17Ni12Mo2；0Cr17Ni12Mo2",
            "is_default": "0"
        },
        {
            "model": "套管材质",
            "code": "RH",
            "description": "316H不锈钢",
            "param": "316H",
            "is_default": "0"
        },
        {
            "model": "套管材质",
            "code": "RL",
            "description": "316L不锈钢",
            "param": "316L；316LSS；SS316L；S31603；022Cr17Ni12Mo2；00Cr17Ni14Mo2",
            "is_default": "0"
        },
        {
            "model": "套管材质",
            "code": "SN",
            "description": "321不锈钢",
            "param": "321SS；1Cr18Ni9Ti；321；S32168；06Cr18Ni11Ti；0Cr18Ni10Ti",
            "is_default": "0"
        },
        {
            "model": "套管材质",
            "code": "TA",
            "description": "钛(TA2)",
            "param": "TA2；钛",
            "is_default": "0"
        },
        {
            "model": "套管材质",
            "code": "DP",
            "description": "2205双相不锈钢",
            "param": "2205；S22053；022Cr25Ni5Mo3N",
            "is_default": "0"
        },
        {
            "model": "套管材质",
            "code": "Z",
            "description": "其它",
            "param": "其它",
            "is_default": "0"
        }
    ],
    "'法兰等级': 'Class150'": [
        {
            "model": "过程连接（法兰等级）",
            "code": "1",
            "description": "PN2.0（150#）RF",
            "param": "PN2.0 RF；150# RF；PN20 RF；Class150 RF；150LB RF；□□-20 RF；CL150 RF；",
            "is_default": "0"
        },
        {
            "model": "过程连接（法兰等级）",
            "code": "2",
            "description": "PN5.0（300#）RF",
            "param": "PN5.0 RF；300# RF；PN50 RF；Class300 RF；300LB RF；□□-50 RF；CL300 RF；PN5.0 □□ RF；300# □□ RF；PN50 □□ RF；Class300 □□ RF；300LB □□ RF；CL300 □□ RF；",
            "is_default": "0"
        },
        {
            "model": "过程连接（法兰等级）",
            "code": "3",
            "description": "PN11.0（600#）RF",
            "param": "PN11.0 RF；600# RF；PN110 RF；Class600 RF；600LB RF；□□-110 RF；CL600 RF；",
            "is_default": "0"
        },
        {
            "model": "过程连接（法兰等级）",
            "code": "4",
            "description": "PN15.0（900#）RJ",
            "param": "PN15.0 RJ；900# RJ；PN150 RJ；Class900 RJ；900LB RJ；□□-150 RJ；CL900 RJ；PN15.0 RTJ；900# RTJ；PN150 RTJ；Class900 RTJ；900LB RTJ；□□-150 RTJ；CL900 RTJ；",
            "is_default": "0"
        },
        {
            "model": "过程连接（法兰等级）",
            "code": "5",
            "description": "PN26.0（1500#）RJ",
            "param": "PN26.0 RJ；1500# RJ；PN260 RJ；Class1500 RJ；1500LB RJ；□□-260 RJ；CL1500 RJ；PN26.0 RTJ；1500# RTJ；PN260 RTJ；Class1500 RTJ；1500LB RTJ；□□-260 RTJ；CL1500 RTJ；",
            "is_default": "0"
        },
        {
            "model": "过程连接（法兰等级）",
            "code": "6",
            "description": "PN42.0（2500#）RJ",
            "param": "PN42.0 RJ；2500# RJ；PN420 RJ；Class2500 RJ；2500LB RJ；□□-420 RJ；CL2500 RJ；PN42.0 RTJ；2500# RTJ；PN420 RTJ；Class2500 RTJ；2500LB RTJ；□□-420 RTJ；CL2500 RTJ；",
            "is_default": "0"
        },
        {
            "model": "过程连接（法兰等级）",
            "code": "7",
            "description": "PN1.0 RF",
            "param": "PN1.0 RF；PN10 RF；□□-10 RF；",
            "is_default": "0"
        },
        {
            "model": "过程连接（法兰等级）",
            "code": "8",
            "description": "PN1.6 RF",
            "param": "PN1.6 RF；PN16 RF；□□-16 RF；",
            "is_default": "0"
        },
        {
            "model": "过程连接（法兰等级）",
            "code": "9",
            "description": "PN2.5 RF",
            "param": "PN2.5 RF；PN25 RF；□□-25 RF；",
            "is_default": "0"
        },
        {
            "model": "过程连接（法兰等级）",
            "code": "10",
            "description": "PN4.0 RF",
            "param": "PN4.0 RF；PN40 RF；□□-40 RF；",
            "is_default": "0"
        },
        {
            "model": "过程连接（法兰等级）",
            "code": "11",
            "description": "PN6.3 RF",
            "param": "PN6.3 RF；PN63 RF；□□-63 RF；",
            "is_default": "0"
        },
        {
            "model": "过程连接（法兰等级）",
            "code": "Z",
            "description": "其它",
            "param": "其它",
            "is_default": "0"
        }
    ],
    "'根部直径（Q）': '根部不大于28,套管厚度由供货商根据振动频率和温度计算确定'": [
        {
            "model": "根部直径 (Q)",
            "code": "-22",
            "description": "22mm",
            "param": "",
            "is_default": "0"
        },
        {
            "model": "根部直径 (Q)",
            "code": "-27",
            "description": "27mm",
            "param": "（不适用于DN25（1\"））",
            "is_default": "0"
        },
        {
            "model": "根部直径 (Q)",
            "code": "-%int%",
            "description": "单位mm",
            "param": "",
            "is_default": "0"
        }
    ],
    "'过程连接（法兰尺寸（Fs））': 'DN40'": [
        {
            "model": "过程连接（法兰尺寸（Fs））",
            "code": "1",
            "description": "DN25（1\"）",
            "param": "DN25；□□DN25□□；1\"；□□1\"□□； 25-□□",
            "is_default": "0"
        },
        {
            "model": "过程连接（法兰尺寸（Fs））",
            "code": "2",
            "description": "DN40（1-1/2\"）",
            "param": "DN40；□□DN40□□；1-1/2\"；□□1-1/2\"□□；1 1/2\"；□□1 1/2\"□□；40-□□",
            "is_default": "0"
        },
        {
            "model": "过程连接（法兰尺寸（Fs））",
            "code": "3",
            "description": "DN50（2\"）",
            "param": "DN50；□□DN50□□；2\"；□□2\"□□；50-□□",
            "is_default": "0"
        },
        {
            "model": "过程连接（法兰尺寸（Fs））",
            "code": "Z",
            "description": "其它",
            "param": "其它",
            "is_default": "0"
        }
    ],
    "'过程连接（法兰标准）': 'HG/T20615-2009'": [
        {
            "model": "过程连接（法兰标准）",
            "code": "-A",
            "description": "ANSI",
            "param": "ANSI；ANSI B16.5； ASME；ASME B16.5",
            "is_default": "0"
        },
        {
            "model": "过程连接（法兰标准）",
            "code": "-D",
            "description": "DIN",
            "param": "DIN；DIN □□",
            "is_default": "0"
        },
        {
            "model": "过程连接（法兰标准）",
            "code": "-G",
            "description": "GB/T 9123",
            "param": "GB/T9123；GB/T9123 □□",
            "is_default": "0"
        },
        {
            "model": "过程连接（法兰标准）",
            "code": "-H",
            "description": "HG20592、HG20615",
            "param": "HG□□；化工法兰",
            "is_default": "0"
        },
        {
            "model": "过程连接（法兰标准）",
            "code": "-Z",
            "description": "其它",
            "param": "GB□□；SH□□；JIS□□；API□□；",
            "is_default": "0"
        }
    ],
    "'法兰材质': '316'": [
        {
            "model": "法兰材质",
            "code": "GH",
            "description": "GH3030",
            "param": "GH3030",
            "is_default": "0"
        },
        {
            "model": "法兰材质",
            "code": "GN",
            "description": "GH3039",
            "param": "GH3039",
            "is_default": "0"
        },
        {
            "model": "法兰材质",
            "code": "HC",
            "description": "Hastelloy C-276",
            "param": "Hastelloy C-276；哈氏C276；HC276",
            "is_default": "0"
        },
        {
            "model": "法兰材质",
            "code": "JN",
            "description": "Inconel 625",
            "param": "Inconel 625",
            "is_default": "0"
        },
        {
            "model": "法兰材质",
            "code": "N1",
            "description": "碳钢（SA-105）",
            "param": "SA-105",
            "is_default": "0"
        },
        {
            "model": "法兰材质",
            "code": "N2",
            "description": "碳钢（15CrMo）",
            "param": "15CrMo",
            "is_default": "0"
        },
        {
            "model": "法兰材质",
            "code": "PN",
            "description": "304不锈钢",
            "param": "304SS；AISI 304；304；30408；S30408；06Cr19Ni10;0Cr18Ni9",
            "is_default": "0"
        },
        {
            "model": "法兰材质",
            "code": "QN",
            "description": "310不锈钢",
            "param": "S31008；310S；310；310SS；06Cr25Ni20",
            "is_default": "0"
        },
        {
            "model": "法兰材质",
            "code": "RN",
            "description": "316不锈钢",
            "param": "316SS；316；SS316；316S.S；316SST；S.S316；S31608；06Cr17Ni12Mo2；0Cr17Ni12Mo2",
            "is_default": "0"
        },
        {
            "model": "法兰材质",
            "code": "RH",
            "description": "316H不锈钢",
            "param": "316H",
            "is_default": "0"
        },
        {
            "model": "法兰材质",
            "code": "RL",
            "description": "316L不锈钢",
            "param": "316L；316LSS；SS316L；S31603；022Cr17Ni12Mo2；00Cr17Ni14Mo2",
            "is_default": "0"
        },
        {
            "model": "法兰材质",
            "code": "SN",
            "description": "321不锈钢",
            "param": "321SS；1Cr18Ni9Ti；321；S32168；06Cr18Ni11Ti；0Cr18Ni10Ti",
            "is_default": "0"
        },
        {
            "model": "法兰材质",
            "code": "TA",
            "description": "钛(TA2)",
            "param": "TA2；钛",
            "is_default": "0"
        },
        {
            "model": "法兰材质",
            "code": "DP",
            "description": "2205双相不锈钢",
            "param": "2205；S22053；022Cr25Ni5Mo3N",
            "is_default": "0"
        },
        {
            "model": "法兰材质",
            "code": "Z",
            "description": "其它",
            "param": "其它",
            "is_default": "0"
        }
    ],
    "'插入深度（L）': '250'": [
        {
            "model": "插入长度（L）",
            "code": "-%int%",
            "description": "单位mm",
            "param": "铠套热电阻/热点偶；带外保护套管时，此项可省略。",
            "is_default": "1"
        }
    ],
    "'元件类型': '热电阻'": [
        {
            "model": "传感器主型号",
            "code": "HZ",
            "description": "热电阻",
            "param": "",
            "is_default": "1"
        }
    ],
    "'壳体代码': '304'": [
        {
            "model": "套管主型号",
            "code": "TG",
            "description": "保护套管",
            "param": "",
            "is_default": "1"
        }
    ],
    "'接线口': '1/2\" NPT (F)'": [
        {
            "model": "传感器连接螺纹（S）",
            "code": "1",
            "description": "M12×1.5",
            "param": "M12；M12*1.5；M12×1.5",
            "is_default": "0"
        },
        {
            "model": "传感器连接螺纹（S）",
            "code": "2",
            "description": "M16×1.5",
            "param": "M16；M16*1.5；M16×1.5",
            "is_default": "0"
        },
        {
            "model": "传感器连接螺纹（S）",
            "code": "3",
            "description": "M27×2",
            "param": "M27；M27*2；M27×2；M27X2；固定外螺纹 M27×2；FIXED THREAD M27x2；Screw ThreadM27X2",
            "is_default": "0"
        },
        {
            "model": "传感器连接螺纹（S）",
            "code": "4",
            "description": "G1/2",
            "param": "G1/2；G1/2外螺纹(M)；G1/2(M)",
            "is_default": "0"
        },
        {
            "model": "传感器连接螺纹（S）",
            "code": "5",
            "description": "M20×1.5",
            "param": "M20；M20*1.5；M20×1.5；M20*1.5(M)；M20×1.5(M)；M20X1.5；M20x1.5(M)；固定外螺纹M20x1.5",
            "is_default": "0"
        },
        {
            "model": "传感器连接螺纹（S）",
            "code": "6",
            "description": "1/2NPT",
            "param": "1/2NPT；1/2\"NPT；NPT1/2；NPT1/2\"；1/2NPT(M)；1/2\"NPT(M)；NPT1/2(M)；NPT1/2\"(M)；1/2\"NPT(外螺纹)；1/2\"NPT螺纹；热电阻(弹簧式1/2”NPT外螺纹连接)；固定外螺纹 1/2\"NPT；MFR STD",
            "is_default": "0"
        },
        {
            "model": "传感器连接螺纹（S）",
            "code": "Z",
            "description": "其它",
            "param": "其它",
            "is_default": "0"
        }
    ],
    "'防护等级 Enclosure Protection': 'IP65'": [
        {
            "model": "传感器附加规格",
            "code": "",
            "description": "默认",
            "param": "若甲方未提供，则选择此项",
            "is_default": "1"
        },
        {
            "model": "传感器附加规格",
            "code": "/N1",
            "description": "一体化温度变送器隔爆证书",
            "param": "需一体化温度变送器与变送器NEPSI选项为NF2同时生效",
            "is_default": "0"
        },
        {
            "model": "传感器附加规格",
            "code": "/N2",
            "description": "一体化温度变送器本安证书",
            "param": "需一体化温度变送器与变送器NEPSI选项为NS2或NS25同时生效",
            "is_default": "0"
        },
        {
            "model": "传感器附加规格",
            "code": "/w",
            "description": "电气接口弯头",
            "param": "电气接口弯头；带",
            "is_default": "0"
        },
        {
            "model": "传感器附加规格",
            "code": "/A1",
            "description": "焊接座",
            "param": "材质□□□ 长□□□mm；□□□焊接安装座，长□□□mm",
            "is_default": "0"
        },
        {
            "model": "传感器附加规格",
            "code": "/B11",
            "description": "防水接线盒，材质铝合金",
            "param": "合金铝；Aluminum；铸铝合金；防爆铝合金；铝；Aluminum-Alloy；铝合金＋聚氨酯涂层；低铜铝合金；聚氨酯烤漆低铜铸铝合金；铝合金喷塑；aluminium alloy；铸铝镁合金；静电喷涂铝合金；Aluminium Alloy；Epoxy coated aluminum；环氧涂层铝；铸铝(Cast Aluminum)；铸铝+防腐喷涂；铝合金（防腐处理）；铝制；铸铝+环氧涂层；铝合金(环氧树脂烤漆)；铝覆聚氨酯涂层；铝覆聚氨脂；AL&Polyyurethane paint； Epoxy Coated Aluminium；Aluminum Alloy w/ Coating；铝、覆聚氨酯涂层；Low-copper aluminum with polyurethane painting;聚氨酯涂层低铜铝合金；",
            "is_default": "0"
        },
        {
            "model": "传感器附加规格",
            "code": "/B12",
            "description": "防水接线盒，材质不锈钢",
            "param": "不锈钢；316SS；304SS；316；不锈钢SS；316SST；S.S 316；304SS不锈钢",
            "is_default": "0"
        }
    ],
    "'NEPSI': 'Exd II BT4'": [
        {
            "model": "套管附加规格",
            "code": "",
            "description": "默认",
            "param": "若甲方未提供，则选择此项",
            "is_default": "1"
        },
        {
            "model": "套管附加规格",
            "code": "/A3",
            "description": "外保护套管频率强度计算",
            "param": "",
            "is_default": "0"
        },
        {
            "model": "套管附加规格",
            "code": "/A4",
            "description": "外保护套管材质报告",
            "param": "",
            "is_default": "0"
        },
        {
            "model": "套管附加规格",
            "code": "/A5",
            "description": "外保护套管射线探伤报告",
            "param": "",
            "is_default": "0"
        },
        {
            "model": "套管附加规格",
            "code": "/A6",
            "description": "外保护套管着色渗透报告",
            "param": "",
            "is_default": "0"
        },
        {
            "model": "套管附加规格",
            "code": "/R1",
            "description": "外保护套管接液部分禁油处理",
            "param": "",
            "is_default": "0"
        },
        {
            "model": "套管附加规格",
            "code": "/F1",
            "description": "外保护套管水压测试",
            "param": "",
            "is_default": "0"
        },
        {
            "model": "套管附加规格",
            "code": "/F2",
            "description": "外保护套管接液部分抛光处理",
            "param": "",
            "is_default": "0"
        }
    ],
    "'法兰密封面形式': 'RF'": [
        {
            "model": "接头结构",
            "code": "1",
            "description": "弹簧紧压式（弹簧伸缩长度5mm）",
            "param": "带；弹簧压紧式；弹簧压着式；有；带压紧弹簧；Yes；压簧式；接触传热；Y；弹顶装配式；压着式铠装；压着式；带弹簧铠装",
            "is_default": "1"
        },
        {
            "model": "接头结构",
            "code": "2",
            "description": "固定式",
            "param": "不带；NO，防内漏；",
            "is_default": "0"
        }
    ],
    "'管嘴长度 Length mm': '150'": [
        {
            "model": "加强管长度（N）",
            "code": "%int%",
            "description": "指定长度，单位mm",
            "param": "3位代码",
            "is_default": "0"
        }
    ]
}

    # 2. 创建 Selector 实例并执行选择
    try:
        selector = CodeSelector(matched_models_dict=example_matched_models)
        final_selected_codes = selector.select_codes()

        # 3. 打印结果
        print("\n--- 最终选定代码行 ---")
        # 使用 utf-8 编码打印，避免中文乱码
        print(json.dumps(final_selected_codes, indent=4, ensure_ascii=False))

    except ImportError as e:
        logger.error(f"初始化 CodeSelector 失败，可能是缺少库: {e}")
    except Exception as e:
        logger.error(f"执行 CodeSelector 示例时发生意外错误: {e}", exc_info=True)

    logger.info("CodeSelector 示例执行完毕。")
