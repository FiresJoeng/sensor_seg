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
1.  **选择基础**: 你的选择必须基于“输入参数的键和值整体”与“候选行**完整内容**（包括 model, code, description, remark 等字段）”之间的**语义相似度**。你需要理解输入参数的含义，并找到语义上最贴合的那一行候选标准代码。
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
        self.fuzzy_select_threshold = 0.55  # 模糊选择相似度阈值

        # 最终选择结果: {original_input_key: selected_row_dict}
        self.selected_codes: Dict[str, Dict[str, Any]] = {}
        # 模糊选择失败的项: [(input_str, candidate_rows), ...]
        self.failed_fuzzy_selection: List[Tuple[str,
                                                List[Dict[str, Any]]]] = []

        logger.info(
            f"CodeSelector 初始化完成。待处理 {len(self.matched_models_dict)} 个输入参数。")

    # _parse_input_string 函数不再需要，已移除

    def _row_to_string(self, row_dict: Dict[str, Any]) -> str:
        """将 CSV 行字典转换为用于匹配的单一字符串。"""
        # 组合 model, code, description, remark
        # description 通常最重要，放在前面
        return f"{row_dict.get('description', '')} {row_dict.get('model', '')} {row_dict.get('code', '')} {row_dict.get('remark', '')}".strip()

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
                    code = row.get('code', 'N/A')
                    desc = row.get('description', '无描述')
                    item_str += f"  {idx}: code='{code}', description='{desc[:60]}{'...' if len(desc) > 60 else ''}'\n"
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
    "'元件类型 (仪表名称 Inst. Name)': '热电阻'": [
        {
            "model": "元件类型",
            "code": "HZ",
            "description": "热电阻",
            "remark": ""
        }
    ],
    "'元件数量 (类型 Type)': '单支'": [
        {
            "model": "元件数量",
            "code": "-S",
            "description": "单支式",
            "remark": ""
        },
        {
            "model": "元件数量",
            "code": "-D",
            "description": "双支式",
            "remark": ""
        }
    ],
    "'元件类型 (分度号 Type)': 'IEC标准 Pt100'": [
        {
            "model": "分度号",
            "code": "3",
            "description": "PT100 三线",
            "remark": ""
        },
        {
            "model": "分度号",
            "code": "4",
            "description": "PT100 四线",
            "remark": ""
        },
        {
            "model": "分度号",
            "code": "E",
            "description": "镍铬-铜镍",
            "remark": ""
        },
        {
            "model": "分度号",
            "code": "J",
            "description": "铁-铜镍",
            "remark": ""
        },
        {
            "model": "分度号",
            "code": "K",
            "description": "镍铬-镍铝",
            "remark": ""
        },
        {
            "model": "分度号",
            "code": "T",
            "description": "铜-铜镍",
            "remark": ""
        },
        {
            "model": "分度号",
            "code": "Z",
            "description": "其它",
            "remark": ""
        }
    ],
    "'铠套材质 (铠装材质 Armo. Mat'l)': '316'": [
        {
            "model": "铠装材质",
            "code": "PN",
            "description": "304SS",
            "remark": ""
        },
        {
            "model": "铠装材质",
            "code": "SN",
            "description": "321SS",
            "remark": ""
        },
        {
            "model": "铠装材质",
            "code": "RN",
            "description": "316SS",
            "remark": ""
        },
        {
            "model": "铠装材质",
            "code": "GH",
            "description": "GH3030",
            "remark": ""
        },
        {
            "model": "铠装材质",
            "code": "Z",
            "description": "其它",
            "remark": ""
        }
    ],
    "'接线口 (电气连接 Elec. Conn.)': '1/2\" NPT (F)'": [
        {
            "model": "连接螺纹",
            "code": "0",
            "description": "无",
            "remark": "*2：插入长度≤50mm，材质与外保护管一致，如需其他，请注明。"
        },
        {
            "model": "连接螺纹",
            "code": "1",
            "description": "M12×1.5",
            "remark": ""
        },
        {
            "model": "连接螺纹",
            "code": "2",
            "description": "M16×1.5",
            "remark": ""
        },
        {
            "model": "连接螺纹",
            "code": "3",
            "description": "M27×2",
            "remark": ""
        },
        {
            "model": "连接螺纹",
            "code": "4",
            "description": "G1/2",
            "remark": ""
        },
        {
            "model": "连接螺纹",
            "code": "5",
            "description": "M20×1.5",
            "remark": ""
        },
        {
            "model": "连接螺纹",
            "code": "6",
            "description": "1/2NPT",
            "remark": ""
        }
    ],
    "'过程连接（法兰尺寸（Fs）） (连接规格Conn. Size)': 'DN40'": [
        {
            "model": "法兰尺寸 (Fs)",
            "code": "1",
            "description": "DN25（1\"）",
            "remark": ""
        },
        {
            "model": "法兰尺寸 (Fs)",
            "code": "2",
            "description": "DN40（1-1/2\"）",
            "remark": ""
        },
        {
            "model": "法兰尺寸 (Fs)",
            "code": "3",
            "description": "DN50（2\"）",
            "remark": ""
        },
        {
            "model": "法兰尺寸 (Fs)",
            "code": "z",
            "description": "其它",
            "remark": ""
        }
    ],
    "'过程连接 (法兰标准 Flange STD.)': 'HG/T20615-2009'": [
        {
            "model": "法兰标准",
            "code": "-A",
            "description": "ANSI",
            "remark": ""
        },
        {
            "model": "法兰标准",
            "code": "-D",
            "description": "DIN",
            "remark": ""
        },
        {
            "model": "法兰标准",
            "code": "-G",
            "description": "GB/T 9123",
            "remark": ""
        },
        {
            "model": "法兰标准",
            "code": "-H",
            "description": "HG20592、HG20615",
            "remark": ""
        },
        {
            "model": "法兰标准",
            "code": "-Z",
            "description": "其它",
            "remark": ""
        }
    ],
    "'过程连接（法兰等级） (操作/设计压力 Oper. Press. MPa(G))': '0.3/'": [
        {
            "model": "法兰等级",
            "code": "1",
            "description": "PN2.0（150#）RF",
            "remark": ""
        },
        {
            "model": "法兰等级",
            "code": "2",
            "description": "PN5.0（300#）RF",
            "remark": ""
        },
        {
            "model": "法兰等级",
            "code": "3",
            "description": "PN11.0（600#）RF",
            "remark": ""
        },
        {
            "model": "法兰等级",
            "code": "4",
            "description": "PN15.0（900#）RJ",
            "remark": ""
        },
        {
            "model": "法兰等级",
            "code": "5",
            "description": "PN26.0（1500#）RJ",
            "remark": ""
        },
        {
            "model": "法兰等级",
            "code": "6",
            "description": "PN42.0（2500#）RJ",
            "remark": ""
        },
        {
            "model": "法兰等级",
            "code": "7",
            "description": "PN1.0 RF",
            "remark": ""
        },
        {
            "model": "法兰等级",
            "code": "8",
            "description": "PN1.6 RF",
            "remark": ""
        },
        {
            "model": "法兰等级",
            "code": "9",
            "description": "PN2.5 RF",
            "remark": ""
        },
        {
            "model": "法兰等级",
            "code": "10",
            "description": "PN4.0 RF",
            "remark": ""
        },
        {
            "model": "法兰等级",
            "code": "11",
            "description": "PN6.3 RF",
            "remark": ""
        },
        {
            "model": "法兰等级",
            "code": "Z",
            "description": "其它",
            "remark": ""
        }
    ],
    "'过程连接（法兰等级） (管嘴长度 Length mm)': '150'": [
        {
            "model": "加强管长度（N）",
            "code": "0",
            "description": "0mm",
            "remark": "*1：适用于PT1、PT2、PT3金属保护管时，选择000。"
        },
        {
            "model": "加强管长度（N）",
            "code": "100",
            "description": "100mm",
            "remark": ""
        },
        {
            "model": "加强管长度（N）",
            "code": "150",
            "description": "150mm",
            "remark": ""
        },
        {
            "model": "加强管长度（N）",
            "code": "200",
            "description": "200mm",
            "remark": ""
        },
        {
            "model": "加强管长度（N）",
            "code": "%int%",
            "description": "指定长度，单位mm",
            "remark": ""
        }
    ],
    "'连接螺纹 (温度元件型号 Therm. Element Model)': '缺失（文档未提供）'": [
        {
            "model": "传感器连接螺纹 (S)注：带温度元件，此项可省",
            "code": "1",
            "description": "M12×1.5",
            "remark": ""
        },
        {
            "model": "传感器连接螺纹 (S)注：带温度元件，此项可省",
            "code": "2",
            "description": "M16×1.5",
            "remark": ""
        },
        {
            "model": "传感器连接螺纹 (S)注：带温度元件，此项可省",
            "code": "3",
            "description": "M27×2",
            "remark": ""
        },
        {
            "model": "传感器连接螺纹 (S)注：带温度元件，此项可省",
            "code": "4",
            "description": "G1/2",
            "remark": ""
        },
        {
            "model": "传感器连接螺纹 (S)注：带温度元件，此项可省",
            "code": "5",
            "description": "G3/4",
            "remark": ""
        },
        {
            "model": "传感器连接螺纹 (S)注：带温度元件，此项可省",
            "code": "6",
            "description": "1/2NPT",
            "remark": ""
        }
    ],
    "'过程连接（法兰等级） (允差等级 Tolerance Error Rating)': 'A级'": [
        {
            "model": "附加规格选项",
            "code": "/N1",
            "description": "一体化温度变送器防爆粉尘证书编号:GYB22.2844X 适用标准:GBT3836.1-2021、GBT3836.2-2021、 GBT3836.31-2021 防爆标志:ExdbICT5T6Gb ExbICT70CT90C Db 环境温度:T6(气体):-40~75℃、T5(气体):-40~80C、T70℃ (粉尘环境):-30~65℃、T90C(粉尘环境):-30~80℃ 防护等级:IP66电气接口:1/2NPT内螺纹，M20内螺纹",
            "remark": ""
        },
        {
            "model": "附加规格选项",
            "code": "/N2",
            "description": "一体化温度变送器本安粉尘证书编号:GYB202759X 适用标准:GB 3836.1-2010、GB 3836.4-2010、GB 3836.20-2010 Ex ia lIC T4T5 Ga Ex ic IC T4T5 Gc Ex iaD 21/20 T135 环境温度:-40700(T4)，-40~500(T5) 粉尘防爆适用标准:GB 12476.1-2013、GB 12476.4-2010iD 环境温度:-30~70C 防护等级:IP66",
            "remark": ""
        },
        {
            "model": "附加规格选项",
            "code": "/w",
            "description": "电气接口弯头",
            "remark": ""
        },
        {
            "model": "附加规格选项",
            "code": "/A1",
            "description": "焊接座",
            "remark": ""
        },
        {
            "model": "附加规格选项",
            "code": "/B11",
            "description": "防水接线盒，材质铝合金",
            "remark": ""
        },
        {
            "model": "附加规格选项",
            "code": "/B12",
            "description": "防水接线盒，材质不锈钢",
            "remark": ""
        }
    ],
    "'元件类型 (测量端形式 Meas. End Type)': '绝缘型'": [
        {
            "model": "接头结构",
            "code": "1",
            "description": "弹簧紧压式（弹簧伸缩长度5mm）",
            "remark": ""
        },
        {
            "model": "接头结构",
            "code": "2",
            "description": "固定式",
            "remark": ""
        }
    ],
    "'铠套外径(d) (铠装直径 Armo. Dia. (mm))': 'Ф6'": [
        {
            "model": "铠装外径（d）",
            "code": "3",
            "description": "Ø3mm（仅用于固定式接头结构）",
            "remark": ""
        },
        {
            "model": "铠装外径（d）",
            "code": "4",
            "description": "Ø4mm（仅用于固定式接头结构）",
            "remark": ""
        },
        {
            "model": "铠装外径（d）",
            "code": "5",
            "description": "Ø5mm",
            "remark": ""
        },
        {
            "model": "铠装外径（d）",
            "code": "6",
            "description": "Ø6mm",
            "remark": ""
        },
        {
            "model": "铠装外径（d）",
            "code": "8",
            "description": "Ø8mm",
            "remark": ""
        },
        {
            "model": "铠装外径（d）",
            "code": "10",
            "description": "Ø10mm",
            "remark": ""
        },
        {
            "model": "铠装外径（d）",
            "code": "Z",
            "description": "其它",
            "remark": ""
        }
    ],
    "'壳体代码 (接线盒形式 Terminal Box Style)': '防水型'": [
        {
            "model": "接线盒形式",
            "code": "-1",
            "description": "YTA610、YTA710",
            "remark": ""
        },
        {
            "model": "接线盒形式",
            "code": "-2",
            "description": "接线盒，1/2NPT出气接口（仅适用于YTA50、YTA70）*4",
            "remark": "*4：仅适用于YTA50、YTA70。"
        },
        {
            "model": "接线盒形式",
            "code": "-3",
            "description": "接线盒，M20×1.5出气接口（ 仅适用于YTA50、YTA70）*4",
            "remark": "*4：仅适用于YTA50、YTA70。"
        }
    ],
    "'TG套管形式 (套管形式 Well Type)': '整体钻孔锥形保护管'": [
        {
            "model": "选型",
            "code": "-K",
            "description": "K型法兰安装锥形保护套管",
            "remark": ""
        },
        {
            "model": "选型",
            "code": "-L",
            "description": "L型法兰安装直形保护套管",
            "remark": ""
        },
        {
            "model": "选型",
            "code": "-M",
            "description": "M型法兰安装台阶形保护套管",
            "remark": ""
        }
    ],
    "'TG套管形式 (套管材质 Well Mat'l)': '316'": [
        {
            "model": "棒材质",
            "code": "RN",
            "description": "参见表2",
            "remark": ""
        }
    ],
    "'过程连接（法兰等级） (压力等级 Pressure Rating)': 'Class150'": [
        {
            "model": "法兰材质",
            "code": "RN",
            "description": "参见表2",
            "remark": ""
        }
    ],
    "'铠套外径(d) (套管外径 Well Outside Dia. (mm))': '根部不大于28,套管厚度由供货商根据振动频率和强度计算确定'": [
        {
            "model": "根部直径 (Q)",
            "code": "-22",
            "description": "22mm",
            "remark": ""
        },
        {
            "model": "根部直径 (Q)",
            "code": "-27",
            "description": "27mm（不适用于DN25（1\"））",
            "remark": ""
        },
        {
            "model": "根部直径 (Q)",
            "code": "-%int%",
            "description": "单位mm",
            "remark": ""
        }
    ],
    "'过程连接 (制造厂 Manufacturer)': '缺失（文档未提供）'": [
        {
            "model": "附加规格代码",
            "code": "/A3",
            "description": "外保护套管频率强度计算",
            "remark": ""
        },
        {
            "model": "附加规格代码",
            "code": "/A4",
            "description": "外保护套管材质报告",
            "remark": ""
        },
        {
            "model": "附加规格代码",
            "code": "/A5",
            "description": "外保护套管射线探伤报告",
            "remark": ""
        },
        {
            "model": "附加规格代码",
            "code": "/A6",
            "description": "外保护套管着色渗透报告",
            "remark": ""
        },
        {
            "model": "附加规格代码",
            "code": "/R1",
            "description": "外保护套管接液部分禁油处理",
            "remark": ""
        },
        {
            "model": "附加规格代码",
            "code": "/F1",
            "description": "外保护套管水压测试",
            "remark": ""
        },
        {
            "model": "附加规格代码",
            "code": "/F2",
            "description": "外保护套管接液部分抛光处理",
            "remark": ""
        }
    ],
    "'过程连接（法兰等级） (插入深度 Well Length (mm))': '250'": [
        {
            "model": "插入深度 (U)",
            "code": "-%int%",
            "description": "单位mm",
            "remark": ""
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
