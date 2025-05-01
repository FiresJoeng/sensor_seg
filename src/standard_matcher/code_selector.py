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
        self.fuzzy_select_threshold = 0.4  # 模糊选择相似度阈值

        self.selected_codes: Dict[str, Dict[str, Any]] = {} # 最终选择结果: {original_input_key: selected_row_dict}
        self.failed_fuzzy_selection: List[Tuple[str, List[Dict[str, Any]]]] = [] # 模糊选择失败的项: [(input_str, candidate_rows), ...]

        logger.info(f"CodeSelector 初始化完成。待处理 {len(self.matched_models_dict)} 个输入参数。")

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
                selected_row_dict = candidate_rows[0].copy() # 使用副本以防修改原始数据
                # 特殊规则：如果 code 是 %int%，尝试从 input_str 提取数字
                if selected_row_dict.get('code') == '%int%':
                    # 尝试从 input_str (格式 "'key': 'value'") 提取 value 中的数字
                    match = re.search(r":\s*'(\d+)'", input_str)
                    if match:
                        extracted_value = match.group(1)
                        selected_row_dict['code'] = extracted_value # 替换 code
                        logger.info(f"模糊选择成功 (唯一候选，%int% 替换): 键值对 '{input_str}' -> code 替换为 '{extracted_value}'")
                    else:
                        logger.warning(f"唯一候选行的 code 为 '%int%'，但在输入 '{input_str}' 中未能提取到数字值，保留 '%int%'。")

                # 使用原始 input_str 作为 key 存储选择结果
                self.selected_codes[input_str] = selected_row_dict
                logger.info(f"模糊选择成功 (唯一候选): 键值对 '{input_str}' -> 最终匹配结果 {selected_row_dict}")
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
                logger.info(f"模糊选择成功: 键值对 '{input_str}' -> 匹配结果 {selected_row_dict} (分数: {best_score})")
            else:
                # 记录失败项以供 LLM 处理
                self.failed_fuzzy_selection.append((input_str, candidate_rows))
                logger.debug(f"模糊选择失败或分数低: 输入 '{input_str}' (最高分: {best_score}, 最佳候选 code: {best_match_row.get('code', 'N/A') if best_match_row else 'N/A'})")
                # 注意：即使 original_key 解析失败，也应该记录失败项


        # 使用 set(self.selected_codes.keys()) 来获取唯一选中的 input_str 数量
        logger.info(f"模糊选择完成。成功 {len(self.selected_codes)} 项，失败 {len(self.failed_fuzzy_selection)} 项。")

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

        logger.info(f"开始 LLM 选择，处理 {len(self.failed_fuzzy_selection)} 个失败项...")

        # 准备提示词内容
        items_to_select_str_parts = []
        for input_str, candidate_rows in self.failed_fuzzy_selection:
            item_str = f"输入参数: \"{input_str}\"\n候选行列表:\n"
            for i, row in enumerate(candidate_rows):
                # 显示关键信息：code 和 description
                code = row.get('code', 'N/A')
                desc = row.get('description', '无描述')
                item_str += f"  {i}: code='{code}', description='{desc[:60]}{'...' if len(desc) > 60 else ''}'\n" # 限制描述长度
            items_to_select_str_parts.append(item_str)

        items_to_select_str = "\n---\n".join(items_to_select_str_parts)

        user_prompt = SELECTOR_USER_PROMPT_TEMPLATE.format(items_to_select_str=items_to_select_str)

        # 调用 LLM
        llm_response = call_llm_for_match(SELECTOR_SYSTEM_PROMPT, user_prompt, expect_json=True)

        if not llm_response or isinstance(llm_response, str) or llm_response.get("error"):
            logger.error(f"LLM 调用失败或返回错误: {llm_response}")
            # 对于 LLM 失败的情况，可以选择一个默认策略，例如选择第一个候选行，或者标记为未选择
            # 这里暂时不选择，让它们保留在未匹配状态（如果需要最终输出的话）
            return llm_selected_codes

        # 处理 LLM 响应
        try:
            # llm_response 预期是 {"input_key: value": selected_index, ...}
            if not isinstance(llm_response, dict):
                 logger.error(f"LLM 响应不是预期的字典格式: {llm_response}")
                 return llm_selected_codes

            processed_inputs_str = set()
            for input_str, selected_index in llm_response.items():
                processed_inputs_str.add(input_str)
                # 找到原始的候选列表
                original_candidates = None
                for failed_input_str, candidates in self.failed_fuzzy_selection:
                    if failed_input_str == input_str:
                        original_candidates = candidates
                        break

                if original_candidates is None:
                    logger.warning(f"LLM 返回了未知的输入参数: '{input_str}'")
                    continue

                # 验证索引
                try:
                    selected_index = int(selected_index)
                    if 0 <= selected_index < len(original_candidates):
                        selected_row = original_candidates[selected_index]
                        # 使用原始 input_str 作为 key
                        selected_row_dict = selected_row
                        llm_selected_codes[input_str] = selected_row_dict
                        logger.info(f"LLM 选择成功: 键值对 '{input_str}' -> 匹配结果 {selected_row_dict} (选中索引: {selected_index})")
                    else:
                        logger.warning(f"LLM 为输入 '{input_str}' 返回了无效索引: {selected_index} (候选数量: {len(original_candidates)})。将尝试选择第一个。")
                        # 备选策略：选择第一个，使用原始 input_str 作为 key
                        if original_candidates: # 确保有候选行
                            selected_row_dict = original_candidates[0]
                            llm_selected_codes[input_str] = selected_row_dict
                            logger.info(f"LLM 选择失败 (无效索引)，备选策略: 键值对 '{input_str}' -> 匹配结果 {selected_row_dict} (选中索引: 0)")

                except (ValueError, TypeError):
                    logger.warning(f"LLM 为输入 '{input_str}' 返回了非整数索引: '{selected_index}'。将尝试选择第一个。")
                    # 备选策略：选择第一个，使用原始 input_str 作为 key
                    if original_candidates: # 确保有候选行
                        selected_row_dict = original_candidates[0]
                        llm_selected_codes[input_str] = selected_row_dict
                        logger.info(f"LLM 选择失败 (非整数索引)，备选策略: 键值对 '{input_str}' -> 匹配结果 {selected_row_dict} (选中索引: 0)")


            # 检查是否有 LLM 未返回结果的失败项
            missing_inputs_tuples = [(item[0], item[1]) for item in self.failed_fuzzy_selection if item[0] not in processed_inputs_str]
            if missing_inputs_tuples:
                logger.warning(f"LLM 未对以下 {len(missing_inputs_tuples)} 个模糊选择失败项返回结果，将尝试选择第一个候选行:")
                for missing_input_str, original_candidates in missing_inputs_tuples:
                 if original_candidates: # 确保有候选行
                     # 使用原始 missing_input_str 作为 key
                     selected_row_dict = original_candidates[0]
                     llm_selected_codes[missing_input_str] = selected_row_dict
                     logger.warning(f"  - 备选策略: 键值对 '{missing_input_str}' -> 匹配结果 {selected_row_dict} (选中索引: 0)")
                 else:
                     logger.warning(f"  - 备选策略失败: 输入 '{missing_input_str}' 没有候选行。")


        except Exception as e:
            logger.error(f"处理 LLM 响应时出错: {e}", exc_info=True)
            # 出错时返回当前已成功处理的部分
            return llm_selected_codes

        logger.info(f"LLM 选择完成。成功选择 {len(llm_selected_codes)} 项。")
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
        final_selection.update(llm_selections) # 用 LLM 的结果覆盖模糊匹配的结果（现在基于 input_str，不应冲突）

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
            "remark": "*1：适用于PT1、PT2、PT3金属保护管时，选择000。；*2：插入长度≤50mm，材质与外保护管一致，如需其他，请注明。；*3：带外保护套管时，此项可省略。；*4：仅适用于YTA50、YTA70。；*5：仅适用于HR。；*6：仅适用于NEPSI防爆/N2。；*7：产品如带外保护套管，仍需防尘防爆认证。；*9：仅适用于NEPSI防爆/NS2。"
        }
    ],
    "'元件数量 (类型 Type)': '单支'": [
        {
            "model": "元件数量",
            "code": "-S",
            "description": "单支式",
            "remark": "*1：适用于PT1、PT2、PT3金属保护管时，选择000。；*2：插入长度≤50mm，材质与外保护管一致，如需其他，请注明。；*3：带外保护套管时，此项可省略。；*4：仅适用于YTA50、YTA70。；*5：仅适用于HR。；*6：仅适用于NEPSI防爆/N2。；*7：产品如带外保护套管，仍需防尘防爆认证。；*9：仅适用于NEPSI防爆/NS2。"
        },
        {
            "model": "元件数量",
            "code": "-D",
            "description": "双支式",
            "remark": "*1：适用于PT1、PT2、PT3金属保护管时，选择000。；*2：插入长度≤50mm，材质与外保护管一致，如需其他，请注明。；*3：带外保护套管时，此项可省略。；*4：仅适用于YTA50、YTA70。；*5：仅适用于HR。；*6：仅适用于NEPSI防爆/N2。；*7：产品如带外保护套管，仍需防尘防爆认证。；*9：仅适用于NEPSI防爆/NS2。"
        }
    ],
    "'连接螺纹 (温度元件型号 Therm. Element Model)': '缺失（文档未提供）'": [
        {
            "model": "传感器连接螺纹 (S)注：带温度元件，此项可省",
            "code": "1",
            "description": "M12×1.5",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数(温度，压力，介质密度，介质黏度，介质流速，管道尺寸)。 ；(2)粗糙度Ra=0.8，长度不超过500mm"
        },
        {
            "model": "传感器连接螺纹 (S)注：带温度元件，此项可省",
            "code": "2",
            "description": "M16×1.5",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数(温度，压力，介质密度，介质黏度，介质流速，管道尺寸)。 ；(2)粗糙度Ra=0.8，长度不超过500mm"
        },
        {
            "model": "传感器连接螺纹 (S)注：带温度元件，此项可省",
            "code": "3",
            "description": "M27×2",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数(温度，压力，介质密度，介质黏度，介质流速，管道尺寸)。 ；(2)粗糙度Ra=0.8，长度不超过500mm"
        },
        {
            "model": "传感器连接螺纹 (S)注：带温度元件，此项可省",
            "code": "4",
            "description": "G1/2",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数(温度，压力，介质密度，介质黏度，介质流速，管道尺寸)。 ；(2)粗糙度Ra=0.8，长度不超过500mm"
        },
        {
            "model": "传感器连接螺纹 (S)注：带温度元件，此项可省",
            "code": "5",
            "description": "G3/4",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数(温度，压力，介质密度，介质黏度，介质流速，管道尺寸)。 ；(2)粗糙度Ra=0.8，长度不超过500mm"
        },
        {
            "model": "传感器连接螺纹 (S)注：带温度元件，此项可省",
            "code": "6",
            "description": "1/2NPT",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数(温度，压力，介质密度，介质黏度，介质流速，管道尺寸)。 ；(2)粗糙度Ra=0.8，长度不超过500mm"
        }
    ],
    "'过程连接（法兰等级） (允差等级 Tolerance Error Rating)': 'A级'": [
        {
            "model": "附加规格代码",
            "code": "/A3",
            "description": "外保护套管频率强度计算",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数...；(2)粗糙度Ra=0.8，长度不超过500mm"
        },
        {
            "model": "附加规格代码",
            "code": "/A4",
            "description": "外保护套管材质报告",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数...；(2)粗糙度Ra=0.8，长度不超过500mm"
        },
        {
            "model": "附加规格代码",
            "code": "/A5",
            "description": "外保护套管射线探伤报告",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数...；(2)粗糙度Ra=0.8，长度不超过500mm"
        },
        {
            "model": "附加规格代码",
            "code": "/A6",
            "description": "外保护套管着色渗透报告",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数...；(2)粗糙度Ra=0.8，长度不超过500mm"
        },
        {
            "model": "附加规格代码",
            "code": "/R1",
            "description": "外保护套管接液部分禁油处理",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数...；(2)粗糙度Ra=0.8，长度不超过500mm"
        },
        {
            "model": "附加规格代码",
            "code": "/F1",
            "description": "外保护套管水压测试",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数...；(2)粗糙度Ra=0.8，长度不超过500mm"
        },
        {
            "model": "附加规格代码",
            "code": "/F2",
            "description": "外保护套管接液部分抛光处理",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数...；(2)粗糙度Ra=0.8，长度不超过500mm"
        }
    ],
    "'铠套材质 (铠装材质 Armo. Mat'l)': '316'": [
        {
            "model": "铠装材质",
            "code": "PN",
            "description": "304SS",
            "remark": "*1：适用于PT1、PT2、PT3金属保护管时，选择000。；*2：插入长度≤50mm，材质与外保护管一致，如需其他，请注明。；*3：带外保护套管时，此项可省略。；*4：仅适用于YTA50、YTA70。；*5：仅适用于HR。；*6：仅适用于NEPSI防爆/N2。；*7：产品如带外保护套管，仍需防尘防爆认证。；*9：仅适用于NEPSI防爆/NS2。"
        },
        {
            "model": "铠装材质",
            "code": "SN",
            "description": "321SS",
            "remark": "*1：适用于PT1、PT2、PT3金属保护管时，选择000。；*2：插入长度≤50mm，材质与外保护管一致，如需其他，请注明。；*3：带外保护套管时，此项可省略。；*4：仅适用于YTA50、YTA70。；*5：仅适用于HR。；*6：仅适用于NEPSI防爆/N2。；*7：产品如带外保护套管，仍需防尘防爆认证。；*9：仅适用于NEPSI防爆/NS2。"
        },
        {
            "model": "铠装材质",
            "code": "RN",
            "description": "316SS",
            "remark": "*1：适用于PT1、PT2、PT3金属保护管时，选择000。；*2：插入长度≤50mm，材质与外保护管一致，如需其他，请注明。；*3：带外保护套管时，此项可省略。；*4：仅适用于YTA50、YTA70。；*5：仅适用于HR。；*6：仅适用于NEPSI防爆/N2。；*7：产品如带外保护套管，仍需防尘防爆认证。；*9：仅适用于NEPSI防爆/NS2。"
        },
        {
            "model": "铠装材质",
            "code": "GH",
            "description": "GH3030",
            "remark": "*1：适用于PT1、PT2、PT3金属保护管时，选择000。；*2：插入长度≤50mm，材质与外保护管一致，如需其他，请注明。；*3：带外保护套管时，此项可省略。；*4：仅适用于YTA50、YTA70。；*5：仅适用于HR。；*6：仅适用于NEPSI防爆/N2。；*7：产品如带外保护套管，仍需防尘防爆认证。；*9：仅适用于NEPSI防爆/NS2。"
        }
    ],
    "'铠套外径(d) (铠装直径 Armo. Dia. (mm))': 'Ф6'": [
        {
            "model": "铠装外径（d）",
            "code": "3",
            "description": "Ø3mm（仅用于固定式接头结构）",
            "remark": "*1：适用于PT1、PT2、PT3金属保护管时，选择000。；*2：插入长度≤50mm，材质与外保护管一致，如需其他，请注明。；*3：带外保护套管时，此项可省略。；*4：仅适用于YTA50、YTA70。；*5：仅适用于HR。；*6：仅适用于NEPSI防爆/N2。；*7：产品如带外保护套管，仍需防尘防爆认证。；*9：仅适用于NEPSI防爆/NS2。"
        },
        {
            "model": "铠装外径（d）",
            "code": "4",
            "description": "Ø4mm（仅用于固定式接头结构）",
            "remark": "*1：适用于PT1、PT2、PT3金属保护管时，选择000。；*2：插入长度≤50mm，材质与外保护管一致，如需其他，请注明。；*3：带外保护套管时，此项可省略。；*4：仅适用于YTA50、YTA70。；*5：仅适用于HR。；*6：仅适用于NEPSI防爆/N2。；*7：产品如带外保护套管，仍需防尘防爆认证。；*9：仅适用于NEPSI防爆/NS2。"
        },
        {
            "model": "铠装外径（d）",
            "code": "5",
            "description": "Ø5mm",
            "remark": "*1：适用于PT1、PT2、PT3金属保护管时，选择000。；*2：插入长度≤50mm，材质与外保护管一致，如需其他，请注明。；*3：带外保护套管时，此项可省略。；*4：仅适用于YTA50、YTA70。；*5：仅适用于HR。；*6：仅适用于NEPSI防爆/N2。；*7：产品如带外保护套管，仍需防尘防爆认证。；*9：仅适用于NEPSI防爆/NS2。"
        },
        {
            "model": "铠装外径（d）",
            "code": "6",
            "description": "Ø6mm",
            "remark": "*1：适用于PT1、PT2、PT3金属保护管时，选择000。；*2：插入长度≤50mm，材质与外保护管一致，如需其他，请注明。；*3：带外保护套管时，此项可省略。；*4：仅适用于YTA50、YTA70。；*5：仅适用于HR。；*6：仅适用于NEPSI防爆/N2。；*7：产品如带外保护套管，仍需防尘防爆认证。；*9：仅适用于NEPSI防爆/NS2。"
        },
        {
            "model": "铠装外径（d）",
            "code": "8",
            "description": "Ø8mm",
            "remark": "*1：适用于PT1、PT2、PT3金属保护管时，选择000。；*2：插入长度≤50mm，材质与外保护管一致，如需其他，请注明。；*3：带外保护套管时，此项可省略。；*4：仅适用于YTA50、YTA70。；*5：仅适用于HR。；*6：仅适用于NEPSI防爆/N2。；*7：产品如带外保护套管，仍需防尘防爆认证。；*9：仅适用于NEPSI防爆/NS2。"
        },
        {
            "model": "铠装外径（d）",
            "code": "10",
            "description": "Ø10mm",
            "remark": "*1：适用于PT1、PT2、PT3金属保护管时，选择000。；*2：插入长度≤50mm，材质与外保护管一致，如需其他，请注明。；*3：带外保护套管时，此项可省略。；*4：仅适用于YTA50、YTA70。；*5：仅适用于HR。；*6：仅适用于NEPSI防爆/N2。；*7：产品如带外保护套管，仍需防尘防爆认证。；*9：仅适用于NEPSI防爆/NS2。"
        },
        {
            "model": "铠装外径（d）",
            "code": "Z",
            "description": "其它",
            "remark": "*1：适用于PT1、PT2、PT3金属保护管时，选择000。；*2：插入长度≤50mm，材质与外保护管一致，如需其他，请注明。；*3：带外保护套管时，此项可省略。；*4：仅适用于YTA50、YTA70。；*5：仅适用于HR。；*6：仅适用于NEPSI防爆/N2。；*7：产品如带外保护套管，仍需防尘防爆认证。；*9：仅适用于NEPSI防爆/NS2。"
        }
    ],
    "'TG套管形式 (套管形式 Well Type)': '整体钻孔锥形保护管'": [
        {
            "model": "选型",
            "code": "-K",
            "description": "K型法兰安装锥形保护套管",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数(温度，压力，介质密度，介质黏度，介质流速，管道尺寸)。 ；(2)粗糙度Ra=0.8，长度不超过500mm"
        },
        {
            "model": "选型",
            "code": "-L",
            "description": "L型法兰安装直形保护套管",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数(温度，压力，介质密度，介质黏度，介质流速，管道尺寸)。 ；(2)粗糙度Ra=0.8，长度不超过500mm"
        },
        {
            "model": "选型",
            "code": "-M",
            "description": "M型法兰安装台阶形保护套管",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数(温度，压力，介质密度，介质黏度，介质流速，管道尺寸)。 ；(2)粗糙度Ra=0.8，长度不超过500mm"
        }
    ],
    "'TG套管形式 (套管材质 Well Mat'l)': '316'": [
        {
            "model": "TG",
            "code": "TG",
            "description": "保护套管",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数(温度，压力，介质密度，介质黏度，介质流速，管道尺寸)。 ；(2)粗糙度Ra=0.8，长度不超过500mm"
        }
    ],
    "'过程连接（法兰等级） (压力等级 Pressure Rating)': 'Class150'": [
        {
            "model": "过程连接",
            "code": "-S",
            "description": "抱箍式 *1",
            "remark": "*1 采用抱箍将感温片固定在测温点，需提供现场管道直径。；*2 将感温片焊接固定在测温点，需提供现场曲面半径。；*3 铠套端预留螺纹接口，便于在隔爆安装环境下安装。"
        },
        {
            "model": "过程连接",
            "code": "-W",
            "description": "焊接式 *2",
            "remark": "*1 采用抱箍将感温片固定在测温点，需提供现场管道直径。；*2 将感温片焊接固定在测温点，需提供现场曲面半径。；*3 铠套端预留螺纹接口，便于在隔爆安装环境下安装。"
        }
    ],
    "'铠套外径(d) (套管外径 Well Outside Dia. (mm))': '根部不大于28,套管厚度由供货商根据振动频率和强度计算确定'": [
        {
            "model": "根部直径 (Q)",
            "code": "-22",
            "description": "22mm",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数...；(2)粗糙度Ra=0.8，长度不超过500mm"
        },
        {
            "model": "根部直径 (Q)",
            "code": "-27",
            "description": "27mm（不适用于DN25（1\"））",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数...；(2)粗糙度Ra=0.8，长度不超过500mm"
        },
        {
            "model": "根部直径 (Q)",
            "code": "-%int%",
            "description": "单位mm",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数...；(2)粗糙度Ra=0.8，长度不超过500mm"
        }
    ],
    "'过程连接 (法兰标准 Flange STD.)': 'HG/T20615-2009'": [
        {
            "model": "法兰标准",
            "code": "-A",
            "description": "ANSI",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数(温度，压力，介质密度，介质黏度，介质流速，管道尺寸)。 ；(2)粗糙度Ra=0.8，长度不超过500mm"
        },
        {
            "model": "法兰标准",
            "code": "-D",
            "description": "DIN",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数(温度，压力，介质密度，介质黏度，介质流速，管道尺寸)。 ；(2)粗糙度Ra=0.8，长度不超过500mm"
        },
        {
            "model": "法兰标准",
            "code": "-G",
            "description": "GB/T 9123",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数(温度，压力，介质密度，介质黏度，介质流速，管道尺寸)。 ；(2)粗糙度Ra=0.8，长度不超过500mm"
        },
        {
            "model": "法兰标准",
            "code": "-H",
            "description": "HG20592、HG20615",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数(温度，压力，介质密度，介质黏度，介质流速，管道尺寸)。 ；(2)粗糙度Ra=0.8，长度不超过500mm"
        },
        {
            "model": "法兰标准",
            "code": "-Z",
            "description": "其它",
            "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数(温度，压力，介质密度，介质黏度，介质流速，管道尺寸)。 ；(2)粗糙度Ra=0.8，长度不超过500mm段略 (中间类似规则)"
        }
    ],
    "'过程连接（法兰等级） (管嘴长度 Length mm)': '150'": [
        {
            "model": "钎套长度",
            "code": "%int%",
            "description": "单位mm",
            "remark": "*1 采用抱箍将感温片固定在测温点，需提供现场管道直径。；*2 将感温片焊接固定在测温点，需提供现场曲面半径。；*3 铠套端预留螺纹接口，便于在隔爆安装环境下安装。"
        }
    ],
    "'过程连接（法兰等级） (插入深度 Well Length (mm))': '250'": [
        {
            "model": "铠套长度",
            "code": "%int%",
            "description": "单位mm *注3，仅适用于SR)",
            "remark": "*1 采用抱箍将感温片固定在测温点，需提供现场管道直径。；*2 将感温片焊接固定在测温点，需提供现场曲面半径。；*3 铠套端预留螺纹接口，便于在隔爆安装环境下安装。"
        }
    ],
    "'连接螺纹 (测量范围 Meas. Range (°C))': '缺失（文档未提供）'": [
        {
            "model": "补偿导线长度",
            "code": "%int%",
            "description": "单位mm *3",
            "remark": "*1 采用抱箍将感温片固定在测温点，需提供现场管道直径。；*2 将感温片焊接固定在测温点，需提供现场曲面半径。；*3 铠套端预留螺纹接口，便于在隔爆安装环境下安装。"
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
