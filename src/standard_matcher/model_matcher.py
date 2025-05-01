# -*- coding: utf-8 -*-
"""
模块：模型匹配器 (Model Matcher)
功能：将输入的参数键值对与标准库 CSV 文件中的模型条目进行匹配。
      采用模糊匹配优先，LLM 匹配补充的策略。
"""

import json
import logging
import sys
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

# 尝试导入模糊匹配库
try:
    from thefuzz import fuzz, process
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
    from src.standard_matcher.llm import call_llm_for_match  # 假设 llm.py 在同一目录下
except ImportError as e:
    print(
        f"错误：在 model_matcher.py 中导入模块失败 - {e}。"
        f"请检查项目结构和 PYTHONPATH。\n"
        f"项目根目录尝试设置为: {project_root}", file=sys.stderr)
    raise

# 配置日志
logging.basicConfig(level=settings.LOG_LEVEL,  # 直接使用整数级别的 LOG_LEVEL
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- LLM 提示词定义 ---
# 系统提示：定义 LLM 的角色和任务
SYSTEM_PROMPT = """
你是一个智能匹配助手。你的核心任务是将用户提供的“输入参数”（键值对形式）与“可用标准模型库条目”（已按模型名称分组，每个组包含多行详细信息）进行最合适的匹配。

**重要规则:**
1.  **匹配基础**: 匹配决策必须基于“输入参数的键和值整体”与“候选标准模型库条目组内所有行的**完整内容**（包括 model, code, description, remark 等字段）”之间的**语义相似度**。不要仅仅基于模型名称或个别字段进行匹配。你需要理解输入参数的含义，并找到在语义上最贴合的标准模型库条目组。
2.  **唯一性**: 每个输入参数只能匹配到一个标准模型库条目组，反之亦然，每个标准模型库条目组也只能被匹配一次。
3.  **输出格式**: 必须以 JSON 格式返回匹配结果。JSON 的键是原始的输入参数字符串（格式："key: value"），值是匹配到的标准模型库条目的**模型名称 (model)**。
    示例: `{"输入参数键: 输入参数值": "匹配到的模型名称"}`
4.  **完整性**: 确保为每一个提供的“待匹配输入参数”都找到一个匹配项，且只从“可用标准模型库条目”中选择。
"""

# 用户提示模板：包含待匹配项和候选标准项
USER_PROMPT_TEMPLATE = """
请根据语义相似度，将以下“待匹配输入参数”列表中的每一项，与“可用标准模型库条目”列表中的一个条目进行最佳匹配。请仔细考虑每个标准模型库条目组内的完整信息。

**待匹配输入参数:**
{failed_inputs_str}

**可用标准模型库条目 (按模型名称分组，包含内容示例):**
{available_models_str}

请严格按照以下 JSON 格式返回所有待匹配输入参数的匹配结果:
```json
{{
  "输入参数键1: 输入参数值1": "匹配到的模型名称1",
  "输入参数键2: 输入参数值2": "匹配到的模型名称2",
  ...
}}
```
确保每个输入参数都找到一个匹配项，并且每个可用标准模型库条目只被使用一次。
"""


class ModelMatcher:
    """
    负责将输入参数与标准 CSV 模型库进行匹配的核心类。
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
        self.fuzzy_threshold = 0.45  # 模糊匹配相似度阈值

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
        try:
            with open(self.input_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("输入 JSON 文件顶层必须是一个字典。")
                logger.info(f"成功从 {self.input_json_path} 加载输入数据。")
                return data
        except FileNotFoundError:
            logger.error(f"输入 JSON 文件未找到: {self.input_json_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"无法解析输入 JSON 文件: {self.input_json_path}")
            raise
        except Exception as e:
            logger.error(f"加载输入 JSON 文件时发生未知错误: {e}", exc_info=True)
            raise

    def _load_csv_data(self) -> Dict[str, Dict[str, Any]]:
        """加载并处理所有 CSV 文件，按 'model' 列分组。"""
        all_csv_data = {}
        required_columns = {'model', 'code', 'description', 'remark'}

        for category, paths in self.csv_list_map.items():
            logger.debug(f"开始加载类别 '{category}' 的 CSV 文件: {paths}")
            for csv_path_str in paths:
                csv_path = Path(csv_path_str)
                if not csv_path.is_file():
                    logger.warning(f"CSV 文件未找到，跳过: {csv_path}")
                    continue
                try:
                    # 尝试不同的编码读取 CSV
                    try:
                        df = pd.read_csv(csv_path, dtype=str).fillna(
                            '')  # 读取时填充 NaN 为空字符串
                    except UnicodeDecodeError:
                        logger.warning(f"使用 utf-8 读取 {csv_path} 失败，尝试 gbk...")
                        df = pd.read_csv(
                            csv_path, encoding='gbk', dtype=str).fillna('')

                    # 检查必需的列
                    if not required_columns.issubset(df.columns):
                        logger.error(f"CSV 文件 {csv_path} 缺少必需的列。"
                                     f"需要: {required_columns}, 实际: {set(df.columns)}")
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
                            logger.warning(f"模型 '{model_name_str}' 在多个 CSV 文件中找到。"
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

        logger.info(f"CSV 数据加载完成。共加载 {len(all_csv_data)} 个唯一的模型组。")
        return all_csv_data

    def _get_combined_string(self, data: Dict[str, Any] | Tuple[str, str]) -> str:
        """将字典或键值对组合成用于匹配的单一字符串。"""
        if isinstance(data, tuple):  # 输入键值对
            key, value = data
            return f"{key}: {value}"
        elif isinstance(data, dict):  # CSV 行
            # 组合 model, code, description, remark 用于匹配
            # 可以根据需要调整组合方式
            return f"{data.get('model', '')} {data.get('code', '')} {data.get('description', '')} {data.get('remark', '')}".strip()
        return ""

    def _fuzzy_match(self) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Tuple[str, str]]]:
        """
        执行模糊匹配。

        Returns:
            Tuple[Dict[str, List[Dict[str, Any]]], List[Tuple[str, str]]]:
                - fuzzy_matches: 成功匹配的结果 {"input_key: value": [matched_rows]}
                - failed_fuzzy_matches: 匹配失败或分数过低的输入项 [(input_key, input_value), ...]
        """
        fuzzy_matches = {}
        failed_fuzzy_matches = []
        available_models = {name: data for name,
                            data in self.csv_data.items() if not data['used']}

        logger.info("开始模糊匹配...")
        for input_key, input_value in self.input_data.items():
            input_item_tuple = (input_key, input_value)
            input_str = self._get_combined_string(input_item_tuple)
            best_match_model = None
            best_score = -1

            # 遍历所有未使用的模型组
            for model_name, model_data in available_models.items():
                if model_data['used']:  # 双重检查，理论上 available_models 不应包含 used 的
                    continue

                # 计算输入字符串与模型组内每一行的最高相似度
                # 使用 token_sort_ratio 忽略词序差异
                model_group_scores = [
                    fuzz.token_sort_ratio(
                        input_str, self._get_combined_string(row))
                    for row in model_data['rows']
                ]
                current_model_best_score = max(
                    model_group_scores) if model_group_scores else 0

                # 更新最佳匹配
                if current_model_best_score > best_score:
                    best_score = current_model_best_score
                    best_match_model = model_name

            # 判断是否达到阈值
            if best_match_model and best_score >= self.fuzzy_threshold * 100:  # thefuzz 返回 0-100
                match_key = f"'{input_key}': '{input_value}'"  # 修改键格式，添加单引号
                matched_rows = self.csv_data[best_match_model]['rows']
                fuzzy_matches[match_key] = matched_rows
                self.csv_data[best_match_model]['used'] = True  # 标记为已使用
                available_models[best_match_model]['used'] = True  # 更新可用列表状态
                # 使用 logger.info 记录成功的匹配键
                logger.info(
                    f"模糊匹配成功: 输入键 '{input_key}' -> 模型 '{best_match_model}' (分数: {best_score})")
                # logger.debug(f"模糊匹配成功 (详细): '{match_key}' -> '{best_match_model}' (分数: {best_score})") # 保留 debug 级别的详细信息
            else:
                failed_fuzzy_matches.append(input_item_tuple)
                logger.debug(
                    f"模糊匹配失败或分数低: '{input_key}: {input_value}' (最高分: {best_score}, 最佳模型: {best_match_model})")

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
            Dict[str, List[Dict[str, Any]]]: LLM 匹配成功的结果 {"input_key: value": [matched_rows]}。
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
        failed_inputs_str = "\n".join(
            [f"- {self._get_combined_string(item)}" for item in failed_inputs])

        # 准备可用模型列表字符串，包含更丰富的上下文信息
        available_models_str_parts = []
        for name, data in available_models.items():
            # 提取模型组内的一些关键信息作为上下文
            context_lines = []
            # 最多显示前 3 条记录的关键信息 (code: description)
            for i, row in enumerate(data['rows']):
                if i >= 3:
                    context_lines.append("  ...")
                    break
                code = row.get('code', '')
                desc = row.get('description', '')
                # 限制描述长度，避免过长
                context_lines.append(
                    f"  - code: {code}, desc: {desc[:50]}{'...' if len(desc) > 50 else ''}")

            model_context = "\n".join(
                context_lines) if context_lines else "  (无详细条目信息)"
            available_models_str_parts.append(
                f"- 模型名称: {name}\n{model_context}")
        available_models_str = "\n\n".join(
            available_models_str_parts)  # 使用双换行分隔不同的模型组

        user_prompt = USER_PROMPT_TEMPLATE.format(
            failed_inputs_str=failed_inputs_str,
            available_models_str=available_models_str
        )

        # 调用 LLM
        llm_response = call_llm_for_match(
            SYSTEM_PROMPT, user_prompt, expect_json=True)

        if not llm_response or isinstance(llm_response, str) or llm_response.get("error"):
            logger.error(f"LLM 调用失败或返回错误: {llm_response}")
            # 将所有失败的输入记录为无法匹配
            self.unmatched_inputs.update({k: v for k, v in failed_inputs})
            return llm_matches

        # 处理 LLM 响应
        try:
            # llm_response 预期是 {"input_key: value": "matched_model_name", ...}
            if not isinstance(llm_response, dict):
                logger.error(f"LLM 响应不是预期的字典格式: {llm_response}")
                self.unmatched_inputs.update({k: v for k, v in failed_inputs})
                return llm_matches

            processed_inputs = set()
            for input_str, matched_model_name in llm_response.items():
                # 验证匹配的模型是否存在且可用
                if matched_model_name in available_models and not self.csv_data[matched_model_name]['used']:
                    # 从原始 failed_inputs 中找到对应的键值对
                    original_input_tuple = None
                    for k, v in failed_inputs:
                        if f"{k}: {v}" == input_str:
                            original_input_tuple = (k, v)
                            break

                    if original_input_tuple:
                        # 修改键格式，添加单引号
                        match_key = f"'{original_input_tuple[0]}': '{original_input_tuple[1]}'"
                        matched_rows = self.csv_data[matched_model_name]['rows']
                        llm_matches[match_key] = matched_rows
                        # 标记为已使用
                        self.csv_data[matched_model_name]['used'] = True
                        processed_inputs.add(original_input_tuple)
                        # 使用 logger.info 记录成功的匹配，使用新的键格式
                        logger.info(
                            f"LLM 匹配成功: {match_key} -> 模型 '{matched_model_name}'")
                        # logger.debug(f"LLM 匹配成功 (详细): {match_key} -> '{matched_model_name}'") # 保留 debug 级别
                    else:
                        # input_str 仍然是 "key: value" 格式，用于查找
                        logger.warning(f"LLM 返回了无法在失败列表中找到的输入: '{input_str}'")
                elif matched_model_name in self.csv_data and self.csv_data[matched_model_name]['used']:
                    # input_str 仍然是 "key: value" 格式
                    logger.warning(
                        f"LLM 尝试匹配已使用的模型 '{matched_model_name}' 用于输入 '{input_str}'")
                else:  # 模型名称不存在
                    # input_str 仍然是 "key: value" 格式
                    logger.warning(
                        f"LLM 返回了无效的模型名称 '{matched_model_name}' 用于输入 '{input_str}'")

            # 将 LLM 未能成功匹配或处理的项记录为无法匹配
            remaining_failed = [
                item for item in failed_inputs if item not in processed_inputs]
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
            Dict[str, List[Dict[str, Any]]]: 最终的匹配结果 {"input_key: value": [matched_rows]}。
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
        unmatched_models = [name for name, data in self.csv_data.items() if not data['used']]
        if unmatched_models:
            logger.warning(f"有 {len(unmatched_models)} 个标准模型组未被任何输入参数匹配:")
            for model_name in unmatched_models:
                logger.warning(f"  - 未匹配模型组: '{model_name}'")
        # else: # 如果不需要在所有模型都匹配时输出信息，可以注释掉这部分
        #     logger.info("所有标准模型组都已成功匹配或尝试匹配。")

        return self.matched_results

    # def save_results(self, output_path: str): # 移除保存结果的方法
    #     """
    #     将匹配结果保存到 JSON 文件。
    #
    #     Args:
    #         output_path (str): 输出 JSON 文件的路径。
    #     """
    #     output_path_obj = Path(output_path)
    #     output_path_obj.parent.mkdir(parents=True, exist_ok=True) # 确保目录存在
    #
    #     try:
    #         with open(output_path_obj, 'w', encoding='utf-8') as f:
    #             # 为了 JSON 可读性，将匹配结果的 key 转换回原始字典格式可能更好
    #             # 但当前是按 "key: value" 字符串作为键存储的
    #             json.dump(self.matched_results, f, ensure_ascii=False, indent=4)
    #         logger.info(f"匹配结果已保存到: {output_path_obj}")
    #     except Exception as e:
    #         logger.error(f"保存结果到 {output_path_obj} 时出错: {e}", exc_info=True)


# --- 主执行逻辑 (示例) ---
if __name__ == "__main__":
    logger.info("开始执行 ModelMatcher 示例...")

    # 1. 定义 CSV 列表映射 (请根据实际情况修改路径)
    #    使用相对路径或绝对路径均可，但要确保相对于项目根目录或脚本位置正确
    example_csv_map = {
        "transmitter": [],
        "sensor": [
            "libs/standard/sensor/HZ.csv"
        ],
        "tg": [
            "libs/standard/tg/TG_3.csv"
        ]
    }
    # 2. 定义输入 JSON 文件路径 (请确保此文件存在且格式正确)
    example_input_json = "data/output/test.json"  # 使用你之前提供的测试文件

    # 3. 检查输入文件是否存在
    if not Path(example_input_json).is_file():
        logger.error(f"示例输入文件未找到: {example_input_json}")
        logger.error("请确保 'data/output/test.json' 文件存在并包含正确的 JSON 数据。")
        sys.exit(1)  # 退出脚本

    # 4. 创建 Matcher 实例并执行匹配
    try:
        matcher = ModelMatcher(csv_list_map=example_csv_map,
                               input_json_path=example_input_json)
        final_matches = matcher.match()

        # 5. 打印结果
        print("\n--- 最终匹配结果 ---")
        print(json.dumps(final_matches, indent=4, ensure_ascii=False))

        if matcher.unmatched_inputs:
            print("\n--- 未能匹配的输入项 ---")
            print(json.dumps(matcher.unmatched_inputs,
                  indent=4, ensure_ascii=False))

        # 移除文件保存逻辑
        # output_file = "data/output/matching_results.json"
        # matcher.save_results(output_file)
        # print(f"\n结果已保存到: {output_file}")

    except ImportError as e:
        logger.error(f"初始化 ModelMatcher 失败，可能是缺少库: {e}")
    except FileNotFoundError as e:
        logger.error(f"文件未找到错误: {e}")
    except Exception as e:
        logger.error(f"执行 ModelMatcher 示例时发生意外错误: {e}", exc_info=True)

    logger.info("ModelMatcher 示例执行完毕。")
