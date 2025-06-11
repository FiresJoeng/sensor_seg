# -*- coding: utf-8 -*-
"""
模块：获取 CSV 列表 (Fetch CSV List)
功能：负责根据输入的产品要求和索引文件，获取对应的 CSV 文件列表。
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set

# 尝试导入模糊匹配库 (FetchCsvlist 不直接使用 thefuzz，但为了完整性保留相关导入)
# try:
#     from thefuzz import fuzz, process
#     THEFUZZ_AVAILABLE = True
# except ImportError:
#     THEFUZZ_AVAILABLE = False
#     # logger.warning("警告：'thefuzz' 库未安装。模糊匹配功能将不可用。请运行 'pip install thefuzz python-Levenshtein'")

# 确保项目根目录在 sys.path 中以便导入 config, llm
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import settings
    from src.standard_matcher.llm import call_llm_for_match
    # FetchCsvlist 不直接使用 AnalysisJsonProcessor
    # from src.standard_matcher.json_processor import AnalysisJsonProcessor
except ImportError as e:
    logging.getLogger(__name__).critical(
        f"错误：在 fetch_csvlist.py 中导入模块失败 - {e}。"
        f"请检查项目结构和 PYTHONPATH。\n"
        f"项目根目录尝试设置为: {project_root}", exc_info=True)
    raise

# --- 全局配置 ---
logger = logging.getLogger(__name__)

# --- 文件路径定义 ---
DEFAULT_INDEX_JSON_PATH = project_root / "libs" / "standard" / "index.json"
INDEX_JSON_PATH = Path(
    getattr(settings, 'INDEX_JSON_PATH', DEFAULT_INDEX_JSON_PATH))

# TEMP_OUTPUT_DIR 在 standard_matcher.py 的主逻辑中使用，不移到这里

# ==============================================================================
# 1. Fetch CSV List
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
                    f"错误：无法在索引中找到产品 '{product_type}' 的关键词 '{selected_keyword}' (即使是默认值)。跳过此产品。")

        logger.info("所有产品类型处理完毕。")
        return result_csv_lists
