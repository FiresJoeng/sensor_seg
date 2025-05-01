# -*- coding: utf-8 -*-
import json
import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中以便导入 config 和 llm
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import settings
    # 尝试从新的位置导入 LLM 相关函数
    from src.standard_matcher.llm import call_llm_for_match
except ImportError as e:
    print(
        f"ERROR in fetch_csvlist.py: Failed to import dependencies - {e}. Check project structure and PYTHONPATH.", file=sys.stderr)
    # 如果无法导入配置或 LLM 工具，则无法继续
    sys.exit(1) # 退出脚本

# 配置日志记录器
# 注意：如果 utils.logging_config 存在且配置了全局日志，这里可能不需要重复配置
# 但为了脚本独立性，这里保留一个基本的配置
logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 文件路径定义 ---
# 优先使用 settings 中的路径，如果未定义则使用相对于项目根目录的默认路径
DEFAULT_INPUT_JSON_PATH = project_root / "data" / "output" / "test.json"
DEFAULT_INDEX_JSON_PATH = project_root / "libs" / "standard" / "index.json"

INPUT_JSON_PATH = Path(getattr(settings, 'INPUT_JSON_PATH', DEFAULT_INPUT_JSON_PATH))
INDEX_JSON_PATH = Path(getattr(settings, 'INDEX_JSON_PATH', DEFAULT_INDEX_JSON_PATH))

# --- LLM 相关定义 ---
# 更新 System Prompt 以要求 JSON 输出
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

def fetch_csv_lists(input_json_path: Path = INPUT_JSON_PATH, index_json_path: Path = INDEX_JSON_PATH) -> dict:
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
        requirements_str = json.dumps(input_data, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"将输入数据格式化为 JSON 字符串时出错: {e}")
        return {} # 如果无法格式化输入，则无法继续

    # 3. 收集所有产品类型的可用关键词
    all_keywords_data = {}
    product_types_to_process = list(index_data.keys()) # 获取所有产品类型

    for product_type in product_types_to_process:
        keywords_dict = index_data.get(product_type, {})
        if not isinstance(keywords_dict, dict):
            logger.warning(f"产品类型 '{product_type}' 在索引文件中的值不是字典，跳过。")
            product_types_to_process.remove(product_type) # 从待处理列表中移除
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
    llm_response = call_llm_for_match(SYSTEM_PROMPT, user_prompt, expect_json=True)

    # 6. 处理 LLM 响应并确定最终关键词
    matched_keywords = {}
    if isinstance(llm_response, dict) and 'error' not in llm_response:
        logger.info(f"LLM 成功返回 JSON 响应: {llm_response}")
        matched_keywords = llm_response # 直接使用返回的字典
    elif llm_response is None:
        logger.warning("LLM 调用因客户端未初始化或API密钥问题而跳过。所有产品将使用默认值。")
    elif isinstance(llm_response, dict) and 'error' in llm_response:
        logger.error(f"LLM 调用失败。所有产品将使用默认值。错误: {llm_response.get('details', llm_response['error'])}")
    else:
        logger.error(f"LLM 返回了非预期的响应类型 ({type(llm_response)}) 或无效的 JSON。所有产品将使用默认值。响应: {llm_response}")

    # 7. 遍历所有需要处理的产品类型，确定最终 CSV 列表
    for product_type in product_types_to_process:
        keywords_dict = index_data[product_type] # 我们已经确认过它是字典
        available_keywords = all_keywords_data.get(product_type, []) # 获取该类型的可用关键词

        # 从 LLM 响应中获取该产品类型的匹配关键词，如果响应无效或缺失则为 None
        llm_matched_keyword = matched_keywords.get(product_type)

        selected_keyword = "默认" # 默认为 "默认"

        if isinstance(llm_matched_keyword, str) and llm_matched_keyword != "默认":
            # 验证 LLM 返回的非默认关键词是否有效
            if llm_matched_keyword in available_keywords:
                selected_keyword = llm_matched_keyword
                logger.info(f"LLM 为产品 '{product_type}' 成功匹配到有效关键词: '{selected_keyword}'")
            else:
                logger.warning(f"LLM 为产品 '{product_type}' 返回的关键词 '{llm_matched_keyword}' 不在可用列表中。将使用默认值。")
        elif llm_matched_keyword == "默认":
             logger.info(f"LLM 为产品 '{product_type}' 明确选择了 '默认'。")
             selected_keyword = "默认"
        else:
            # 如果 LLM 响应中没有该产品类型，或值无效，则使用默认值
             if product_type not in matched_keywords:
                 logger.warning(f"LLM 响应中未包含产品类型 '{product_type}'。将使用默认值。")
             else:
                 logger.warning(f"LLM 为产品 '{product_type}' 返回了无效值 '{llm_matched_keyword}'。将使用默认值。")
             selected_keyword = "默认"


        # 获取最终的 CSV 列表
        if selected_keyword in keywords_dict:
            csv_list = keywords_dict[selected_keyword]
            result_csv_lists[product_type] = csv_list
            logger.info(f"产品 '{product_type}' 最终选择关键词 '{selected_keyword}'，对应 CSV 列表: {csv_list}")
        else:
            # 理论上 "默认" 应该总是在 keywords_dict 中
            logger.error(f"严重错误：无法在索引中找到产品 '{product_type}' 的关键词 '{selected_keyword}' (即使是默认值)。跳过此产品。")

    logger.info("所有产品类型处理完毕。")
    return result_csv_lists

if __name__ == "__main__":
    # 检查必要的环境变量是否设置（主要用于 LLM）
    if not settings.LLM_API_KEY or settings.LLM_API_KEY == "YOUR_API_KEY_HERE":
        logger.warning("LLM API 密钥未在 .env 文件或环境变量中配置。LLM 匹配功能将受限或不可用。")
        # 这里可以选择是否继续执行（使用默认值）或退出
        # print("错误：LLM API 密钥未配置，无法执行匹配。请在 .env 文件中设置 DEEPSEEK_API_KEY。", file=sys.stderr)
        # sys.exit(1)

    # 执行主逻辑
    final_csv_lists = fetch_csv_lists()

    # 打印结果
    if final_csv_lists:
        print("\n--- 生成的 CSV 列表映射 ---")
        print(json.dumps(final_csv_lists, indent=4, ensure_ascii=False))
        print("---------------------------\n")
        logger.info("脚本执行成功。")
    else:
        print("\n未能成功生成 CSV 列表映射。请检查日志获取详细信息。\n")
        logger.error("脚本执行未能生成结果。")
        sys.exit(1) # 以错误码退出

    sys.exit(0) # 正常退出
