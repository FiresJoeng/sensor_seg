# test_kk/llm_standardizer_with_vector_ref.py
import json
import logging
import sys
import re # Import regex for JSON extraction
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_fixed # Removed retry_if_exception_type as we retry on any Exception

# --- Add project root to sys.path ---
# This allows absolute imports from the project root (e.g., src.xxx)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    # print(f"DEBUG: Added {project_root} to sys.path in llm_standardizer_with_vector_ref.py") # Removed debug print

try:
    # from openai import APIConnectionError # Removed unused import
    from zhipuai import ZhipuAI # Import ZhipuAI
    from config import settings # Assuming settings.py contains necessary configs like META_FIELD_*
    from src.parameter_standardizer.search_service import SearchService
    from src.utils.logging_config import setup_logging # Assuming you have this utility
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules: {e}. Ensure all dependencies are installed and PYTHONPATH is correct.", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
# TODO: Consider moving API_KEY to environment variables or a config file for security.
API_KEY = "afd6f833ae76446eb2efc86341daf7c4.R5kdpHPK8YK1blyU" # ZhipuAI API Key
MODEL_NAME = "glm-4-plus" # ZhipuAI Model Name

INPUT_JSON_PATH = project_root / "data/output/广东石化报价表_extracted_parameters.json"
PROMPT_TEMPLATE_PATH = project_root / "test_kk/standardized_prompt.txt"
OUTPUT_JSON_PATH = project_root / "data/output/广东石化报价表_llm_standardized.json"
LOG_FILE_PATH = project_root / "data/output/llm_standardizer.log"

# --- Setup Logging ---
# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE_PATH, mode='w'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- Constants ---
VECTOR_SUGGESTION_PLACEHOLDER = "{{vector_db_suggestions}}"
INPUT_JSON_MARKER = "1.  **实际设备列表 (JSON):**"
STANDARD_TABLE_MARKER = "2.  **标准参数表 (参考资料):**"

# --- Helper Functions ---
def load_json_data(file_path: Path) -> Optional[Dict[str, Any]]:
    """Loads JSON data from a file."""
    if not file_path.exists():
        logger.error(f"Input JSON file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None

def load_prompt_template(file_path: Path) -> Optional[str]:
    """Loads the prompt template from a file."""
    if not file_path.exists():
        logger.error(f"Prompt template file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading prompt template file {file_path}: {e}")
        return None

# Removed unused format_vector_suggestions function

def extract_json_from_response(response_content: str) -> Optional[str]:
    """Extracts the first valid JSON object string from the LLM response."""
    # Try finding JSON within ```json ... ```
    match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
    if match:
        logger.debug("Found JSON within ```json block.")
        return match.group(1)

    # Try finding JSON starting with { and ending with } using a non-greedy match
    # This is safer than a greedy match if multiple JSON-like structures exist.
    match = re.search(r'(\{.*?\})(?:\s*\Z|\s*[^`])', response_content, re.DOTALL)
    if match:
        potential_json = match.group(1)
        # Basic validation: does it start with { and end with }?
        if potential_json.startswith('{') and potential_json.endswith('}'):
            # Attempt to parse to ensure it's likely the main JSON object
            try:
                json.loads(potential_json)
                logger.debug("Found JSON object using regex search {.*?} ")
                return potential_json
            except json.JSONDecodeError:
                logger.debug("Regex match {.*?} failed to parse, continuing search.")
                pass # Continue searching

    # Fallback: Find the first '{' and the last '}' as a last resort
    start_index = response_content.find('{')
    end_index = response_content.rfind('}')
    if start_index != -1 and end_index != -1 and end_index > start_index:
        potential_json = response_content[start_index : end_index + 1]
        logger.debug("Found JSON using first '{' and last '}' fallback.")
        # We return this even if it doesn't parse, the caller will handle the error
        return potential_json

    logger.warning("无法在响应中定位 JSON 对象。")
    return None

# Define a callback function to log when retries are exhausted
def log_retry_error(retry_state):
    logger.error(f"OpenAI API 调用在 {retry_state.attempt_number} 次尝试后最终失败: {retry_state.outcome.exception()}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    # Retry on any exception for simplicity with ZhipuAI
    retry_error_callback=log_retry_error
)
def call_zhipuai_api(client: ZhipuAI, prompt: str) -> Optional[Dict[str, Any]]:
    """Calls the ZhipuAI API with retry logic and returns the parsed JSON response."""
    logger.info("--- 尝试调用 ZhipuAI API ---")
    logger.debug(f"发送请求到模型: {MODEL_NAME}")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                # Optional: Add a system prompt if needed
                # {"role": "system", "content": "You are an expert in standardizing equipment parameters."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=12000, # Adjust as needed
        )

        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            logger.error("ZhipuAI API response is missing expected content.")
            return None

        response_content = response.choices[0].message.content
        logger.info("--- ZhipuAI API 响应接收成功 ---")
        logger.debug(f"Raw Response Content:\n{response_content}")

        # --- Extract JSON using the helper function ---
        json_string = extract_json_from_response(response_content)

        if not json_string:
             logger.error("无法从 API 响应中提取 JSON 字符串。")
             logger.error(f"原始响应: {response_content}")
             return None

        logger.debug(f"提取的 JSON 字符串:\n{json_string}")
        try:
            parsed_json = json.loads(json_string)
            logger.info("成功解析 API 返回的 JSON。")
            return parsed_json
        except json.JSONDecodeError as json_err:
            # Log the specific error and the string that failed parsing
            logger.error(f"无法解析 ZhipuAI API 返回的 JSON (提取后): {json_err}")
            logger.error(f"JSON 解析错误发生在第 {json_err.lineno} 行, 第 {json_err.colno} 列 (char {json_err.pos})")
            logger.error(f"尝试解析的内容:\n{json_string}")
            # logger.error(f"原始响应内容:\n{response_content}") # Already logged above
            return None # Indicate failure to parse

    # except APIConnectionError as conn_err: # Keep or adapt based on ZhipuAI exceptions
    #     logger.error(f"ZhipuAI API 连接错误 (在重试后仍然发生): {conn_err}")
    #     return None
    except Exception as e:
        # Catch other unexpected errors during the API call or processing
        # Tenacity will handle retries based on the decorator settings
        logger.error(f"处理 ZhipuAI API 响应时发生意外错误 (可能在重试中): {e}", exc_info=True)
        # If retries are exhausted, tenacity raises the exception, otherwise this logs intermediate errors.
        # Let tenacity handle raising the final error after retries.
        raise # Re-raise the exception so tenacity can handle retry logic


def save_json_data(data: Dict[str, Any], file_path: Path):
    """Saves data to a JSON file."""
    try:
        # Ensure the output directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"成功将标准化结果写入: {file_path}")
    except Exception as e:
        logger.error(f"写入 JSON 文件 {file_path} 时出错: {e}")

# --- Refactored Main Logic ---

def fetch_vector_suggestions(
    search_service: SearchService,
    device_list: List[Dict[str, Any]]
) -> Dict[Tuple[int, str], str]:
    """Phase 1: Pre-fetches vector suggestions for all parameters."""
    logger.info("--- [阶段 1] 开始预获取所有向量建议 ---")
    all_vector_suggestions: Dict[Tuple[int, str], str] = {}
    total_params_to_search = sum(len(d.get("共用参数", {})) for d in device_list)
    processed_params_count = 0
    logger.info(f"共需查询 {total_params_to_search} 个参数的向量建议。")

    for i, device_data in enumerate(device_list):
        device_tag_str = ", ".join(device_data.get("位号", [f"未知设备_{i+1}"]))
        actual_params = device_data.get("共用参数", {})
        if not actual_params:
            logger.debug(f"设备 {device_tag_str} 无共用参数，跳过向量搜索。")
            continue

        logger.debug(f"处理设备 {device_tag_str} 的向量建议...")
        for actual_key, actual_value in actual_params.items():
            processed_params_count += 1
            logger.debug(f"  ({processed_params_count}/{total_params_to_search}) 查询参数: '{actual_key}' = '{actual_value}'")
            query_text_combined = f"{actual_key}: {actual_value}"
            # Consider adding a public method to SearchService for encoding if possible
            query_embedding_array = search_service._encode_query(query_text_combined)

            suggestion_text_for_param = f"参数 '{actual_key}':\n"
            if query_embedding_array is None:
                logger.warning(f"    无法编码参数 '{actual_key}', 跳过向量建议。")
                suggestion_text_for_param += "  - 编码失败"
            else:
                query_embedding_list = [query_embedding_array.tolist()]
                try:
                    # Consider adding a wrapper in SearchService for querying
                    vector_results = search_service.vector_store.query_collection(
                        query_embeddings=query_embedding_list,
                        n_results=5,
                        include_fields=['metadatas', 'distances'] # Ensure these fields exist
                    )
                    if vector_results and vector_results.get('ids', [[]])[0]:
                        metadatas = vector_results.get('metadatas', [[]])[0]
                        distances = vector_results.get('distances', [[]])[0]
                        param_suggestions_lines = []
                        for meta, dist in zip(metadatas, distances):
                            # Use .get() with defaults for safety
                            std_name = meta.get(settings.META_FIELD_PARAM_TYPE, "N/A")
                            std_value = meta.get(settings.META_FIELD_STANDARD_VALUE, "N/A")
                            param_suggestions_lines.append(f"  - 标准名: '{std_name}', 标准值: '{std_value}' (距离: {dist:.4f})")
                        if param_suggestions_lines:
                            suggestion_text_for_param += "\n".join(param_suggestions_lines)
                        else:
                            suggestion_text_for_param += "  - 未找到有效元数据"
                    else:
                        suggestion_text_for_param += "  - 未找到向量建议"
                except Exception as e:
                    logger.error(f"    为参数 '{actual_key}' 查询向量数据库时出错: {e}", exc_info=False)
                    suggestion_text_for_param += f"  - 查询向量数据库出错: {e}"

            all_vector_suggestions[(i, actual_key)] = suggestion_text_for_param

    logger.info("--- [阶段 1] 所有向量建议预获取完成 ---")
    return all_vector_suggestions

def construct_llm_prompt(
    prompt_template: str,
    device_tag: List[str],
    actual_params: Dict[str, Any],
    combined_suggestions_text: str
) -> str:
    """Constructs the prompt for the LLM API call."""
    input_device_json_str = json.dumps({
        "设备列表": [{
            "位号": device_tag,
            "共用参数": actual_params
        }]
    }, ensure_ascii=False, indent=2)

    # Replace placeholder for suggestions first
    prompt_with_suggestions = prompt_template.replace(VECTOR_SUGGESTION_PLACEHOLDER, combined_suggestions_text)

    # Then insert the JSON data using markers (less prone to issues than replacing suggestions within JSON)
    prompt_parts = prompt_with_suggestions.split(INPUT_JSON_MARKER)
    if len(prompt_parts) == 2:
        before_json = prompt_parts[0] + INPUT_JSON_MARKER + "\n" + input_device_json_str + "\n\n"
        after_json_part = prompt_parts[1]
        # Ensure the standard table marker exists to avoid inserting JSON in the wrong place
        standard_table_start_index = after_json_part.find(STANDARD_TABLE_MARKER)
        if standard_table_start_index != -1:
            after_json = after_json_part[standard_table_start_index:]
            prompt = before_json + after_json
        else:
            logger.warning(f"无法在 Prompt 模板中找到 '{STANDARD_TABLE_MARKER}'，JSON 插入可能不准确。")
            # Fallback: Replace the marker directly, hoping it's unique enough
            prompt = prompt_with_suggestions.replace(INPUT_JSON_MARKER, INPUT_JSON_MARKER + "\n" + input_device_json_str + "\n\n")
    else:
        logger.error(f"无法在 Prompt 模板中准确找到 '{INPUT_JSON_MARKER}' 进行 JSON 插入。")
        prompt = prompt_with_suggestions # Return prompt without JSON insertion if marker not found correctly

    return prompt

def process_devices_with_llm(
    client: ZhipuAI,
    device_list: List[Dict[str, Any]],
    all_vector_suggestions: Dict[Tuple[int, str], str],
    prompt_template: str
) -> List[Dict[str, Any]]:
    """Phase 2: Processes devices using LLM with pre-fetched suggestions."""
    logger.info("--- [阶段 2] 开始使用 LLM 进行标准化 ---")
    standardized_results_list = []

    for i, device_data in enumerate(device_list):
        device_tag = device_data.get("位号", [f"未知设备_{i+1}"])
        device_tag_str = ", ".join(device_tag)
        logger.info(f"--- ({i+1}/{len(device_list)}) 开始处理设备: {device_tag_str} ---")

        actual_params = device_data.get("共用参数", {})
        if not actual_params:
            logger.warning(f"设备 {device_tag_str} 没有 '共用参数'，跳过 LLM 处理。")
            standardized_results_list.append({
                "位号": device_tag,
                "标准化共用参数": {},
                "处理说明": "无共用参数可处理"
            })
            continue

        # Combine pre-fetched suggestions for this device
        device_suggestions_list = [
            all_vector_suggestions.get((i, actual_key), f"参数 '{actual_key}': 预获取建议时出错或未找到")
            for actual_key in actual_params.keys()
        ]
        combined_suggestions_text = "\n".join(device_suggestions_list)
        logger.debug(f"组合后的向量建议:\n{combined_suggestions_text}")

        # Construct Prompt
        prompt = construct_llm_prompt(prompt_template, device_tag, actual_params, combined_suggestions_text)
        logger.debug(f"构建的 Prompt (部分):\n{prompt[:500]}...") # Log beginning of prompt

        # Call LLM API
        llm_response_json = None
        try:
            llm_response_json = call_zhipuai_api(client, prompt)
        except Exception as e: # Catch final error after tenacity retries
            logger.error(f"调用 ZhipuAI API 最终失败 (设备 {device_tag_str}): {e}", exc_info=True)
            # llm_response_json remains None

        # Process LLM Response
        if llm_response_json and isinstance(llm_response_json, dict):
            standardized_device_list_from_llm = llm_response_json.get("标准化设备组列表", [])
            if standardized_device_list_from_llm and isinstance(standardized_device_list_from_llm, list) and len(standardized_device_list_from_llm) > 0:
                # Assume LLM returns one device result per call as per current logic
                processed_device_data = standardized_device_list_from_llm[0]
                if isinstance(processed_device_data, dict) and "位号" in processed_device_data and "标准化共用参数" in processed_device_data:
                    standardized_results_list.append(processed_device_data)
                    logger.info(f"成功处理设备 {device_tag_str} 的标准化结果。")
                else:
                    logger.error(f"LLM 返回的设备数据格式不正确: {processed_device_data}")
                    standardized_results_list.append({
                        "位号": device_tag, "标准化共用参数": {}, "处理说明": "LLM返回格式错误", "原始LLM响应": llm_response_json
                    })
            else:
                logger.error(f"LLM 响应中未找到有效的 '标准化设备组列表': {llm_response_json}")
                standardized_results_list.append({
                    "位号": device_tag, "标准化共用参数": {}, "处理说明": "LLM未返回有效列表", "原始LLM响应": llm_response_json
                })
        else:
            logger.error(f"未能从 LLM 获取设备 {device_tag_str} 的有效标准化结果。")
            standardized_results_list.append({
                "位号": device_tag, "标准化共用参数": {}, "处理说明": "LLM调用失败或返回无效"
            })

    logger.info("--- [阶段 2] LLM 参数标准化流程完成 ---")
    return standardized_results_list


def main():
    """Main function to run the standardization pipeline."""
    logger.info("--- 开始 LLM 参数标准化流程 (带向量参考 - 两阶段优化) ---")

    # 1. Load Input Data and Prompt Template
    input_data = load_json_data(INPUT_JSON_PATH)
    prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)
    if not input_data or not prompt_template:
        logger.error("无法加载输入数据或 Prompt 模板，流程终止。")
        return

    device_list = input_data.get("设备列表", [])
    if not device_list:
        logger.warning("输入 JSON 文件中未找到 '设备列表' 或列表为空。")
        save_json_data({"标准化设备组列表": []}, OUTPUT_JSON_PATH)
        return

    # 2. Initialize Services
    logger.info("初始化 SearchService...")
    try:
        search_service = SearchService()
        if not search_service.is_ready():
            logger.error("SearchService 未就绪，无法获取向量建议。流程终止。")
            return
    except Exception as e:
        logger.error(f"初始化 SearchService 失败: {e}", exc_info=True)
        return

    logger.info("初始化 ZhipuAI 客户端...")
    try:
        client = ZhipuAI(api_key=API_KEY)
    except Exception as e:
        logger.error(f"初始化 ZhipuAI 客户端失败: {e}", exc_info=True)
        return

    # 3. Phase 1: Fetch Vector Suggestions
    all_vector_suggestions = fetch_vector_suggestions(search_service, device_list)

    # 4. Phase 2: Process Devices with LLM
    standardized_results = process_devices_with_llm(
        client, device_list, all_vector_suggestions, prompt_template
    )

    # 5. Phase 3: Save Final Results
    final_output_data = {"标准化设备组列表": standardized_results}
    save_json_data(final_output_data, OUTPUT_JSON_PATH)
    logger.info("--- LLM 参数标准化流程完成 ---")


if __name__ == "__main__":
    main()
