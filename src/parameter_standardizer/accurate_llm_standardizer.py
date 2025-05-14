# src/parameter_standardizer/accurate_llm_standardizer.py
import json
import logging
import sys
import re
import time # Added time import
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_fixed
import pandas as pd # 导入 pandas 库
import io # 导入 io 库

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    # 导入 OpenAI 库
    from openai import OpenAI
    # 假设 API Key 和模型名在 settings 中
    from config import settings
    # from src.parameter_standardizer.search_service import SearchService # 已移除
except ImportError as e:
    # 如果导入失败，打印错误信息并退出
    print(f"ERROR: Failed to import necessary modules in accurate_llm_standardizer.py: {e}. Ensure all dependencies are installed and PYTHONPATH is correct.", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

# --- 常量 ---
# Prompt 模板路径，相对于项目根目录
# 注意：路径分隔符在 Windows 和 Linux/macOS 上可能不同，Path 对象会处理
PROMPT_TEMPLATE_PATH = Path(__file__).parent / "standardized_prompt.txt"
VECTOR_SUGGESTION_PLACEHOLDER = "{{vector_db_suggestions}}"
INPUT_JSON_MARKER = "1.  **实际设备列表 (JSON):**"
STANDARD_TABLE_MARKER = "2.  **标准参数表 (参考资料):**"
# 定义完整语义表 Excel 文件的路径
FULL_SEMANTIC_TABLE_PATH = Path(project_root) / "libs" / "一体化温度变送器语义库.xlsx"

# --- 辅助函数 ---
def load_prompt_template(file_path: Path) -> Optional[str]:
    """Loads the prompt template from a file."""
    # 从文件加载 Prompt 模板
    if not file_path.exists():
        # 如果文件不存在，记录错误并返回 None
        logger.error(f"Prompt template file not found: {file_path}")
        return None
    try:
        # 打开文件并读取内容
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        # 如果读取文件时发生错误，记录错误并返回 None
        logger.error(f"Error reading prompt template file {file_path}: {e}")
        return None

def extract_json_from_response(response_content: str) -> Optional[str]:
    """从 LLM 响应中提取第一个有效的 JSON 对象字符串。"""
    # 从 LLM 响应中提取第一个有效的 JSON 对象字符串
    # 尝试查找 ```json ... ``` 中的 JSON
    match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
    if match:
        # 如果找到，记录调试信息并返回匹配的组
        logger.debug("Found JSON within ```json block.")
        return match.group(1)

    # 尝试使用非贪婪匹配查找以 { 开头并以 } 结尾的 JSON
    match = re.search(r'(\{.*?\})(?:\s*\Z|\s*[^`])', response_content, re.DOTALL)
    if match:
        potential_json = match.group(1)
        # 检查是否以 { 开头并以 } 结尾
        if potential_json.startswith('{') and potential_json.endswith('}'):
            try:
                # 尝试解析以确保它可能是主要的 JSON 对象
                json.loads(potential_json)
                logger.debug("Found JSON object using regex search {.*?} ")
                return potential_json
            except json.JSONDecodeError:
                # 如果解析失败，记录调试信息并继续搜索
                logger.debug("Regex match {.*?} failed to parse, continuing search.")
                pass

    # Fallback: 查找第一个 '{' 和最后一个 '}'
    start_index = response_content.find('{')
    end_index = response_content.rfind('}')
    if start_index != -1 and end_index != -1 and end_index > start_index:
        potential_json = response_content[start_index : end_index + 1]
        logger.debug("Found JSON using first '{' and last '}' fallback.")
        # 即使无法解析也返回，调用者将处理错误
        return potential_json

    # 如果无法定位 JSON 对象，记录警告并返回 None
    logger.warning("无法在响应中定位 JSON 对象。")
    return None

# 移除 ZhipuAI 特定的重试日志函数
# def log_retry_error(retry_state):
#     logger.error(f"ZhipuAI API 调用在 {retry_state.attempt_number} 次尝试后最终失败: {retry_state.outcome.exception()}")

class AccurateLLMStandardizer:
    """
    使用 LLM 进行参数标准化的类。
    移除了向量参考功能。
    """
    # 修改 __init__ 方法，接受 OpenAI 客户端
    def __init__(self, client: OpenAI): # 移除了 search_service 参数
        """
        初始化标准化器。

        Args:
            client: 已初始化的 OpenAI 客户端实例。
        """
        # 初始化 OpenAI 客户端
        # self.search_service = search_service # 已移除
        self.client = client
        # 加载 Prompt 模板
        self.prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)
        if not self.prompt_template:
            # 如果无法加载 Prompt 模板，则引发 ValueError
            raise ValueError("无法加载 Prompt 模板，标准化器无法初始化。")

        # 加载完整语义表 Excel 内容
        self.full_semantic_table_content = self._load_excel_as_csv_string(FULL_SEMANTIC_TABLE_PATH)
        if not self.full_semantic_table_content:
             # 如果无法加载 Excel 内容，记录错误但允许继续（Prompt 中将缺少标准表）
             logger.error(f"无法加载完整语义表 Excel 文件: {FULL_SEMANTIC_TABLE_PATH}。Prompt 中将缺少标准表内容。")
             self.full_semantic_table_content = "无法加载标准参数表。" # 提供一个默认文本

        # 从 settings 获取模型名称，使用通用的 LLM 配置
        self.model_name = settings.LLM_MODEL_NAME
        self.temperature = settings.LLM_TEMPERATURE # 使用 settings 中的温度参数
        self.request_timeout = settings.LLM_REQUEST_TIMEOUT # 使用 settings 中的超时参数
        logger.info(f"AccurateLLMStandardizer 初始化完成，使用模型: {self.model_name}")

    def _load_excel_as_csv_string(self, file_path: Path) -> Optional[str]:
        """
        读取 Excel 文件中的所有表格，合并后将其内容转换为 CSV 格式的字符串。
        参考了 InfoExtractor 中的 Excel 处理逻辑。
        """
        # 读取 Excel 文件中的所有表格，合并后将其内容转换为 CSV 格式的字符串
        if not file_path.exists():
            logger.error(f"Excel file not found: {file_path}")
            return None
        try:
            # 使用 ExcelFile 读取所有 Sheet
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
            all_sheets_df = []

            if not sheet_names:
                logger.warning(f"Excel file {file_path} contains no sheets.")
                return None

            logger.info(f"Reading sheets: {sheet_names} from {file_path.name}")

            for sheet_name in sheet_names:
                try:
                    df = xls.parse(sheet_name)
                    if not df.empty:
                        all_sheets_df.append(df)
                        logger.debug(f"Successfully read sheet '{sheet_name}' with {len(df)} rows.")
                    else:
                        logger.debug(f"Sheet '{sheet_name}' is empty.")
                except Exception as sheet_e:
                    logger.error(f"Error reading sheet '{sheet_name}' from {file_path}: {sheet_e}")
                    # 继续处理下一个 Sheet，即使当前 Sheet 读取失败

            if not all_sheets_df:
                logger.error(f"No data found in any sheets of Excel file: {file_path}")
                return None

            # 合并所有 DataFrame
            combined_df = pd.concat(all_sheets_df, ignore_index=True)
            logger.info(f"Combined data from all sheets into a single DataFrame with {len(combined_df)} rows.")

            # 将合并后的 DataFrame 转换为 CSV 字符串
            csv_buffer = io.StringIO()
            combined_df.to_csv(csv_buffer, index=False, encoding='utf-8')
            csv_content = csv_buffer.getvalue()
            logger.info(f"成功将所有 Excel 表格内容合并并转换为 CSV 格式字符串: {file_path.name}")
            return csv_content

        except FileNotFoundError:
            logger.error(f"Excel file not found: {file_path}")
            return None
        except pd.errors.EmptyDataError:
            logger.error(f"Excel file is empty or contains no valid data: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {e}")
            return None


    def _fetch_suggestions_for_group_data(self, group_data_for_llm: Dict[str, Any]) -> str:
        """为单个设备组数据中的所有参数预获取向量建议，并格式化为字符串。"""
        # 为单个设备组数据中的所有参数预获取向量建议，并格式化为字符串
        suggestions_list = []

        device_groups = group_data_for_llm.get('设备列表', [])
        if not device_groups or not isinstance(device_groups, list) or len(device_groups) == 0:
            logger.debug("输入数据中无设备组或格式不正确，无向量建议。")
            return "无设备组可供查询向量建议。"

        # 只处理第一个设备组，因为这个方法是为单组数据设计的
        device_group = device_groups[0]
        if not isinstance(device_group, dict):
             logger.warning(f"设备组数据格式不正确，无法获取建议。内容: {device_group}")
             return "设备组数据格式不正确，无法获取建议。"

        group_tags = device_group.get('位号', ['未知组'])
        common_params = device_group.get('共用参数', {})
        diff_params = device_group.get('不同参数', {})

        logger.debug(f"为设备组 {', '.join(group_tags)} 的参数获取向量建议...")

        # 统计总参数数量以便日志记录进度
        total_params_count = len(common_params)
        if isinstance(diff_params, dict):
             for param_name, tag_value_map in diff_params.items():
                  if isinstance(tag_value_map, dict):
                       total_params_count += len(tag_value_map)

        processed_params_count = 0

        # 处理共用参数
        if common_params:
            suggestions_list.append("  共用参数:")
            for actual_key, actual_value in common_params.items():
                processed_params_count += 1
                logger.debug(f"    ({processed_params_count}/{total_params_count}) 查询共用参数: '{actual_key}' = '{actual_value}'")
                query_text_combined = f"{actual_key}: {actual_value}"
                suggestion_text_for_param = f"    参数 '{actual_key}':\n"
                try:
                    suggestions_result = self.search_service.get_vector_suggestions(query_text=query_text_combined, n_results=3) # 减少建议数量以控制 Prompt 长度
                    if suggestions_result:
                        for suggestion in suggestions_result:
                            meta = suggestion.get('metadata', {})
                            dist = suggestion.get('distance', float('inf'))
                            std_name = meta.get(settings.META_FIELD_PARAM_TYPE, "N/A")
                            std_value = meta.get(settings.META_FIELD_STANDARD_VALUE, "N/A")
                            suggestion_text_for_param += f"      - 标准名: '{std_name}', 标准值: '{std_value}' (距离: {dist:.4f})\n"
                    else:
                        suggestion_text_for_param += "      - 未找到向量建议\n"
                except Exception as e:
                    logger.error(f"    为共用参数 '{actual_key}' 查询向量数据库时出错: {e}", exc_info=False)
                    suggestion_text_for_param += f"      - 查询向量数据库出错: {e}\n"
                suggestions_list.append(suggestion_text_for_param.strip())

        # 处理不同参数
        if isinstance(diff_params, dict) and diff_params:
            suggestions_list.append("  不同参数:")
            for param_name, tag_value_map in diff_params.items():
                if isinstance(tag_value_map, dict):
                    suggestions_list.append(f"    参数 '{param_name}':")
                    for tag_no, actual_value in tag_value_map.items():
                        processed_params_count += 1
                        logger.debug(f"      ({processed_params_count}/{total_params_count}) 查询不同参数: '{param_name}' (位号 '{tag_no}') = '{actual_value}'")
                        query_text_combined = f"{param_name}: {actual_value}"
                        suggestion_text_for_param = f"      位号 '{tag_no}':\n"
                        try:
                            suggestions_result = self.search_service.get_vector_suggestions(query_text=query_text_combined, n_results=3) # 减少建议数量
                            if suggestions_result:
                                for suggestion in suggestions_result:
                                    meta = suggestion.get('metadata', {})
                                    dist = suggestion.get('distance', float('inf'))
                                    std_name = meta.get(settings.META_FIELD_PARAM_TYPE, "N/A")
                                    std_value = meta.get(settings.META_FIELD_STANDARD_VALUE, "N/A")
                                    suggestion_text_for_param += f"        - 标准名: '{std_name}', 标准值: '{std_value}' (距离: {dist:.4f})\n"
                            else:
                                suggestion_text_for_param += "        - 未找到向量建议\n"
                        except Exception as e:
                            logger.error(f"    为不同参数 '{param_name}' (位号 '{tag_no}') 查询向量数据库时出错: {e}", exc_info=False)
                            suggestion_text_for_param += f"        - 查询向量数据库出错: {e}\n"
                        suggestions_list.append(suggestion_text_for_param.strip())
                else:
                    logger.warning(f"设备组 {', '.join(group_tags)} 的不同参数 '{param_name}' 格式不正确，跳过获取建议。内容: {tag_value_map}")
                    suggestions_list.append(f"    参数 '{param_name}': 格式错误，无法获取建议。")
        elif isinstance(diff_params, dict):
             # diff_params 是空字典
             pass
        else:
             logger.warning(f"设备组 {', '.join(group_tags)} 的不同参数格式不正确，跳过获取建议。内容: {diff_params}")
             suggestions_list.append("  不同参数: 格式错误，无法获取建议。")


        # 返回组合后的建议字符串
            # return "\n".join(suggestions_list) if suggestions_list else "无参数可查询向量建议。" # 这行是 _fetch_suggestions_for_group_data 的一部分，应随方法移除


    # _fetch_suggestions_for_group_data 方法已整体移除

    def _construct_llm_prompt_for_group_data(self, group_data_for_llm: Dict[str, Any]) -> str: # 移除了 combined_suggestions_text 参数
        """为单个设备组数据构建 LLM Prompt。"""
        # 构建输入 JSON 部分，包含单个设备组数据
        input_data_json_str = json.dumps(group_data_for_llm, ensure_ascii=False, indent=2)

        # 不再需要替换建议占位符，因为向量建议功能已移除
        # prompt_with_suggestions = self.prompt_template.replace(VECTOR_SUGGESTION_PLACEHOLDER, combined_suggestions_text)
        # 直接使用 self.prompt_template

        # 插入单组数据 JSON 和完整语义表内容
        prompt_parts = self.prompt_template.split(INPUT_JSON_MARKER) # 基于原始模板分割
        if len(prompt_parts) == 2:
            before_json = prompt_parts[0] + INPUT_JSON_MARKER + "\n" + input_data_json_str + "\n\n"
            after_json_part = prompt_parts[1]
            standard_table_start_index = after_json_part.find(STANDARD_TABLE_MARKER)
            if standard_table_start_index != -1:
                # 在 STANDARD_TABLE_MARKER 后面插入完整语义表内容
                after_json = after_json_part[:standard_table_start_index + len(STANDARD_TABLE_MARKER)] + "\n" + self.full_semantic_table_content + "\n\n" + after_json_part[standard_table_start_index + len(STANDARD_TABLE_MARKER):]
                prompt = before_json + after_json
            else:
                # 如果找不到标准表标记，记录警告并直接替换输入 JSON 标记
                logger.warning(f"无法在 Prompt 模板中找到 '{STANDARD_TABLE_MARKER}'，完整语义表内容将不会被插入。")
                # 此处应基于 self.prompt_template 进行替换，而不是 prompt_with_suggestions
                prompt = self.prompt_template.replace(INPUT_JSON_MARKER, INPUT_JSON_MARKER + "\n" + input_data_json_str + "\n\n")
        else:
            # 如果找不到输入 JSON 标记，记录错误并返回原始 Prompt
            logger.error(f"无法在 Prompt 模板中准确找到 '{INPUT_JSON_MARKER}' 进行 JSON 插入。")
            prompt = self.prompt_template # Fallback

        return prompt

    @retry(
        stop=stop_after_attempt(5), # 最多重试 5 次
        wait=wait_fixed(1), # 每次重试等待 1 秒
        # retry_error_callback=log_retry_error # 移除 ZhipuAI 特定的重试日志函数
    )
    # 修改方法名称并更新日志信息
    def _call_llm_api(self, prompt: str) -> Optional[Dict[str, Any]]:
        """调用 LLM API 并返回解析后的 JSON 响应。"""
        # 调用 LLM API 并返回解析后的 JSON 响应
        logger.info("--- 尝试调用 LLM API ---")
        logger.debug(f"发送请求到模型: {self.model_name}")

        try:
            # 调用 OpenAI API (兼容 Gemini)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature, # 使用 settings 中的温度参数
                timeout=self.request_timeout,# 使用 settings 中的超时参数
                # max_tokens=settings.LLM_MAX_TOKENS, # 使用 settings 中的最大 token 数 (如果定义了)
                # top_p=settings.LLM_TOP_P, # 使用 settings 中的 top_p (如果定义了)
                # presence_penalty=settings.LLM_PRESENCE_PENALTY, # 使用 settings 中的 presence_penalty (如果定义了)
                # frequency_penalty=settings.LLM_FREQUENCY_PENALTY, # 使用 settings 中的 frequency_penalty (如果定义了)
                # 添加 Gemini 特有的参数 (如果适用且在 settings 中配置)
                # 例如:
                # candidate_count=settings.GEMINI_CANDIDATE_COUNT,
                # stop_sequences=settings.GEMINI_STOP_SEQUENCES,
                # reasoning_effort='high', # 移除不支持的参数
            )

            # 检查响应是否有效
            if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
                logger.error("LLM API response is missing expected content.")
                return None

            response_content = response.choices[0].message.content
            logger.info("--- LLM API 响应接收成功 ---")
            logger.debug(f"Raw Response Content:\n{response_content[:500]}...") # 截断日志

            # 从响应中提取 JSON 字符串
            json_string = extract_json_from_response(response_content)

            if not json_string:
                 # 如果无法提取 JSON 字符串，记录错误并返回 None
                 logger.error("无法从 API 响应中提取 JSON 字符串。")
                 # 记录完整的原始响应，如果不太长的话
                 if len(response_content) < 2000: # 限制日志长度
                     logger.error(f"原始响应内容:\n{response_content}")
                 else:
                     logger.error(f"原始响应内容 (部分):\n{response_content[:1000]}...")
                 return None

            logger.debug(f"提取的 JSON 字符串:\n{json_string}")
            try:
                # 解析 JSON 字符串
                parsed_json = json.loads(json_string)
                logger.info("成功解析 API 返回的 JSON。")
                return parsed_json
            except json.JSONDecodeError as json_err:
                # 如果解析 JSON 失败，记录错误并返回 None
                logger.error(f"无法解析 LLM API 返回的 JSON (提取后): {json_err}")
                logger.error(f"JSON 解析错误发生在第 {json_err.lineno} 行, 第 {json_err.colno} 列 (char {json_err.pos})")
                # 记录导致解析错误的 JSON 字符串
                logger.error(f"尝试解析的内容:\n{json_string}")
                return None

        except Exception as e:
            # 如果处理 API 响应时发生意外错误，记录错误并重新引发异常以供 tenacity 处理
            logger.error(f"处理 LLM API 响应时发生意外错误 (可能在重试中): {e}", exc_info=True)
            raise # Re-raise for tenacity


    def standardize(self, extracted_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        对提取的（人工核对后的）完整数据进行标准化，按设备组分批处理。

        Args:
            extracted_data: 包含设备组列表和备注的字典，格式来自 InfoExtractor 输出。
                            例如: {'设备列表': [...], '备注': {...}}

        Returns:
            Optional[Dict[str, Any]]: 包含标准化后设备组列表和原始备注的字典。
                                      例如: {'设备列表': [...], 备注: {...}}
                                      其中每个设备组包含 "位号", "标准化共用参数", "标准化不同参数"。
                                      如果处理失败则返回 None。
        """
        logger.info("--- 开始对提取的完整数据进行分批标准化 ---")

        if '设备列表' not in extracted_data or not isinstance(extracted_data['设备列表'], list):
            logger.error("输入数据中缺少'设备列表'或格式不正确，无法执行标准化。")
            return None

        original_device_groups = extracted_data['设备列表']
        standardized_device_groups = []
        remarks = extracted_data.get('备注') # 保留原始备注信息

        if not original_device_groups:
            logger.warning("没有设备组可供标准化。")
            return {"设备列表": [], "备注": remarks} # 如果没有设备组，直接返回空结果

        # 检查是否成功加载了完整语义表
        if self.full_semantic_table_content == "无法加载标准参数表。":
            logger.critical("完整语义表未成功加载，无法进行参数标准化。请检查文件路径和读取权限。")
            return None # 如果标准表未加载，则无法进行准确标准化

        logger.info(f"准备分批标准化 {len(original_device_groups)} 个设备组...")

        for idx, device_group in enumerate(original_device_groups):
            if not isinstance(device_group, dict):
                logger.warning(f"设备组项 {idx+1} 不是字典格式，跳过。内容: {device_group}")
                # 将格式不正确的组添加到结果中，并添加处理说明
                failed_group_entry = device_group # 直接使用原始数据，因为它不是字典
                if not isinstance(failed_group_entry, dict): # 确保可以添加键
                     failed_group_entry = {"原始数据": failed_group_entry} # 如果不是字典，包装一下
                failed_group_entry["处理说明"] = failed_group_entry.get("处理说明", "") + "格式不正确，跳过标准化。"
                standardized_device_groups.append(failed_group_entry)
                continue

            group_tags = device_group.get("位号", [])
            common_params = device_group.get("共用参数", {})
            diff_params = device_group.get("不同参数", {})

            if not group_tags:
                logger.warning(f"设备组 {idx+1} 缺少 '位号'，跳过标准化。")
                # 即使跳过标准化，也保留原始组结构
                failed_group_entry = device_group.copy()
                failed_group_entry["处理说明"] = failed_group_entry.get("处理说明", "") + "缺少位号，跳过标准化。"
                standardized_device_groups.append(failed_group_entry)
                continue

            logger.info(f"--- ({idx+1}/{len(original_device_groups)}) 处理设备组: {', '.join(group_tags)} ---")

            # 构建当前设备组的输入数据结构 (用于发送给 LLM 的小批量数据)
            current_group_data_for_llm = {
                "设备列表": [device_group],
                "备注": remarks # 将原始备注信息也包含在每个批次的输入中，提供完整上下文
            }

            # 1. 获取向量建议 (只为当前设备组的参数) - 此步骤已移除
            # combined_suggestions_text = self._fetch_suggestions_for_group_data(current_group_data_for_llm)
            # logger.debug(f"组合后的向量建议:\n{combined_suggestions_text}")

            # 2. 构建 Prompt (只包含当前设备组的数据和完整语义表内容)
            prompt = self._construct_llm_prompt_for_group_data(current_group_data_for_llm) # 不再传递 combined_suggestions_text
            logger.debug(f"构建的 Prompt (部分):\n{prompt[:500]}...")

            # 3. 调用 LLM API 处理当前设备组
            llm_response_json = None
            try:
                logger.info(f"设备组 {', '.join(group_tags)}: 等待10秒以避免LLM速率限制...")
                time.sleep(5) # Add delay before LLM call
                # 调用通用的 LLM API 方法
                llm_response_json = self._call_llm_api(prompt)
            except Exception as e: # Catch final error after tenacity retries
                # 更新日志信息
                logger.error(f"调用 LLM API 标准化设备组 {', '.join(group_tags)} 最终失败: {e}", exc_info=True)
                # 标准化失败，保留原始组数据并添加处理说明
                failed_group_entry = device_group.copy()
                failed_group_entry["处理说明"] = failed_group_entry.get("处理说明", "") + "标准化失败。"
                standardized_device_groups.append(failed_group_entry)
                continue # 继续处理下一个设备组

            # 4. 处理 LLM 响应并收集结果
            if llm_response_json and isinstance(llm_response_json, dict):
                # 期望 LLM 返回的 JSON 结构是 {"设备列表": [标准化后的当前组], "备注": ...}
                standardized_group_list_from_llm = llm_response_json.get("设备列表", [])
                if standardized_group_list_from_llm and isinstance(standardized_group_list_from_llm, list) and len(standardized_group_list_from_llm) > 0:
                    # 取第一个设备组的结果 (LLM应该只处理一个组)
                    standardized_group_entry = standardized_group_list_from_llm[0]
                    if isinstance(standardized_group_entry, dict):
                        # 检查是否包含标准化参数键，例如 "标准化共用参数", "标准化不同参数"
                        if "标准化共用参数" in standardized_group_entry or "标准化不同参数" in standardized_group_entry:
                             logger.info(f"成功从 LLM 获取设备组 {', '.join(group_tags)} 的标准化结果。")
                             standardized_device_groups.append(standardized_group_entry)
                        else:
                             logger.warning(f"LLM 返回的设备组 {', '.join(group_tags)} 结果未包含预期的标准化参数键。返回内容: {standardized_group_entry}")
                             # 未包含标准化键，保留原始组数据并添加处理说明
                             failed_group_entry = device_group.copy()
                             failed_group_entry["处理说明"] = failed_group_entry.get("处理说明", "") + "标准化结果格式异常。"
                             standardized_device_groups.append(failed_group_entry)
                    else:
                         logger.error(f"LLM 返回的设备组列表第一项格式不正确: {standardized_group_entry}")
                         # 格式不正确，保留原始组数据并添加处理说明
                         failed_group_entry = device_group.copy()
                         failed_group_entry["处理说明"] = failed_group_entry.get("处理说明", "") + "标准化结果格式错误。"
                         standardized_device_groups.append(failed_group_entry)
                else:
                    logger.error(f"LLM 响应中设备组 {', '.join(group_tags)} 未返回有效的 '设备列表' 或列表为空。响应: {llm_response_json}")
                    # 未找到有效设备列表，保留原始组数据并添加处理说明
                    failed_group_entry = device_group.copy()
                    failed_group_entry["处理说明"] = failed_group_entry.get("处理说明", "") + "标准化结果列表为空或格式错误。"
                    standardized_device_groups.append(failed_group_entry)
            else:
                logger.error(f"未能从 LLM 获取设备组 {', '.join(group_tags)} 的有效标准化结果或解析失败。")
                # 未能获取有效结果，保留原始组数据并添加处理说明
                failed_group_entry = device_group.copy()
                failed_group_entry["处理说明"] = failed_group_entry.get("处理说明", "") + "未能获取有效标准化结果。"
                standardized_device_groups.append(failed_group_entry)

            logger.info(f"--- 设备组 {', '.join(group_tags)} 处理完毕 ---")
            # 结束 for 循环

        logger.info("所有设备组标准化处理完成。")

        # 返回包含所有标准化设备组列表和原始备注的新字典
        return {"设备列表": standardized_device_groups, "备注": remarks}


# --- 可选的测试代码 ---

# 模拟依赖项 - MockSearchService 已不再需要，可以删除
# class MockSearchService:
#     def get_vector_suggestions(self, query_text, n_results):
#             print(f"MockSearchService: 获取 '{query_text}' 的向量建议 ({n_results}条)...")
#             # 返回假的建议结果列表
#             # 确保 settings 对象及其属性可用，否则这里会出错
#             try:
#                 param_type_field = settings.META_FIELD_PARAM_TYPE
#                 standard_value_field = settings.META_FIELD_STANDARD_VALUE
#             except AttributeError:
#                 logger.warning("MockSearchService: settings 中缺少 META_FIELD_PARAM_TYPE 或 META_FIELD_STANDARD_VALUE，使用默认字段名。")
#                 param_type_field = "param_type" # 默认值
#                 standard_value_field = "standard_value" # 默认值
#
#             return [
#                 {'metadata': {param_type_field: '标准参数A', standard_value_field: '标准值1'}, 'distance': 0.1},
#                 {'metadata': {param_type_field: '标准参数B', standard_value_field: '标准值2'}, 'distance': 0.2}
#             ]

# 修改模拟类以匹配 OpenAI 库的结构
class MockOpenAI:
    def __init__(self): # 添加构造函数
        self.chat = self.MockChat() # 实例化 MockChat 并赋值给 self.chat

    class MockChat:
        def __init__(self): # 添加构造函数
            self.completions = self.MockCompletions() # 实例化 MockCompletions

        class MockCompletions:
            def create(self, model, messages, temperature, timeout, reasoning_effort, **kwargs): # 添加 **kwargs 以处理其他可能的参数
                print(f"MockOpenAI: 接收到 Prompt (部分): {messages[0]['content'][:100]}...")
                # 返回一个假的、符合预期的 JSON 响应字符串
                # 模拟 LLM 返回标准化后的完整结构
                # 注意：这里模拟 LLM 只返回当前处理的设备组
                # 从 Prompt 中提取输入数据
                input_data_match = re.search(r'1\.  \*\*实际设备列表 \(JSON\):\*\*\s*(\{.*?\})\s*2\.  \*\*标准参数表 \(参考资料\):\*\*', messages[0]['content'], re.DOTALL)
                input_data = {}
                if input_data_match:
                        try:
                            input_data = json.loads(input_data_match.group(1))
                        except json.JSONDecodeError:
                            print("MockOpenAI: 无法解析 Prompt 中的输入 JSON。")
                            input_data = {"设备列表": []} # Fallback

                original_group = input_data.get('设备列表', [{}])[0] # 获取第一个设备组，如果列表为空则使用空字典
                original_tags = original_group.get('位号', [])
                original_common = original_group.get('共用参数', {})
                original_diff = original_group.get('不同参数', {})

                standardized_common = {f"标准_{k}": f"标准_{v}" for k, v in original_common.items()}
                standardized_diff = {}
                if isinstance(original_diff, dict):
                    for param_name, tag_value_map in original_diff.items():
                        if isinstance(tag_value_map, dict):
                            standardized_diff[f"标准_{param_name}"] = {
                                tag: f"标准_{v}"
                                for tag, v in tag_value_map.items()
                            }

                # 构建符合 JSON 规范的响应
                response_dict = {
                    "设备列表": [
                        {
                            "位号": original_tags,
                            "标准化共用参数": standardized_common,
                            "标准化不同参数": standardized_diff
                        }
                    ],
                    "备注": input_data.get("备注", {})
                }

                # 使用 json.dumps 确保输出正确的 JSON 格式
                response_content = (
                    f"标准化处理完成。\n```json\n"
                    f"{json.dumps(response_dict, ensure_ascii=False, indent=2)}\n"
                    f"```"
                )

                class MockMessage:
                    content = response_content
                class MockChoice:
                    message = MockMessage()
                class MockResponse:
                    def __init__(self, model_name): # 添加构造函数接收 model 名称
                        self.choices = [MockChoice()]
                        self.id = "mock-cmpl-xxxxxxxxxxxxxx"
                        self.model = model_name # 使用传入的 model 名称
                        self.object = "chat.completion"
                        self.created = int(Path(__file__).stat().st_mtime)
                return MockResponse(model_name=model) # 将 model 参数传递给构造函数


if __name__ == "__main__":
    # 配置基本日志
    logging.basicConfig(
        level=logging.DEBUG,  # 设置为 DEBUG 以查看详细日志，包括 Prompt 内容
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout # 将日志输出到标准输出
    )
    logger.info("========== 开始运行 accurate_llm_standardizer.py 主测试程序 ==========")

    # 模拟 settings (如果需要)
    class MockSettings:
        # 更新为通用的 LLM 配置
        LLM_API_KEY = "mock_api_key" # ZhipuAI/OpenAI API Key
        LLM_MODEL_NAME = "mock_llm_model" # 使用的模型名称
        LLM_API_URL = "mock_api_url" # API 地址 (如果需要覆盖 OpenAI 默认)
        LLM_TEMPERATURE = 0.3 # 使用 settings 中的温度
        LLM_REQUEST_TIMEOUT = 300 # 使用 settings 中的超时
        # LLM_MAX_TOKENS = 4096 # 示例
        # LLM_TOP_P = 0.9 # 示例
        # LLM_PRESENCE_PENALTY = 0.0 # 示例
        # LLM_FREQUENCY_PENALTY = 0.0 # 示例
        # META_FIELD_PARAM_TYPE = "std_name" # 不再需要，因为 MockSearchService 被移除
        # META_FIELD_STANDARD_VALUE = "std_value" # 不再需要
        # VECTOR_STORE_DIR = "data/vector_store" # 示例
        # DEFAULT_COLLECTION_NAME = "sensor_params" # 示例
    settings = MockSettings() # 覆盖导入的 settings

    # 1. 初始化服务 (使用模拟服务)
    # mock_search_service = MockSearchService() # 已移除
    # logger.info("模拟 SearchService 初始化成功。")

    # 使用 MockOpenAI 客户端
    mock_llm_client = MockOpenAI()
    logger.info("模拟 OpenAI 客户端初始化成功。")


    # 2. 初始化 AccurateLLMStandardizer
    try:
        standardizer = AccurateLLMStandardizer( # 不再传递 mock_search_service
            client=mock_llm_client
        )
        logger.info("AccurateLLMStandardizer 初始化成功 (使用模拟服务)。")
    except ValueError as e:
        logger.error(f"初始化 AccurateLLMStandardizer 失败: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"初始化 AccurateLLMStandardizer 时发生未知错误: {e}", exc_info=True)
        sys.exit(1)

    # 3. 准备示例输入数据
    sample_extracted_data = {
        "设备列表": [
            {
                "位号": [
                    "353-TE-35305"
                ],
                "共用参数": {
                    "仪表名称 Inst. Name": "热电阻",
                    "类型 Type": "单支",
                    "温度元件型号 Therm. Element Model": "缺失（文档未提供）",
                    "分度号 Type": "IEC标准 Pt100",
                    "允差等级 Tolerance Error Rating": "A级",
                    "测量端形式 Meas. End Type": "绝缘型",
                    "铠装材质 Armo. Mat'l": "316",
                    "铠装直径 Armo. Dia. (mm)": "Φ6",
                    "接线盒形式 Terminal Box Style": "防水型",
                    "接线盒材质 Terminal Box Mat'1": "304",
                    "电气连接 Elec. Conn.": "1/2\" NPT (F)",
                    "防护等级 Enclosure Protection": "IP65",
                    "防爆等级 Explosion Proof": "Exd II BT4",
                    "套管形式 Well Type": "整体钻孔锥形保护管",
                    "套管材质 Well Mat'l 压力等级 Pressure Rating": "316 Class150",
                    "套管外径 Well Outside Dia. (mm)": "根部不大于28,套管厚度由供货商根据振动频率和强度计算确定",
                    "过程连接形式 Process Conn. 连接规格Conn. Size": "固定法兰",
                    "连接规格Conn. Size": "DN40",
                    "法兰标准 Flange STD. 等级 Rating": "HG/T20615-2009",
                    "法兰材质 Flange Mat'l 密封面形式 Facing": "316 RF",
                    "制造厂 Manufacturer": "缺失（文档未提供）",
                    "备注": "缺失（文档未提供）",
                    "用途": "RLA-202 液体石蜡装车温度",
                    "管道设备号": "100-MO-1002-2B1A",
                    "介质 Fluid": "液体石蜡",
                    "操作/设计温度 Oper. Temp. (°C)": "40/",
                    "操作/设计压力 Oper. Press.": "0.3/",
                    "最大流速 Max. Velocity": "150",
                    "管嘴长度 Length": "250",
                    "插入深度 Well Length": "缺失（文档未提供）",
                    "测量范围 Meas. Range": "缺失（文档未提供）"
                }
            },
            { # 第二个设备组用于测试分批处理
                "位号": ["TEST003"],
                "共用参数": {
                    "实际共参A": "实际值A"
                },
                "不同参数": {}
            }
        ],
        "备注": { # 顶层备注
            "原始备注": "一些备注信息"
        }
    }
    logger.info(f"准备使用以下示例数据进行标准化:\n{json.dumps(sample_extracted_data, ensure_ascii=False, indent=2)}")

    # 4. 调用 standardize 方法
    logger.info("--- 开始调用 standardizer.standardize ---")
    standardized_result = standardizer.standardize(sample_extracted_data)
    logger.info("--- standardizer.standardize 调用结束 ---")

    # 5. 打印结果
    if standardized_result:
        logger.info("========== 标准化结果 ==========")
        # 使用 print 直接输出 JSON，避免日志格式干扰
        print(json.dumps(standardized_result, ensure_ascii=False, indent=4))
    else:
        logger.error("标准化处理失败，未返回有效结果。")

    logger.info("========== accurate_llm_standardizer.py 主测试程序运行结束 ==========")
