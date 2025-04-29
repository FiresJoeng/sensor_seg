# new_sensor_project/src/utils/llm_utils.py
import json
import logging
from openai import OpenAI, OpenAIError
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中以便导入 config
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import settings
except ImportError as e:
    print(f"ERROR in llm_utils.py: Failed to import settings - {e}. Check project structure and PYTHONPATH.", file=sys.stderr)
    raise

logger = logging.getLogger(__name__)

# 使用 config 中的设置初始化 OpenAI 客户端
# 初始化前检查 API 密钥是否已加载
if settings.LLM_API_KEY and settings.LLM_API_KEY != "YOUR_API_KEY_HERE":
    try:
        client = OpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_API_URL
        )
        logger.info("OpenAI 客户端初始化成功。")
    except Exception as e:
        logger.error(f"初始化 OpenAI 客户端失败: {e}", exc_info=True)
        client = None # 如果初始化失败，确保 client 为 None
else:
    logger.warning("在设置中未找到 LLM_API_KEY 或其为占位符。LLM 功能将被禁用。")
    client = None

def call_llm_for_match(system_prompt: str, user_prompt: str, expect_json: bool = True) -> dict | str | None:
    """
    根据提供的提示词调用配置好的 LLM。

    Args:
        system_prompt: 定义 LLM 角色和任务的系统提示词。
        user_prompt: 包含具体匹配数据的用户提示词。
        expect_json: 是否期望 LLM 返回 JSON 格式。如果为 False，则直接返回原始文本。

    Returns:
        Optional[Union[dict, str]]:
            - 如果 expect_json 为 True 且成功解析: 包含 LLM 响应的字典。
            - 如果 expect_json 为 False 且成功获取: LLM 返回的原始字符串。
            - 如果调用失败、被禁用或返回无效数据: 包含错误信息的字典或 None。
            - 注意：即使 expect_json 为 True，如果解析失败，也可能返回包含 'raw_content' 的错误字典。
    """
    if not client:
        logger.warning("LLM 客户端未初始化或缺少 API 密钥。跳过 LLM 调用。")
        return None

    logger.debug(f"调用 LLM。模型: {settings.LLM_MODEL_NAME}")
    logger.debug(f"系统提示词: {system_prompt[:100]}...") # 记录截断的提示词
    logger.debug(f"用户提示词: {user_prompt[:200]}...")   # 记录截断的提示词

    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=settings.LLM_TEMPERATURE,
            # max_tokens=8192, # 如果需要，可以考虑使 max_tokens 可配置
            timeout=settings.LLM_REQUEST_TIMEOUT
        )

        logger.debug(f"LLM API 响应: {response}")

        if response.choices:
            raw_content = response.choices[0].message.content
            logger.debug(f"LLM 原始响应内容: {raw_content}")

            # 清理可能的 markdown 代码块标记 (对两种情况都可能有用)
            cleaned_content = raw_content
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content.strip("```json").strip()
            elif cleaned_content.startswith("```"):
                 cleaned_content = cleaned_content.strip("```").strip()

            # 如果不期望 JSON，直接返回清理后的文本
            if not expect_json:
                logger.info("LLM 调用成功，返回原始文本响应。")
                return cleaned_content # 返回清理后的字符串

            # 如果期望 JSON，尝试解析
            try:
                result = json.loads(cleaned_content)
                logger.info("LLM 调用成功并解析了 JSON 响应。")
                return result # 返回解析后的字典
            except json.JSONDecodeError as json_err:
                logger.error(f"LLM 响应不是有效的 JSON (期望 JSON): {json_err}")
                logger.error(f"原始内容为: {raw_content}") # 记录原始未清理的内容
                # 即使期望 JSON 但解析失败，也返回包含原始内容的错误字典，以便上游处理
                return {"error": "LLM 响应格式错误", "details": str(json_err), "raw_content": raw_content}
            except Exception as parse_err:
                 logger.error(f"解析 LLM 响应时出错 (期望 JSON): {parse_err}", exc_info=True)
                 return {"error": "LLM 响应解析错误", "details": str(parse_err), "raw_content": raw_content}
        else:
            logger.warning("LLM 响应未包含任何选项。")
            # 根据是否期望 JSON 返回不同错误？目前统一返回字典错误
            return {"error": "LLM 未返回选项"}

    except OpenAIError as api_err:
        logger.error(f"LLM API 调用失败: {api_err}", exc_info=True)
        return {"error": "LLM API 错误", "details": str(api_err)}
    except Exception as e:
        logger.error(f"LLM 调用期间发生意外错误: {e}", exc_info=True)
        return {"error": "意外的 LLM 调用错误", "details": str(e)}

# 示例用法（用于测试目的）
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not client:
        print("LLM 客户端不可用。无法运行测试。")
    else:
        print("测试 LLM 调用...")
        # 示例提示词（在实际测试中替换为真实的提示词）
        test_system_prompt = "你是一个帮助匹配参数的助手。请以 JSON 格式响应。"
        test_user_prompt_json = "将输入参数 '颜色' 与以下选项之一匹配: ['红色', '蓝色', '绿色']。以 {'match': '选中的选项'} 的形式返回最佳匹配。"
        test_user_prompt_text = "用一句话描述红色。"

        print("\n--- 测试期望 JSON ---")
        match_result_json = call_llm_for_match(test_system_prompt, test_user_prompt_json, expect_json=True)
        if match_result_json:
            print("LLM JSON 结果:")
            # 检查返回的是否是字典，以防 API 错误返回错误字典
            if isinstance(match_result_json, dict):
                print(json.dumps(match_result_json, indent=2, ensure_ascii=False))
            else:
                print(f"收到非字典结果: {match_result_json}")
        else:
            print("LLM 调用失败或未返回结果。")

        print("\n--- 测试期望 Text ---")
        match_result_text = call_llm_for_match(test_system_prompt, test_user_prompt_text, expect_json=False)
        if match_result_text:
            print("LLM Text 结果:")
            # 检查返回的是否是字符串
            if isinstance(match_result_text, str):
                print(match_result_text)
            # 也可能返回错误字典
            elif isinstance(match_result_text, dict) and 'error' in match_result_text:
                 print(f"收到错误字典: {match_result_text}")
            else:
                 print(f"收到意外类型结果: {type(match_result_text)} - {match_result_text}")
        else:
            print("LLM 调用失败或未返回结果。")
