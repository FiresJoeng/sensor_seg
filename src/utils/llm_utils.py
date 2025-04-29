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

def call_llm_for_match(system_prompt: str, user_prompt: str) -> dict | None:
    """
    根据提供的提示词调用配置好的 LLM 来查找匹配项。

    Args:
        system_prompt: 定义 LLM 角色和任务的系统提示词。
        user_prompt: 包含具体匹配数据的用户提示词。

    Returns:
        Optional[dict]: 包含 LLM 响应（预期为 JSON）的字典，
                       如果调用失败、被禁用或返回无效数据，则返回 None。
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
            # 尝试将响应内容解析为 JSON
            try:
                # 清理可能的 markdown 代码块标记
                if raw_content.startswith("```json"):
                    raw_content = raw_content.strip("```json").strip()
                elif raw_content.startswith("```"):
                     raw_content = raw_content.strip("```").strip()

                result = json.loads(raw_content)
                logger.info("LLM 调用成功并解析了响应。")
                return result
            except json.JSONDecodeError as json_err:
                logger.error(f"LLM 响应不是有效的 JSON: {json_err}")
                logger.error(f"原始内容为: {raw_content}")
                return {"error": "LLM 响应格式错误", "details": str(json_err), "raw_content": raw_content}
            except Exception as parse_err:
                 logger.error(f"解析 LLM 响应时出错: {parse_err}", exc_info=True)
                 return {"error": "LLM 响应解析错误", "details": str(parse_err), "raw_content": raw_content}
        else:
            logger.warning("LLM 响应未包含任何选项。")
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
        test_user_prompt = "将输入参数 '颜色' 与以下选项之一匹配: ['红色', '蓝色', '绿色']。以 {'match': '选中的选项'} 的形式返回最佳匹配。"

        match_result = call_llm_for_match(test_system_prompt, test_user_prompt)

        if match_result:
            print("\nLLM 匹配结果:")
            print(json.dumps(match_result, indent=2, ensure_ascii=False))
        else:
            print("\nLLM 调用失败或未返回结果。")
