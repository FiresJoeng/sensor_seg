# 导入依赖
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# 环境变量
load_dotenv()
llm_api_key = os.getenv("LLM_API_KEY")
llm_api_url = os.getenv("LLM_API_URL")
llm_model_name = os.getenv("LLM_MODEL_NAME")


def func():
    try:

        # 构造系统提示
        system_prompt = f'''
    测试
    '''

        # LLM 客户端初始化
        client = OpenAI(
            api_key=llm_api_key,
            base_url=llm_api_url
        )

        # 调用API
        response = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "user", "content": f'收到测试消息请回复Hello World!'}
            ],
            temperature=0.1,
            max_tokens=8192
        )

        print(response)

        # 解析响应
        if response.choices:
            result = json.loads(response.choices[0].message.content)
            return result

        return {"错误": "未找到匹配的规格代码。"}

    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        return None
    except Exception as e:
        print(f"大模型检索失败: {e}")
        return None


if __name__ == "__main__":
    func()
