'''
逻辑：
遍历不同的标准参数，
筛选规格代码，
再通过LLM模糊检索实际参数值以确认型号
'''


# 导入依赖
import json
import os
from dotenv import load_dotenv
from openai import OpenAI


class Fliters:
    # 大模型检索
    @staticmethod
    def llm_retrieval(standard_params, standard_df):
        """
        根据标准参数，使用大模型检索，从标准库中筛选对应的规格代码

        参数:
            standard_params (str): 标准参数
            standard_df (dict): 包含所有表格的DataFrame字典

        返回:
            dict: 检索出的规格代码
        """
        try:
            # 环境变量
            load_dotenv()
            llm_api_key = os.getenv("LLM_API_KEY")
            llm_api_url = os.getenv("LLM_API_URL")
            llm_model_name = os.getenv("LLM_MODEL_NAME")

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
                    {"role": "system", "content": system_prompt},
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
    # 测试
    Fliters.llm_retrieval('','')