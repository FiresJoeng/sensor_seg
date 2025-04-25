'''
逻辑：
遍历不同的标准参数，
筛选规格代码，
再通过LLM模糊检索实际参数值以确认型号
'''


# 导入依赖
import pandas as pd
import json
import os
import random
import re
from dotenv import load_dotenv
from openai import OpenAI
from src.utils import xlsx2md

# 定义变量
standard_lib_path = "libs/standard.xlsx"


# 类: 规格代码筛选器
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
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            if not deepseek_api_key:
                raise ValueError("未找到DEEPSEEK_API_KEY环境变量")

            # 函数: 将Excel转换为Markdown格式的文本
            standard_lib = xlsx2md(standard_lib_path)

            # 构造系统提示
            system_prompt = f'''
            你是一个大模型检索器，你的任务是在标准库中，筛选与“标准参数”相关的所有“部件”，然后输出其对应的“规格代码”。“部件”在标准表的第一列，“规格代码”在标准表的第二列（规格代码）。
            “标准参数”会在接下来的对话中给出，“标准参数”不一定与“部件”逐字对应。因此检索相关“部件”时要考虑模糊性。若确实没有检索到与“标准参数”相关的“部件”则忽略。
            以下为标准库: ```{standard_lib}```
            输出格式为: ```{{"表格名称": ["规格代码"]}}```，其中列表里的“规格代码”数量是若干个，未检索到则为空列表。
            '''

            # LLM 客户端初始化
            client = OpenAI(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com/v1"
            )

            # 调用API
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f'请按照以上要求，检索与标准参数```{standard_params}```相关的所有规格代码，然后严格按照输出格式输出，不要输出其他内容。'}
                ],
                temperature=0.1,
                max_tokens=8192
            )

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

    # 规则匹配
    @staticmethod
    def rule_matching(standard_params, standard_df):
        """
        根据标准参数，使用规则匹配，从标准库中筛选对应的规格代码

        参数:
            standard_params (str): 标准参数
            standard_df (dict): 包含所有表格的DataFrame字典

        返回:
            dict: 匹配的规格代码
        """
        if not standard_params:
            raise ValueError("错误: 标准参数不能为空!")

        result = {}
        found_any_match = False

        # 正则表达式设置
        pattern = re.compile(
            re.escape(standard_params), re.IGNORECASE)

        # 遍历所有表格
        for sheet_name, df in standard_df.items():
            # 如果少于两列则跳过
            if df.shape[1] < 2:
                continue

            matches = []

            # 遍历行
            for idx, row in df.iterrows():
                # 遍历除第二列之外的其他列
                for j, col in enumerate(df.columns):
                    if j == 1:
                        continue
                    cell = row[col]
                    if isinstance(cell, str) and pattern.search(cell):
                        # 匹配成功，取第二列的值
                        val = row.iloc[1]
                        if isinstance(val, str) and val.strip():
                            matches.append(val)
                        break

            # 判断匹配结果
            if matches:
                result[sheet_name] = matches
                found_any_match = True

        if not found_any_match:
            selector = input(
                f'未找到标准参数"{standard_params}"，是否使用大模型检索? (Y/N) : ')
            if selector.strip().upper() == 'Y':
                return Fliters.llm_retrieval(standard_params, standard_df)

        return result

    @staticmethod
    def load_filters(standard_params, method="rule_matching"):
        """
        规格代码筛选器，接受标准参数和筛选方法作为输入，返回提取的规格代码

        参数:
            standard_params (str): 从语义库中获得的标准参数
            method (str): 使用的方法，可选 "rule_matching" 或 "llm_retrieval"

        返回:
            dict: 提取的规格代码
        """
        if method not in ["rule_matching", "llm_retrieval"]:
            raise ValueError("方法必须是 'rule_matching' 或 'llm_retrieval'")

        # 加载标准库
        try:
            standard_df = pd.read_excel(standard_lib_path, sheet_name=None)
        except Exception as e:
            print(f"加载标准库失败: {e}")
            exit(1)

        if method == "rule_matching":
            return Fliters.rule_matching(standard_params, standard_df)
        else:
            return Fliters.llm_retrieval(standard_params, standard_df)


# 类: 型号生成器
class Generator:
    @staticmethod
    def temp_func():
        pass


# 测试或示例执行
if __name__ == "__main__":
    # 示例标准参数
    standard_params = "TG套管形式"

    # 使用规则匹配提取规格代码
    print("使用规则匹配:")
    model_list = Fliters.load_filters(
        standard_params, "rule_matching")
    print("匹配结果:", model_list)
