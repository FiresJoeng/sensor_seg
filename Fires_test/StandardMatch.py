import pandas as pd
import json
import os
import random
import re

# 获取项目根目录的路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 设置标准库和语义库的路径
standard_lib_path = os.path.join(root_dir, "libs", "standard.xlsx")
semantic_lib_path = os.path.join(root_dir, "libs", "semantic.xlsx")


def load_standard_params(standard_para, method="rule_matching"):
    """
    从标准参数库提取参数

    参数:
        standard_para (str): 从语义库中获得的标准参数
        method (str): 使用的方法，可选 "rule_matching" 或 "llm_retrieval"

    返回:
        dict: 提取的标准参数
    """
    if method not in ["rule_matching", "llm_retrieval"]:
        raise ValueError("方法必须是 'rule_matching' 或 'llm_retrieval'")

    # 加载标准库
    try:
        standard_df = pd.read_excel(standard_lib_path, sheet_name=None)
    except Exception as e:
        print(f"加载标准库失败: {e}")
        return None

    if method == "rule_matching":
        return rule_matching(standard_para, standard_df)
    else:
        return llm_retrieval(standard_para, standard_df)


def rule_matching(standard_para, standard_df):
    """
    使用规则匹配从标准库中提取参数

    参数:
        standard_para (str): 标准参数
        standard_df (dict): 包含所有表格的DataFrame字典

    返回:
        dict: 匹配的参数
    """
    if not standard_para:
        print("未提供标准参数，无法进行匹配")
        return None

    result = {}

    # 遍历所有表格
    for sheet_name, df in standard_df.items():
        if 'B' not in df.columns:
            continue

        # 在B列中查找匹配项
        # 使用正则表达式进行更精确的匹配
        pattern = re.compile(
            r'\b' + re.escape(standard_para) + r'\b', re.IGNORECASE)

        matches = []
        for item in df['B'].dropna():
            if pattern.search(str(item)):
                matches.append(item)

        if matches:
            result[sheet_name] = matches

    return result


def llm_retrieval(standard_para, standard_df):
    """
    使用大模型检索从标准库中提取参数

    参数:
        standard_para (str): 标准参数
        standard_df (dict): 包含所有表格的DataFrame字典

    返回:
        dict: 检索的参数
    """
    try:
        # 准备数据
        data = {}
        for sheet_name, df in standard_df.items():
            if 'B' in df.columns:
                data[sheet_name] = df['B'].dropna().tolist()

        # 构建提示词
        prompt = f"""
        我有一个标准参数: {standard_para}
        请从以下数据中找出与该参数最相关的项:
        {json.dumps(data, ensure_ascii=False)}
        
        请以JSON格式返回结果，格式为:
        {{
            "sheet_name1": ["匹配项1", "匹配项2"],
            "sheet_name2": ["匹配项3"]
        }}
        """

        # 这里应该调用大模型API
        # 以下为示例代码，实际使用时需要替换为真实的API调用

        # 示例：使用DeepSeek API
        # from deepseek import DeepSeekAPI
        # api = DeepSeekAPI(api_key="your_api_key")
        # response = api.chat.completions.create(
        #     model="deepseek-chat",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # result = json.loads(response.choices[0].message.content)

        # 由于没有实际API调用，这里返回模拟结果
        print("注意: 这是模拟的大模型检索结果，实际使用时需要替换为真实API调用")
        result = {"模拟结果": ["这是大模型检索的模拟结果，请替换为实际API调用"]}

        return result

    except Exception as e:
        print(f"大模型检索失败: {e}")
        return None


def extract_model_component(param_text, category):
    """
    从参数文本中提取型号组件

    参数:
        param_text (str): 参数文本
        category (str): 参数类别（表格名）

    返回:
        str: 提取的型号组件
    """
    # 简化参数文本，提取关键词或首字母
    # 这里的逻辑可以根据实际需求调整

    # 示例：提取首个单词的前3个字符
    words = param_text.split()
    if words:
        # 取首个单词的前3个字符，转为大写
        return words[0][:3].upper()

    # 如果无法提取，则使用类别的首字母
    return category[0].upper()


def generate_model(params):
    """
    基于提取的参数生成型号

    参数:
        params (dict): 提取的参数

    返回:
        str: 生成的型号
    """
    if not params:
        print("没有可用的参数，无法生成型号")
        return None

    # 型号生成逻辑
    model_components = []

    # 从每个表格中提取关键信息
    for sheet_name, items in params.items():
        if items and len(items) > 0:
            # 提取有意义的特征作为型号组件
            # 这里的逻辑可以根据实际需求调整
            component = extract_model_component(items[0], sheet_name)
            if component:
                model_components.append(component)

    # 组合型号
    if model_components:
        # 添加前缀
        model_name = "SEN-" + "-".join(model_components)
        # 添加随机序列号
        model_name += f"-{random.randint(1000, 9999)}"
    else:
        model_name = f"SEN-GENERIC-{random.randint(1000, 9999)}"

    print(f"生成型号: {model_name}")
    print("注意: 此型号需要人工审核")

    return model_name


def human_review(model_name, standard_para, standard_params):
    """
    提供人工审核型号的接口

    参数:
        model_name (str): 生成的型号
        standard_para (str): 标准参数
        standard_params (dict): 匹配的参数

    返回:
        str: 审核后的型号
    """
    print(f"请人工审核型号: {model_name}")
    print("参考的标准参数:", standard_para)
    print("匹配的参数:", standard_params)

    # 在实际应用中，这里可以是一个用户界面，让用户进行审核
    # 示例中简单返回原型号
    return model_name


if __name__ == "__main__":
    # 示例标准参数
    standard_para = "温度传感器"

    # 使用规则匹配提取参数
    print("使用规则匹配:")
    rule_params = load_standard_params(standard_para, method="rule_matching")
    print("匹配结果:", rule_params)

    # 生成型号
    rule_model = generate_model(rule_params)
    print("生成型号:", rule_model)

    print("\n" + "-"*50 + "\n")

    # 使用大模型检索提取参数
    print("使用大模型检索:")
    llm_params = load_standard_params(standard_para, method="llm_retrieval")
    print("检索结果:", llm_params)

    # 生成型号
    llm_model = generate_model(llm_params)
    print("生成型号:", llm_model)

    # 人工审核
    final_model = human_review(llm_model, standard_para, llm_params)
    print("最终型号:", final_model)
