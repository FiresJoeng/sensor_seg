# 导入依赖
import pandas as pd
import json
import os
import random
import re

# 定义变量
standard_lib_path = "libs/standard.xlsx"


# 类: 参数筛选器
class ParamsFilters:
    # 大模型检索
    @staticmethod
    def llm_retrieval(standard_params, standard_df):
        """
        使用大模型检索从标准库中提取参数

        参数:
            standard_params (str): 标准参数
            standard_df (dict): 包含所有表格的DataFrame字典

        返回:
            dict: 检索的参数
        """
        try:
            pass
        except Exception as e:
            print(f"大模型检索失败: {e}")
            return None

    # 规则匹配
    @staticmethod
    def rule_matching(standard_params, standard_df):
        """
        使用规则匹配从标准库中提取参数

        参数:
            standard_params (str): 标准参数
            standard_df (dict): 包含所有表格的DataFrame字典

        返回:
            dict: 匹配的参数
        """
        if not standard_params:
            raise ValueError("错误: 标准参数不能为空!")

        result = {}

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
            else:
                selector = input(
                    f'在"{sheet_name}"中未找到"{standard_params}"，是否使用大模型检索? (Y/N) : ')
                if selector.strip().upper() == 'Y':
                    return ParamsFilters.llm_retrieval(standard_params, standard_df)
                else:
                    result[sheet_name] = matches
        return result

    @staticmethod
    def load_filters(standard_params, method="rule_matching"):
        """
        从标准参数库提取参数

        参数:
            standard_params (str): 从语义库中获得的标准参数
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
            exit(1)

        if method == "rule_matching":
            return ParamsFilters.rule_matching(standard_params, standard_df)
        else:
            return ParamsFilters.llm_retrieval(standard_params, standard_df)


# 类: 型号生成器
class Generator:
    @staticmethod
    def temp_func():
        pass


# 测试或示例执行
if __name__ == "__main__":
    # 示例标准参数
    standard_params = "输出信号"

    # 使用规则匹配提取参数
    print("使用规则匹配:")
    rule_params = ParamsFilters.load_filters(
        standard_params, method="rule_matching")
    print("匹配结果:", rule_params)
