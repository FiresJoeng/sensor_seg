import pandas as pd
import index
from typing import Dict, List, Any
from difflib import SequenceMatcher


def calculate_similarity(a: str, b: str) -> float:
    """
    计算两个字符串之间的语义相似度。
    使用SequenceMatcher计算相似度，返回0到1之间的值。
    """
    return SequenceMatcher(None, a, b).ratio()


class StandardParamsMatcher:
    """
    标准参数匹配器：根据优先级（精确、模糊、LLM）在CSV数据中匹配标准参数（model列）。
    """

    def __init__(self, df: pd.DataFrame, input_params: Dict[str, Any]):
        self.df = df
        self.input_params = input_params

    def exact_match(self, param_name: str) -> pd.DataFrame:
        """直接按model列精确匹配。"""
        return self.df[self.df['model'] == param_name]

    def fuzzy_match(self, param_name: str) -> pd.DataFrame:
        """使用语义相似度进行模糊匹配，相似度0.7以上即匹配成功。"""
        matches = []
        for model in self.df['model'].unique():
            similarity = calculate_similarity(param_name, model)
            if similarity >= 0.7:
                matches.append(model)

        if matches:
            return self.df[self.df['model'].isin(matches)]
        return pd.DataFrame()

    def lm_match(self, param_name: str) -> pd.DataFrame:
        """这里先不做！！！"""
        pass

    def match(self, param_name: str) -> pd.DataFrame:
        """按顺序(精确->模糊->LLM)执行匹配。"""
        result = self.exact_match(param_name)
        if not result.empty:
            return result
        result = self.fuzzy_match(param_name)
        if not result.empty:
            return result
        result = self.lm_match(param_name)
        if result is not None and not result.empty:
            return result
        raise ValueError(f"匹配失败: 参数 {param_name} 无法在标准参数中找到对应项。")


def match_standard_params(input_params: Dict[str, Any], csv_files: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    标准参数匹配阶段：读取并合并CSV，对每个输入参数执行匹配
    并校验是否有未匹配的model。
    返回一个字典，键为参数名，值为匹配到的记录列表。
    """
    dfs = []
    for path in csv_files:
        dfs.append(pd.read_csv(path))
    combined = pd.concat(dfs, ignore_index=True)

    matcher = StandardParamsMatcher(combined, input_params)
    matched_dfs: Dict[str, pd.DataFrame] = {}
    for param in input_params.keys():
        # 跳过主型号
        if param == input_params.get('温度变送器') or param.lower() == '变送器':
            continue
        matched_dfs[param] = matcher.match(param)

    all_models = set(combined['model'].unique())
    matched_models = set(matched_dfs.keys())

    # 将DataFrame转换为字典列表
    matched_dict = {}
    for key, df in matched_dfs.items():
        # 将DataFrame转换为字典列表
        records = df.to_dict(orient='records')
        matched_dict[key] = records

    print(f"""
匹配结果:
{matched_dict}
""")

    # 返回结果字典
    return matched_dict


if __name__ == '__main__':
    input_params = {
        '说明书语言': '英语',
        '变 送器': 'YTA系列710',
        '输出信号': '4~20mA DC',
        '传感器输入': '双支输入',
        '壳体代码': '不锈钢',
        '接线口': '1/2NPT"（F）',
        '内置指示器': '液晶数显LCD',
        '安装支架': '2"管垂直安装',
        'NEPSI': 'GB3836.1-2010、GB3836.2-2010、GB12476.1-2013、GB12476.5-2013'
    }
    main_model, csv_files = index.main(
        input_params, "libs/standard/transmitter/index.json")

    matched_dict = match_standard_params(input_params, csv_files)
