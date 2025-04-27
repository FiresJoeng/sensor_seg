import index
import model_matching
from difflib import SequenceMatcher
import csv


class CodeMatcher:
    """
    对输入参数和匹配字典进行模糊匹配，选出每个参数关联度最高的条目。
    """

    def __init__(self):
        pass

    @staticmethod
    def _fuzzy_score(a: str, b: str) -> float:
        """计算两个字符串之间的相似度分数（0~1）。"""
        return SequenceMatcher(None, a, b).ratio()

    def match_code(self, input_params: dict, matched_dict: dict) -> dict:
        """
        :param input_params: 原始输入参数，格式如 {'输出信号': '4~20mA DC', ...}
        :param matched_dict: 匹配后得到的候选列表，格式如 {'输出信号': [ {...}, {...} ], ...}
        :return: 新字典，key 对应参数名，value 为最佳匹配的那条字典
        """
        result = {}
        for key, input_val in input_params.items():
            candidates = matched_dict.get(key, [])
            if not candidates:
                # 无候选时可以跳过或赋值 None
                result[key] = None
                continue

            best_item = None
            best_score = -1.0

            # 对每个候选条目，尝试用 description、code、model 三个字段做匹配
            for cand in candidates:
                for field in ('description', 'code', 'model'):
                    text = str(cand.get(field, ''))
                    score = self._fuzzy_score(input_val, text)
                    if score > best_score:
                        best_score = score
                        best_item = cand

            result[key] = best_item
        return result


def sort_code(result: dict, refer_csv: str) -> dict:
    """
    根据参考CSV文件中model列的顺序重新排列结果字典
    
    :param result: match_code返回的结果字典
    :param refer_csv: 参考CSV文件路径
    :return: 重新排序后的结果字典
    """
    # 读取CSV文件，获取model列的顺序
    model_order = []
    try:
        with open(refer_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'model' in row and row['model'] not in model_order:
                    model_order.append(row['model'])
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return result  # 如果读取失败，返回原始结果
    
    # 创建一个新的有序字典
    sorted_result = {}
    
    # 首先按照CSV中model的顺序排列
    for model in model_order:
        for key, value in result.items():
            if value is not None and value.get('model') == model:
                sorted_result[key] = value
    
    # 将剩余未排序的项添加到结果中
    for key, value in result.items():
        if key not in sorted_result:
            sorted_result[key] = value
    
    return sorted_result


if __name__ == "__main__":
    # 示例输入参数
    input_params = {
        '输出信号': '4~20mA DC',
        '变 送器': 'YTA系列710',
        '说明书语言': '英语',
        '传感器输入': '双支输入',
        '壳体代码': '不锈钢',
        '接线口': '1/2NPT"（F）',
        '内置指示器': '液晶数显LCD',
        '安装支架': '2"管垂直安装',
        'NEPSI': 'GB3836.1-2010、GB3836.2-2010、GB12476.1-2013、GB12476.5-2013'
    }

    # 从索引加载模型并匹配标准参数
    main_model, csv_files = index.main(
        input_params, "libs/standard/transmitter/index.json"
    )
    matched_dict = model_matching.match_standard_params(
        input_params, csv_files
    )

    # 进行模糊匹配并获取最终结果
    matcher = CodeMatcher()
    result = matcher.match_code(input_params, matched_dict)
    
    # 使用csv_files列表的第一个值作为参考CSV进行排序
    refer_csv = csv_files[0] if csv_files else None
    if refer_csv:
        sorted_result = sort_code(result, refer_csv)
    else:
        sorted_result = result

    # 直接打印字典
    print(f"""
筛选后的结果:
{sorted_result}
""")

    # 打印匹配结果
    print("匹配代码结果:")
    for key, best in sorted_result.items():
        if best is None:
            print(f"{key} -> ?")
        else:
            print(f"{key} -> {best.get('code')}")