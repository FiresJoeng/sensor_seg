import json
import os
import re
from typing import Dict, Any, List, Tuple, Optional
from difflib import SequenceMatcher


# 字符串相似度计算函数
def str_similarity(a: str, b: str) -> float:
    """计算两个字符串的相似度"""
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0

    try:
        a_clean = a.replace(" ", "").lower()
        b_clean = b.replace(" ", "").lower()
        return SequenceMatcher(None, a_clean, b_clean).ratio()
    except Exception:
        return 0.0


# 向量匹配函数
def vector(target_key: str, target_value: str, input_params: Dict[str, str],
           available_models: List[str]) -> Tuple[str, str, float]:
    """查找最匹配的键值对组合"""
    if not input_params or not available_models:
        return None, None, 0.0

    key_mapping = {
        "变送器": ["温度变送器", "YTA"],
        "温度变送器": ["变送器", "YTA"],
        "YTA": ["温度变送器", "变送器"]
    }

    best_match_key = None
    best_match_model = None
    best_score = 0

    try:
        for input_key, input_value in input_params.items():
            if not input_key or not input_value:
                continue

            key_score = str_similarity(target_key, input_key)

            potential_matches = key_mapping.get(input_key, []) + [input_key]
            for potential in potential_matches:
                potential_key_score = str_similarity(target_key, potential)
                if potential_key_score > key_score:
                    key_score = potential_key_score

            for model in available_models:
                if not model:
                    continue

                model_score = str_similarity(input_value, model)
                combined_score = (key_score * 0.4) + (model_score * 0.6)

                if combined_score > best_score:
                    best_score = combined_score
                    best_match_key = input_key
                    best_match_model = model
    except Exception as e:
        print(f"向量匹配过程中发生错误: {str(e)}")
        return None, None, 0.0

    return best_match_key, best_match_model, best_score


# 主函数，支持"精确匹配"和"模糊匹配(阈值为0.85)"两种匹配方式
def main(input_params: Dict[str, Any], index_path: str, similarity_threshold: float = 0.85) -> Tuple[str, List[str]]:
    """确定产品主型号并定位关联的CSV数据源文件，支持精确匹配和模糊匹配"""
    if not input_params:
        raise ValueError("输入参数不能为空")

    if not index_path or not os.path.isfile(index_path):
        raise FileNotFoundError(f"找不到index.json文件: {index_path}")

    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到index.json文件: {index_path}")
    except json.JSONDecodeError:
        raise ValueError(f"index.json文件格式错误: {index_path}")
    except Exception as e:
        raise Exception(f"读取index.json时发生未知错误: {str(e)}")

    if not index_data:
        raise ValueError("index.json文件内容为空")

    model_value = None

    # 尝试精确匹配
    if '温度变送器' in input_params and input_params['温度变送器'] in index_data:
        model_value = input_params['温度变送器']
        print(f"精确匹配成功: 温度变送器 = {model_value}")
    else:
        # 尝试模糊匹配
        print("精确匹配失败，尝试模糊匹配...")

        try:
            available_models = list(index_data.keys())
            best_key, best_model, combined_score = vector(
                '温度变送器', '', input_params, available_models)

            if best_key is None or best_model is None or combined_score < similarity_threshold:
                raise ValueError(
                    f"无法找到匹配的键值对 (最佳匹配: 键={best_key}, 型号={best_model}, 得分: {combined_score:.2f})")

            print(
                f"找到最匹配的组合: 键={best_key}, 型号={best_model} (综合相似度: {combined_score:.2f})")
            model_value = best_model
        except Exception as e:
            raise ValueError(f"模糊匹配过程中发生错误: {str(e)}")

    if model_value not in index_data:
        raise ValueError(f"找不到匹配的型号: {model_value}")

    # 获取关联的CSV文件路径
    csv_files = index_data.get(model_value, [])
    if not csv_files:
        raise ValueError(f"型号 '{model_value}' 没有关联的CSV文件")

    valid_csv_files = []
    base_dir = os.path.dirname(index_path)

    # 检查对应CSV文件是否存在
    for csv_file in csv_files:
        try:
            csv_path = os.path.join(base_dir, csv_file)
            if os.path.exists(csv_path):
                valid_csv_files.append(csv_path)
            else:
                print(f"警告: CSV文件不存在: {csv_path}")
        except Exception as e:
            print(f"处理CSV文件路径时发生错误: {str(e)}")

    if not valid_csv_files:
        raise FileNotFoundError(f"找不到与型号 '{model_value}' 关联的任何CSV文件")

    return model_value, valid_csv_files


# 底层测试入口
if __name__ == "__main__":
    index_path = "libs/standard/transmitter/index.json"

    # 测试精确匹配
    input_params = {
        '温度变送器': 'YTA710',
        '输出信号': '4~20mA DC',
        '说明书语言': '英语',
        '传感器输入': '双支输入',
        '壳体代码': '不锈钢',
        '接线口': '1/2NPT"（F）',
        '内置指示器': '液晶数显LCD',
        '安装支架': '2"管垂直安装',
        'NEPSI': 'GB3836.1-2010、GB3836.2-2010、GB12476.1-2013、GB12476.5-2013'
    }

    # 测试模糊匹配 - 键不同
    input_params_fuzzy_key = {
        '变送器': 'YTA710',
        '输出信号': '4~20mA DC',
        '说明书语言': '英语'
    }

    # 测试模糊匹配 - 值有空格
    input_params_fuzzy_value = {
        '温度变送器': 'YTA 710',
        '输出信号': '4~20mA DC',
        '说明书语言': '英语'
    }

    # 测试模糊匹配 - 键和值都不同
    input_params_fuzzy_both = {
        '变送器': 'YTA 610',
        '输出信号': '4~20mA DC',
        '说明书语言': '英语'
    }

    # 测试更复杂的情况
    input_params_complex = {
        '温度 变送器': 'YTA系列50',
        '输出': '4-20mA',
        '语言': '英文'
    }

    test_cases = [
        ("精确匹配测试", input_params),
        ("模糊匹配测试 - 键不同", input_params_fuzzy_key),
        ("模糊匹配测试 - 值有空格", input_params_fuzzy_value),
        ("模糊匹配测试 - 键和值都不同", input_params_fuzzy_both),
        ("模糊匹配测试 - 复杂情况", input_params_complex)
    ]

    for test_name, test_params in test_cases:
        print(f"\n===== {test_name} =====")
        try:
            main_model, csv_files = main(test_params, index_path)
            print(f"主型号: {main_model}")
            print(f"关联的CSV文件:")
            for csv_file in csv_files:
                print(f"  - {csv_file}")
        except Exception as e:
            print(f"错误: {e}")
