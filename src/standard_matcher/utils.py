# new_sensor_project/src/standard_matcher/utils.py
import csv
import logging
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Any, Optional

# 获取日志记录器实例
logger = logging.getLogger(__name__)

def calculate_string_similarity(a: Optional[str], b: Optional[str]) -> float:
    """
    计算两个字符串之间的相似度分数（0 到 1）。
    使用 SequenceMatcher 进行计算。处理 None 或非字符串输入。

    Args:
        a: 第一个字符串。
        b: 第二个字符串。

    Returns:
        float: 相似度得分，0.0 到 1.0 之间。
    """
    # 处理 None 或非字符串输入
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0

    # 移除空格并转为小写以进行不区分大小写和空格的比较
    a_clean = a.replace(" ", "").lower()
    b_clean = b.replace(" ", "").lower()

    try:
        # 计算相似度
        similarity = SequenceMatcher(None, a_clean, b_clean).ratio()
        # logger.debug(f"计算相似度: '{a}' vs '{b}' -> {similarity:.4f}")
        return similarity
    except Exception as e:
        # 记录计算过程中可能出现的任何异常
        logger.error(f"计算字符串相似度时出错 ('{a}' vs '{b}'): {e}", exc_info=True)
        return 0.0

def get_model_order_from_csv(reference_csv_path: Path) -> Optional[List[str]]:
    """
    从指定的 CSV 文件中读取 'model' 列的值，并返回其顺序列表。

    Args:
        reference_csv_path: 参考 CSV 文件路径。

    Returns:
        Optional[List[str]]: 包含 'model' 列值的顺序列表，如果文件不存在、
                             缺少 'model' 列或读取出错，则返回 None。
    """
    logger.debug(f"开始从 CSV 文件 '{reference_csv_path.name}' 读取 'model' 列顺序。")
    model_order: List[str] = []
    try:
        # 检查文件是否存在
        if not reference_csv_path.is_file():
            logger.error(f"参考 CSV 文件未找到: {reference_csv_path}")
            return None # 文件不存在

        # 读取 CSV 文件，获取 'model' 列的顺序
        with open(reference_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # 检查 'model' 列是否存在
            if 'model' not in reader.fieldnames:
                 logger.error(f"参考 CSV 文件 '{reference_csv_path.name}' 缺少 'model' 列。")
                 return None # 缺少列

            for row in reader:
                model_value = row.get('model')
                # 只添加非空且唯一的 model 值
                if model_value and model_value not in model_order:
                    model_order.append(model_value)
        logger.debug(f"从 CSV 获取到的 'model' 顺序: {model_order}")
        return model_order

    except Exception as e:
        logger.error(f"读取参考 CSV 文件 '{reference_csv_path.name}' 时出错: {e}", exc_info=True)
        return None # 读取失败

def sort_results_by_csv_order(result_dict: Dict[str, Optional[Dict[str, Any]]], reference_csv_path: Path) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    根据参考 CSV 文件中 'model' 列的顺序重新排列结果字典。
    确保原始字典中的所有键都存在于排序后的字典中。
    **注意**: 此函数现在依赖于 get_model_order_from_csv。
    确保原始字典中的所有键都存在于排序后的字典中。

    Args:
        result_dict: 待排序的字典，键是标准参数名，值是匹配到的最佳字典或 None。
                     例如: {'输出信号': {'model': '输出信号', 'code': 'A', ...}, ...}
        reference_csv_path: 用于确定顺序的参考 CSV 文件路径。

    Returns:
        Dict[str, Optional[Dict[str, Any]]]: 重新排序后的字典。如果读取 CSV 失败，返回原始字典。
    Args:
        result_dict: 待排序的字典，键是标准参数名，值是匹配到的最佳字典或 None。
                     例如: {'输出信号': {'model': '输出信号', 'code': 'A', ...}, ...}
        reference_csv_path: 用于确定顺序的参考 CSV 文件路径。

    Returns:
        Dict[str, Optional[Dict[str, Any]]]: 重新排序后的字典。如果读取 CSV 失败，返回原始字典。
    """
    logger.debug(f"开始根据 CSV 文件 '{reference_csv_path.name}' 对结果进行排序。")
    # 调用新函数获取 model 顺序
    model_order = get_model_order_from_csv(reference_csv_path)

    # 如果获取顺序失败，直接返回原始字典
    if model_order is None:
        logger.warning(f"无法从 '{reference_csv_path.name}' 获取 'model' 顺序，排序取消。")
        return result_dict

    # 创建一个新的有序字典
    sorted_result: Dict[str, Optional[Dict[str, Any]]] = {}
    processed_keys = set() # 跟踪已处理的键

    # 1. 首先按照 CSV 中 'model' 的顺序添加条目
    for model_name in model_order:
        # 遍历原始字典，查找 model 匹配的条目
        for key, value_dict in result_dict.items():
            # 确保 value_dict 不是 None 并且包含 'model' 键
            if value_dict and value_dict.get('model') == model_name:
                # 如果这个 key 尚未处理，则添加到排序后的字典中
                if key not in processed_keys:
                    sorted_result[key] = value_dict
                    processed_keys.add(key)
                    logger.debug(f"按 CSV 顺序添加键 '{key}' (model: {model_name})")

    # 2. 将原始字典中剩余的、未按 CSV 顺序处理的项添加到末尾
    # 这确保了即使某些键的 model 不在 CSV 的 model_order 中，它们也不会丢失
    remaining_items = 0
    for key, value_dict in result_dict.items():
        if key not in processed_keys:
            sorted_result[key] = value_dict
            remaining_items += 1
            logger.debug(f"添加剩余键 '{key}'")

    if remaining_items > 0:
         logger.debug(f"添加了 {remaining_items} 个不在 CSV model 顺序中的剩余键。")

    logger.debug("结果排序完成。")
    return sorted_result

# --- 可选的测试入口 ---
if __name__ == '__main__':
    # 配置基本日志记录以进行测试
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 测试字符串相似度
    print("\n--- 测试字符串相似度 ---")
    print(f"'M20×1.5' vs 'M20 x 1.5': {calculate_string_similarity('M20×1.5', 'M20 x 1.5'):.4f}")
    print(f"'不锈钢' vs 'Stainless Steel': {calculate_string_similarity('不锈钢', 'Stainless Steel'):.4f}")
    print(f"'4~20mA DC' vs '4-20mA': {calculate_string_similarity('4~20mA DC', '4-20mA'):.4f}")
    print(f"'None' vs 'Test': {calculate_string_similarity(None, 'Test'):.4f}")

    # 测试排序 (需要一个临时的 CSV 文件)
    print("\n--- 测试排序 ---")
    temp_dir = Path("./temp_test_sort")
    temp_dir.mkdir(exist_ok=True)
    temp_csv_path = temp_dir / "test_order.csv"
    with open(temp_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'code', 'description'])
        writer.writerow(['输出信号', 'A', 'Signal Output'])
        writer.writerow(['壳体代码', 'S', 'Housing Code'])
        writer.writerow(['传感器输入', 'U', 'Sensor Input'])
        writer.writerow(['接线口', '2', 'Wiring Port']) # 这个顺序将被用来排序

    test_result_dict = {
        '传感器输入': {'model': '传感器输入', 'code': 'U', 'description': '...'},
        '输出信号': {'model': '输出信号', 'code': 'A', 'description': '...'},
        '接线口': {'model': '接线口', 'code': '2', 'description': '...'},
        '壳体代码': {'model': '壳体代码', 'code': 'S', 'description': '...'},
        '未在CSV中的参数': {'model': '其他', 'code': 'X', 'description': '...'},
        '值为None的参数': None
    }

    print("原始字典顺序:", list(test_result_dict.keys()))
    sorted_dict = sort_results_by_csv_order(test_result_dict, temp_csv_path)
    print("排序后字典顺序:", list(sorted_dict.keys()))

    # 清理临时文件
    # import shutil
    # shutil.rmtree(temp_dir)
    print(f"(测试用的临时 CSV 文件位于: {temp_csv_path})")
