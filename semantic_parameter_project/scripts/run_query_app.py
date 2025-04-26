# scripts/run_query_app.py
import sys
import os
import pandas as pd # 导入 pandas 用于创建和打印表格
import time # 导入 time 用于计时
from typing import List, Dict, Any

# --- 动态添加 src 目录到 Python 路径 ---
scripts_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(scripts_dir)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- 导入必要的模块 ---
try:
    from search_service import SearchService # 使用 V3 Logic 或更新版本
    import config # 导入 V3 配置
    from utils import sanitize_input, logger
except ImportError as e:
    print(f"错误: 无法导入 src 目录下的模块。请确保项目结构正确且 src 在 Python 路径中。")
    print(f"详细错误: {e}")
    sys.exit(1)

def display_results(results: List[Dict[str, Any]]):
    """
    格式化并以表格形式打印搜索结果。
    ** V3 Logic: 按照原始 Excel 列顺序显示所有信息。**

    Args:
        results (List[Dict[str, Any]]): 包含搜索结果字典的列表。
                                        字典的键应该是 config.py 中定义的 META_FIELD_ 常量。
    """
    if not results:
        print("  未找到相关结果。")
        return

    print("\n--- 查询结果 ---")

    # 定义表头顺序 (与 config.ALL_ORIGINAL_COLS 一致，加上排名和距离)
    # 注意：这里的 headers 对应的是最终显示的列名，应与原始 Excel 一致
    display_headers = ["排名", "距离"] + config.ALL_ORIGINAL_COLS

    # 准备表格数据
    table_data = []
    for result in results:
        # 从 result 字典中获取元数据字段的值，使用 config 中定义的 META_FIELD_ 常量作为 key
        # 并按照 ALL_ORIGINAL_COLS 的顺序排列
        row_data = [
            result.get('rank', 'N/A'),
            result.get('distance', 'N/A'),
            result.get(config.META_FIELD_COMPONENT, '无'),
            result.get(config.META_FIELD_PARAM_TYPE, '无'),
            result.get(config.META_FIELD_ACTUAL_PARAM_DESC, '无'),
            result.get(config.META_FIELD_STANDARD_VALUE, '无'),
            result.get(config.META_FIELD_ACTUAL_VALUE, '无'),
            result.get(config.META_FIELD_DEFAULT, '无'),
            result.get(config.META_FIELD_CODE, '无'),
            result.get(config.META_FIELD_FIELD_DESC, '无'),
            result.get(config.META_FIELD_REMARK, '无')
        ]
        table_data.append(row_data)

    # 使用 pandas 创建 DataFrame 以便格式化打印
    try:
        df_results = pd.DataFrame(table_data, columns=display_headers)
        # 使用 to_markdown 打印表格，需要安装 tabulate: pip install tabulate
        print(df_results.to_markdown(index=False))
        print("\n注意: '实际参数' 和 '实际参数值...' 列显示的是向量数据库中匹配到的具体写法。")
    except ImportError:
        print(" (推荐安装 pandas 和 tabulate 以获得更好的表格输出: pip install pandas tabulate) ")
        print("\t".join(display_headers)) # 打印表头
        for row_data in table_data:
            # 将所有元素转为字符串再 join
            print("\t".join(map(str, row_data)))
        print("\n注意: '实际参数' 和 '实际参数值...' 列显示的是向量数据库中匹配到的具体写法。")
    except Exception as e:
        print(f"打印结果时出错: {e}")
        # 备用打印方式
        print("--- 备用打印格式 ---")
        for i, result in enumerate(results):
             print(f"\n--- 结果 {i+1} ---")
             print(f"  排名: {result.get('rank', 'N/A')}")
             print(f"  距离: {result.get('distance', 'N/A')}")
             # 打印所有元数据字段
             for col_name, meta_key in zip(config.ALL_ORIGINAL_COLS, [
                 config.META_FIELD_COMPONENT, config.META_FIELD_PARAM_TYPE,
                 config.META_FIELD_ACTUAL_PARAM_DESC, config.META_FIELD_STANDARD_VALUE,
                 config.META_FIELD_ACTUAL_VALUE, config.META_FIELD_DEFAULT,
                 config.META_FIELD_CODE, config.META_FIELD_FIELD_DESC,
                 config.META_FIELD_REMARK
             ]):
                 print(f"  {col_name}: {result.get(meta_key, '无')}")


def main():
    """主函数，运行交互式查询应用。"""
    print("正在初始化语义参数查询系统...")
    logger.info("启动查询应用...")
    search_service = SearchService() # 使用 V3 Logic 或更新版本

    if not search_service.is_ready():
        print("\n错误：查询服务未能成功初始化，无法启动。")
        logger.error("查询服务初始化失败，应用退出。")
        sys.exit(1)

    print("\n======================================")
    print("=== 欢迎使用语义参数查询系统 ===")
    print("======================================")
    print("请输入参数类型和参数值进行查询。")
    print("例如:")
    print("  类型: 输出信号, 值: 4-20mA HART")
    print("输入 'q' 或按 Ctrl+C 退出。")

    while True:
        try:
            param_type_input_raw = input("\n请输入 参数类型 (例如: 输出信号): ")
            param_type_input = sanitize_input(param_type_input_raw)
            if param_type_input.lower() == 'q': break
            if not param_type_input: print("错误: 参数类型不能为空。"); continue

            param_value_input_raw = input(f"请输入 '{param_type_input}' 对应的 参数值: ")
            param_value_input = sanitize_input(param_value_input_raw)
            if not param_value_input: print("错误: 参数值不能为空。"); continue

            logger.info(f"收到查询 - 类型: '{param_type_input}', 值: '{param_value_input}'")
            print(f"正在搜索 类型='{param_type_input}', 值='{param_value_input}'...")
            query_start_time = time.time()

            search_results = search_service.search(
                parameter_type=param_type_input,
                parameter_value=param_value_input,
                n_results=config.DEFAULT_N_RESULTS
            )
个
            query_end_time = time.time()
            logger.info(f"查询耗时: {query_end_time - query_start_time:.4f} 秒")

            if search_results is not None:
                 display_results(search_results) # 使用 V3 display_results
            else:
                 print("  查询过程中遇到问题，未能获取结果。")
                 logger.warning(f"查询失败 - 类型: '{param_type_input}', 值: '{param_value_input}'")

        except (EOFError, KeyboardInterrupt): print("\n检测到退出信号。"); logger.info("用户请求退出应用。"); break
        except Exception as e: print(f"\n发生意外错误: {e}"); logger.exception(f"查询循环中发生意外错误: {e}")

    print("\n======================================")
    print("=== 查询结束，感谢使用！ ===")
    print("======================================")
    logger.info("查询应用正常关闭。")

if __name__ == "__main__":
    main()
