# check_types.py
import sys
import os

# --- 动态添加 src 目录到 Python 路径 ---
scripts_dir = os.path.dirname(os.path.abspath(__file__))
# 假设脚本放在项目根目录，如果放在 scripts/，需要调整 project_root
project_root = scripts_dir
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- 导入必要的模块 ---
try:
    from vector_store_manager import VectorStoreManager
    import config # 导入配置以获取元数据字段名
except ImportError as e:
    print(f"错误: 无法导入 src 目录下的模块。")
    print(f"详细错误: {e}")
    sys.exit(1)

def main():
    print("正在连接数据库以检查 'parameter_type' 的唯一值...")
    manager = VectorStoreManager()
    if not manager.connect():
        print("数据库连接失败。")
        return

    # 获取 'parameter_type' 字段的所有唯一值
    # 使用 config.META_FIELD_PARAM_TYPE 获取正确的字段名
    unique_param_types = manager.get_all_metadata_values(config.META_FIELD_PARAM_TYPE)

    if unique_param_types:
        print("\n数据库中找到的唯一 'parameter_type' 值:")
        for p_type in unique_param_types:
            # 使用 repr() 来明确显示字符串，包括可能的首尾空格
            print(f"- {repr(p_type)}")
        print(f"\n总共找到 {len(unique_param_types)} 个唯一类型。")
        print("请在查询时使用上面列表中的确切字符串作为'参数类型'进行过滤。")
    else:
        print("\n未能从数据库中获取到任何 'parameter_type' 的唯一值。")
        print("这可能意味着：")
        print("  1. 知识库尚未构建或为空。")
        print(f"  2. 在构建索引时，未能正确存储 '{config.META_FIELD_PARAM_TYPE}' 元数据。")
        print(f"  3. config.py 中的 META_FIELD_PARAM_TYPE ('{config.META_FIELD_PARAM_TYPE}') 配置错误。")

if __name__ == "__main__":
    main()
