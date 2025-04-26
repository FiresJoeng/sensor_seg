# src/config.py
import os

# --- 基础路径设置 ---
# 获取项目根目录 (假设 config.py 在 src/ 下)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 输入数据配置 ---
# !! 重要 !!: 确认此路径指向您的 Excel 文件
EXCEL_FILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'semantic_source.xlsx')
# !! 重要 !!: 确认这些是您 Excel 文件中需要处理的工作表名称
SHEET_NAMES = [
    '变送器部分',
    '传感器部分',
    '保护管部分'
]

# --- 中间文件路径 ---
PREPROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'prepared_data.csv')
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, 'vector_store', 'embeddings.npy')
METADATA_PATH = os.path.join(PROJECT_ROOT, 'vector_store', 'metadata_for_index.csv') # 元数据文件

# --- 向量数据库配置 ---
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, 'vector_store', 'chroma_db_semantic_final')
COLLECTION_NAME = "semantic_parameters_v6" # 使用新名称以反映嵌入内容变化

# --- 模型配置 ---
EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

# --- 数据列名配置 (!! 请务必核对与您 Excel 文件中的列名完全一致 !!) ---
COL_COMPONENT_PART = '元器件部位'
COL_STANDARD_PARAM = '标准参数'
COL_ACTUAL_PARAM_DESC = '实际参数' # 将包含处理后的单一变体
COL_STANDARD_VALUE_DESC = '规格书代码的说明（多个值用|隔开）' # !! 核对 Excel 列名 !!
COL_ACTUAL_VALUE_VARIATIONS = '实际参数值（多个值用|隔开）一体化' # 将包含处理后的单一变体，用于嵌入 !! 核对 Excel 列名 !!
COL_DEFAULT_VALUE = '缺省默认值'
COL_STANDARD_CODE = '对应代码'
COL_FIELD_DESC = '字段说明'
COL_REMARK = '备注'

# 定义所有需要保留的原始列的列表 (确保包含所有 Excel 列)
ALL_ORIGINAL_COLS = [
    COL_COMPONENT_PART,
    COL_STANDARD_PARAM,
    COL_ACTUAL_PARAM_DESC,
    COL_STANDARD_VALUE_DESC,
    COL_ACTUAL_VALUE_VARIATIONS,
    COL_DEFAULT_VALUE,
    COL_STANDARD_CODE,
    COL_FIELD_DESC,
    COL_REMARK
]

# --- ChromaDB 元数据字段名 ---
# (这些是存储在向量数据库中的字段名，建议使用英文)
META_FIELD_COMPONENT = 'component_part'
META_FIELD_PARAM_TYPE = 'parameter_type' # 标准参数类型
META_FIELD_ACTUAL_PARAM_DESC = 'actual_param_desc' # 实际参数描述变体
META_FIELD_STANDARD_VALUE = 'standard_value'
META_FIELD_ACTUAL_VALUE = 'actual_value' # 实际参数值变体 (嵌入内容的一部分)
META_FIELD_DEFAULT = 'default_value' # 缺省默认值
META_FIELD_CODE = 'standard_code' # 对应代码
META_FIELD_FIELD_DESC = 'field_description' # 字段说明
META_FIELD_REMARK = 'remark' # 备注

# --- 查询参数 ---
DEFAULT_N_RESULTS = 10
INITIAL_QUERY_MULTIPLIER = 3

# --- 打印配置信息 (可选) ---
def print_config():
    """打印关键配置信息以供调试"""
    print("--- 项目配置信息 (Regenerated V4) ---")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Excel 文件路径: {EXCEL_FILE_PATH}")
    print(f"预处理数据路径: {PREPROCESSED_DATA_PATH}")
    print(f"嵌入向量路径: {EMBEDDINGS_PATH}")
    print(f"元数据路径 (CSV): {METADATA_PATH}")
    print(f"ChromaDB 路径: {CHROMA_DB_PATH}")
    print(f"ChromaDB 集合名称: {COLLECTION_NAME}")
    print(f"嵌入模型名称: {EMBEDDING_MODEL_NAME}")
    print("--- Excel 列名配置 (请务必核对!) ---")
    for col in ALL_ORIGINAL_COLS:
        print(f"  {col}")
    print("--- ChromaDB 元数据字段名 ---")
    print(f"  Component: {META_FIELD_COMPONENT}")
    print(f"  Param Type: {META_FIELD_PARAM_TYPE}")
    print(f"  Actual Param Desc: {META_FIELD_ACTUAL_PARAM_DESC}")
    print(f"  Standard Value: {META_FIELD_STANDARD_VALUE}")
    print(f"  Actual Value: {META_FIELD_ACTUAL_VALUE}")
    print(f"  Default Value: {META_FIELD_DEFAULT}")
    print(f"  Code: {META_FIELD_CODE}")
    print(f"  Field Desc: {META_FIELD_FIELD_DESC}")
    print(f"  Remark: {META_FIELD_REMARK}")
    print("-----------------------------")

if __name__ == '__main__':
    print_config()
