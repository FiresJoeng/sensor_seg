# new_sensor_project/config/settings.py
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# --- Base Directory ---
# 项目根目录 (相对于此文件是 settings.py 的父级的父级)
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Environment Variables ---
# 加载 .env 文件 (应位于项目根目录 new_sensor_project/)
# 注意: .env 文件应包含 DEEPSEEK_API_KEY=your_actual_key
ENV_PATH = BASE_DIR / '.env'
load_dotenv(dotenv_path=ENV_PATH)

# --- API Configuration ---
# 从环境变量获取 API 密钥
# 明确尝试加载 OPENAI_API_KEY，如果不存在则回退到 LLM_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_API_KEY = OPENAI_API_KEY if OPENAI_API_KEY else os.getenv("LLM_API_KEY") # Use DeepSeek key if available, else fallback

# 如果未设置环境变量，发出警告
if not LLM_API_KEY:
    # 使用 print 因为 logging 可能尚未配置
    print("警告: 未在 .env 文件中找到 OPENAI_API_KEY 或 LLM_API_KEY。信息提取功能可能无法使用。") # Updated message
    # 或者可以设置一个默认的无效值，让后续代码处理
    # LLM_API_KEY = "YOUR_API_KEY_HERE"

# Load LLM Model and URL from environment variables
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "default-model-name") # Provide a default or handle if missing
LLM_API_URL = os.getenv("LLM_API_URL", "default-api-url") # Provide a default or handle if missing

# Add warnings if the environment variables are not set
if LLM_MODEL_NAME == "default-model-name":
    print("警告: 未在 .env 文件中找到 LLM_MODEL_NAME。将使用默认值 'default-model-name'。")
if LLM_API_URL == "default-api-url":
    print("警告: 未在 .env 文件中找到 LLM_API_URL。将使用默认值 'default-api-url'。")


LLM_TEMPERATURE = 0.3 # LLM 温度参数
LLM_REQUEST_TIMEOUT = 300 # LLM API 请求超时时间 (秒) - Kept increased timeout

# --- File Paths ---
# 数据目录
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"


# 知识库目录
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
KB_SOURCE_DIR = KNOWLEDGE_BASE_DIR / "source"
KB_VECTOR_STORE_DIR = KNOWLEDGE_BASE_DIR / "vector_store"
# 知识库源文件 (Excel) - 需要确认实际文件名
# 假设与原项目一致
KB_SOURCE_FILE = KB_SOURCE_DIR / "semantic_source.xlsx"
# ChromaDB 集合名称 (与原项目 semantic_parameter_project/src/config.py 对应)
VECTOR_DB_COLLECTION_NAME = "sensor_params_v4" # 或者根据需要更新

# 标准库目录 (需要将 sensor_seg/libs/standard 复制过来)
LIBS_DIR = BASE_DIR / "libs"
STANDARD_LIBS_DIR = LIBS_DIR / "standard"
# 标准库索引文件 (用于查找主型号对应的 CSV) - 假设结构与原项目一致
STANDARD_INDEX_JSON = STANDARD_LIBS_DIR / "transmitter" / "index.json" # 示例路径

# --- Logging Configuration ---
# 从环境变量读取日志级别，默认为 INFO
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
# 将字符串级别转换为 logging 模块的常量
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_TO_FILE = True # 是否将日志写入文件
LOG_FILE = OUTPUT_DIR / "pipeline.log" # 日志文件路径

# --- Semantic Search Configuration ---
# 嵌入模型名称 (与原项目 semantic_parameter_project/src/config.py 对应)
EMBEDDING_MODEL_NAME = 'shibing624/text2vec-base-chinese' # Switched to smaller model due to OOM error
# 向量数据库查询参数
DEFAULT_N_RESULTS = 5 # 默认返回结果数量
INITIAL_QUERY_MULTIPLIER = 10 # 向量数据库初始查询数量倍数 (用于后续重排)
# 元数据字段名 (与原项目 semantic_parameter_project/src/config.py 对应)
META_FIELD_COMPONENT = "component"
META_FIELD_PARAM_TYPE = "parameter_type"
META_FIELD_ACTUAL_PARAM_DESC = "actual_parameter_description"
META_FIELD_STANDARD_VALUE = "standard_value"
META_FIELD_ACTUAL_VALUE = "actual_value"
META_FIELD_DEFAULT = "default"
META_FIELD_CODE = "code"
META_FIELD_FIELD_DESC = "field_description"
META_FIELD_REMARK = "remark"
META_FIELD_SOURCE = "source_excel" # 假设添加了来源信息

# --- Knowledge Base Build Configuration ---
# 这些是知识库源 Excel 文件中的列名，用于数据处理和嵌入生成
# !! 重要 !!: 确保这些名称与你的 semantic_source.xlsx 文件中的列名完全一致
SOURCE_COL_COMPONENT_PART = '元器件部位'
SOURCE_COL_STANDARD_PARAM = '标准参数'
SOURCE_COL_ACTUAL_PARAM_DESC = '实际参数' # 用于嵌入
SOURCE_COL_STANDARD_VALUE_DESC = '规格书代码的说明（多个值用|隔开）'
SOURCE_COL_ACTUAL_VALUE_VARIATIONS = '实际参数值（多个值用|隔开）一体化' # 用于嵌入
SOURCE_COL_DEFAULT_VALUE = '缺省默认值'
SOURCE_COL_STANDARD_CODE = '对应代码'
SOURCE_COL_FIELD_DESC = '字段说明'
SOURCE_COL_REMARK = '备注'

# --- Standard Matching Configuration ---
# 模糊匹配阈值
FUZZY_MATCH_THRESHOLD = 0.7 # 用于 model_matching 和 code_matching
# 主型号匹配阈值 (来自 standard_matching/index.py)
MAIN_MODEL_SIMILARITY_THRESHOLD = 0.8

# --- Ensure Directories Exist ---
# 确保必要的输出和日志目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KB_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True) # 也创建输入目录

# --- Initial Print Statement (Optional) ---
# 使用 print 因为 logging 可能尚未配置
print(f"--- Configuration Loaded ---")
print(f"Project Base Directory: {BASE_DIR}")
print(f"Log Level: {LOG_LEVEL_STR} ({LOG_LEVEL})")
print(f"Log to file ({LOG_FILE}): {LOG_TO_FILE}")
print(f"Input Directory: {INPUT_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Vector Store Directory: {KB_VECTOR_STORE_DIR}")
print(f"Standard Libs Directory: {STANDARD_LIBS_DIR}")
print(f"LLM API Key Loaded: {'Yes' if LLM_API_KEY else 'No'}")
print(f"LLM Model Name: {LLM_MODEL_NAME}")
print(f"LLM API URL: {LLM_API_URL}")
print(f"--- End Configuration ---")
