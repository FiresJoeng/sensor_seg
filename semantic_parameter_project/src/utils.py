# src/utils.py
import time
import logging
import functools # 用于装饰器保留原函数信息

# --- 配置日志记录器 ---
# 配置基本的日志记录，级别为 INFO，格式包含时间、级别和消息
# 可以根据需要将日志输出到文件，例如添加 FileHandler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
# 获取名为当前模块的 logger 实例
logger = logging.getLogger(__name__)

def time_it(func):
    """
    一个简单的函数装饰器，用于测量并记录函数执行时间。
    """
    @functools.wraps(func) # 保留原函数的元信息 (如名称、文档字符串)
    def wrapper(*args, **kwargs):
        # 记录函数开始执行的时间
        start_time = time.time()
        # 执行原函数
        result = func(*args, **kwargs)
        # 记录函数结束执行的时间
        end_time = time.time()
        # 计算执行耗时
        elapsed_time = end_time - start_time
        # 使用 logger 记录函数名和耗时
        logger.info(f"函数 '{func.__name__}' 执行耗时: {elapsed_time:.4f} 秒")
        # 返回原函数的执行结果
        return result
    return wrapper

# --- 文本处理辅助函数 ---

def sanitize_input(text: str) -> str:
    """
    清理用户输入字符串。
    目前只执行去除首尾空格的操作，可以根据需要扩展。

    Args:
        text (str): 原始输入字符串。

    Returns:
        str: 清理后的字符串。
    """
    if isinstance(text, str):
        return text.strip()
    return text # 如果不是字符串，原样返回

# --- 可以在这里添加更多通用的辅助函数 ---
# 例如：
# - 检查文件是否存在并具有读取权限
# - 安全地读取 JSON 或 YAML 配置文件
# - 实现更复杂的文本清理逻辑（如去除特殊字符、转换为小写等）
# - 数据格式转换函数

# --- 可选的直接执行入口 (用于测试辅助函数) ---
if __name__ == '__main__':
    # 测试 logger
    logger.info("这是一个 utils 模块的 INFO 级别日志。")
    logger.warning("这是一个 utils 模块的 WARNING 级别日志。")

    # 测试 time_it 装饰器
    @time_it
    def example_function(duration):
        """一个休眠指定时间的示例函数。"""
        print(f"  示例函数开始执行，将休眠 {duration} 秒...")
        time.sleep(duration)
        print(f"  示例函数执行完毕。")
        return "完成"

    print("\n测试 @time_it 装饰器:")
    result = example_function(0.5)
    print(f"示例函数返回值: {result}")

    # 测试 sanitize_input
    print("\n测试 sanitize_input 函数:")
    dirty_inputs = ["  带空格的输入 ", "无空格", "\t带制表符\n", None, 123]
    for inp in dirty_inputs:
        clean_inp = sanitize_input(inp)
        print(f"  原始输入: {repr(inp)} -> 清理后: {repr(clean_inp)}")
