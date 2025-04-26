# src/utils.py
import time
import logging
import functools # 用于装饰器保留原函数信息

# --- 配置日志记录器 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

def time_it(func):
    """
    一个简单的函数装饰器，用于测量并记录函数执行时间。
    """
    @functools.wraps(func) # 保留原函数的元信息
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"函数 '{func.__name__}' 执行耗时: {elapsed_time:.4f} 秒")
        return result
    return wrapper

# --- 文本处理辅助函数 ---
def sanitize_input(text: str) -> str:
    """
    清理用户输入字符串。
    """
    if isinstance(text, str):
        return text.strip()
    return text # 如果不是字符串，原样返回

# --- 可选的直接执行入口 (用于测试辅助函数) ---
if __name__ == '__main__':
    logger.info("这是一个 utils 模块的 INFO 级别日志。")
    logger.warning("这是一个 utils 模块的 WARNING 级别日志。")

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

    print("\n测试 sanitize_input 函数:")
    dirty_inputs = ["  带空格的输入 ", "无空格", "\t带制表符\n", None, 123]
    for inp in dirty_inputs:
        clean_inp = sanitize_input(inp)
        print(f"  原始输入: {repr(inp)} -> 清理后: {repr(clean_inp)}")
