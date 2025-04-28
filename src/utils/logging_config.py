# new_sensor_project/src/utils/logging_config.py
import logging
import sys
from logging.handlers import RotatingFileHandler

# 尝试从 .config 导入设置，如果失败（例如，直接运行此脚本），则从 config 导入
try:
    from config import settings
except ImportError:
    # 如果作为脚本直接运行或在不同上下文中导入，尝试不同的导入路径
    # 这对于确保模块在不同执行方式下都能找到配置很有用
    try:
        from config import settings
    except ImportError:
        print("错误：无法导入配置文件 settings.py。请确保 PYTHONPATH 正确设置。")
        # 可以选择抛出异常或设置默认值
        raise

def setup_logging():
    """
    配置全局日志记录器。

    根据 config/settings.py 中的设置，配置日志级别、格式，
    并将日志输出到控制台和可选的文件。
    """
    log_level = settings.LOG_LEVEL
    log_format = settings.LOG_FORMAT
    log_date_format = settings.LOG_DATE_FORMAT
    log_to_file = settings.LOG_TO_FILE
    log_file = settings.LOG_FILE

    # 获取根日志记录器
    root_logger = logging.getLogger()
    # 设置根日志记录器的级别
    root_logger.setLevel(log_level)

    # 清除任何现有的处理器，以避免重复日志条目
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 创建格式化器
    formatter = logging.Formatter(log_format, datefmt=log_date_format)

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 如果配置了日志文件，则创建文件处理器
    if log_to_file:
        try:
            # 确保日志文件所在的目录存在
            log_file.parent.mkdir(parents=True, exist_ok=True)
            # 使用 RotatingFileHandler 实现日志轮转
            # maxBytes=5*1024*1024 表示每个日志文件最大 5MB
            # backupCount=3 表示保留最近 3 个备份文件
            file_handler = RotatingFileHandler(
                log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logging.info(f"日志将同时输出到文件: {log_file}")
        except Exception as e:
            logging.error(f"无法配置日志文件处理器: {e}", exc_info=True)
            print(f"错误：无法设置日志文件 {log_file}。错误信息: {e}") # 在日志系统完全工作前使用 print

    logging.info(f"日志系统已配置。级别: {logging.getLevelName(log_level)}")

# --- 可选的测试入口 ---
if __name__ == "__main__":
    # 这个部分仅在直接运行此脚本时执行，用于测试日志配置
    print("正在测试日志配置...")
    setup_logging()
    logging.debug("这是一条 DEBUG 级别的日志。")
    logging.info("这是一条 INFO 级别的日志。")
    logging.warning("这是一条 WARNING 级别的日志。")
    logging.error("这是一条 ERROR 级别的日志。")
    try:
        1 / 0
    except ZeroDivisionError:
        logging.exception("这是一条 EXCEPTION 级别的日志（包含堆栈跟踪）。")
    print(f"测试完成。请检查控制台输出和日志文件（如果已启用）：{settings.LOG_FILE if settings.LOG_TO_FILE else '未启用'}")
