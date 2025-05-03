# new_sensor_project/src/pipeline/main_pipeline.py
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# --- Early Logging Setup ---
# Setup basic logging first to catch import errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get logger instance early

# --- Module Imports ---
# Use absolute imports by ensuring the project root is in sys.path
try:
    project_root = Path(__file__).resolve().parent.parent.parent # Get project root (new_sensor_project)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logger.info(f"Added project root to sys.path: {project_root}")

    from config import settings
    # Import necessary modules using absolute paths from project root
    from src.utils import logging_config
    from src.info_extractor.extractor import InfoExtractor
    from src.parameter_standardizer.search_service import SearchService
    # standard_matcher is no longer used as code generation is removed
    # from src.standard_matcher.matcher import generate_product_code

    # Now setup the full logging configuration from the utils module
    # Re-get logger after full setup to apply configured handlers/formatters
    logging_config.setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Module imports successful and full logging configured.")

except ImportError as e:
    logger.exception(f"CRITICAL: Failed to import necessary modules: {e}. Check PYTHONPATH and project structure.")
    # Print to stderr as well, in case logging to file fails
    print(f"CRITICAL: Failed to import necessary modules: {e}. Check PYTHONPATH and project structure.", file=sys.stderr)
    sys.exit(1)


# --- 人工核对辅助函数 ---

def prompt_for_manual_check(prompt_message: str) -> bool:
    """通用的人工确认提示函数"""
    while True:
        # 使用 print 直接输出到控制台，绕过日志级别限制
        print(f"\n--- 人工核对 ---")
        print(prompt_message)
        response = input("完成后请按 Enter 继续，或输入 'skip' 跳过当前项，输入 'abort' 中止整个流程: ").lower().strip()
        if response == '':
            logger.info("人工确认：继续处理。")
            return True # 继续
        elif response == 'skip':
            logger.warning("人工确认：跳过当前项。")
            return False # 跳过当前项（例如设备）
        elif response == 'abort':
            logger.error("人工确认：中止流程。")
            raise KeyboardInterrupt("用户中止流程") # 使用异常中断流程
        else:
            print("无效输入。请按 Enter, 输入 'skip', 或 'abort'。")

def save_and_verify_json(data: Dict[str, Any], file_path: Path, prompt_prefix: str) -> Optional[Dict[str, Any]]:
    """保存 JSON 数据，提示用户核对/修改，然后重新加载"""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"{prompt_prefix} 数据已保存至: {file_path}")
    except Exception as e:
        logger.error(f"保存 {prompt_prefix} JSON 数据到 {file_path} 时出错: {e}", exc_info=True)
        print(f"错误：无法保存 {prompt_prefix} JSON 文件。")
        return None # 保存失败，无法继续

    prompt = f"{prompt_prefix} 数据已保存至文件，请检查或修改:\n{file_path}"
    if not prompt_for_manual_check(prompt):
        # 如果用户选择 skip 或 abort (abort 会抛异常)
        # 对于整体 JSON，skip 没有意义，视为 abort
        logger.error("提取的 JSON 数据核对未通过或被跳过，处理中止。")
        return None # 返回 None 表示核对失败或跳过

    # 重新加载可能被修改的文件
    try:
        logger.info(f"重新加载核对后的文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            reloaded_data = json.load(f)
        logger.info("文件重新加载成功。")
        return reloaded_data
    except FileNotFoundError:
        logger.error(f"错误：重新加载时未找到文件 {file_path}。流程中止。")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"错误：重新加载文件 {file_path} 时 JSON 解析失败: {e}。请确保文件格式正确。流程中止。")
        print(f"错误：文件 {file_path} 包含无效的 JSON。请修正后重试。")
        return None
    except Exception as e:
        logger.error(f"重新加载文件 {file_path} 时发生意外错误: {e}", exc_info=True)
        return None

# Note: save_and_verify_json is still used for extracted parameters check.

def process_document(input_file_path: Path, skip_extraction: bool = False) -> Optional[Path]:
    """
    处理单个输入文档，生成标准化参数文件。

    Args:
        input_file_path: 输入文档的路径 (例如 PDF)。
        skip_extraction: 如果为 True，则尝试跳过提取和标准化，直接加载中间文件。

    Returns:
        Optional[Path]: 成功时返回标准化参数文件的 Path 对象，失败时返回 None。
    """
    logger.info(f"===== 开始处理文档: {input_file_path.name} (跳过提取: {skip_extraction}) =====")
    all_processed_devices = [] # 初始化为空列表
    combined_standardized_path = settings.OUTPUT_DIR / f"{input_file_path.stem}_standardized_all.json"

    # --- 尝试跳过提取和标准化 ---
    if skip_extraction:
        logger.info(f"尝试跳过提取，检查文件: {combined_standardized_path}")
        if combined_standardized_path.is_file():
            try:
                # 仅检查文件是否存在且有效，不真正加载全部内容（除非需要验证）
                # 可以在这里添加一个快速的 JSON 验证逻辑如果需要
                logger.info(f"找到已存在的标准化文件: {combined_standardized_path}。跳过提取和标准化步骤。")
                logger.info(f"===== 文档处理完成 (使用已存在文件): {input_file_path.name} =====")
                return combined_standardized_path # 返回已存在文件的路径
            except Exception as e:
                logger.error(f"检查已存在的标准化文件 {combined_standardized_path.name} 时出错: {e}。将继续执行完整流程。", exc_info=True)
                # 出错则继续执行完整流程，所以这里不返回
        else:
            logger.error(f"请求跳过提取，但标准化文件 {combined_standardized_path.name} 未找到。无法继续。")
            return None # 无法跳过且文件不存在

    # --- 如果不跳过或跳过失败，则执行完整提取和标准化 ---
    # --- 1. 初始化服务 ---
    try:
        logger.info("初始化 InfoExtractor...")
        info_extractor = InfoExtractor()
        logger.info("初始化 SearchService...")
        search_service = SearchService()

        if not search_service.is_ready():
             logger.error("SearchService 未就绪，无法继续处理。")
             return None

    except Exception as e:
        logger.exception(f"初始化服务时出错: {e}")
        return None

    # --- 2. 信息提取 ---
    try:
        logger.info("开始信息提取...")
        extracted_data = info_extractor.extract_parameters_from_pdf(input_file_path)
        if extracted_data is None:
            logger.error("从PDF提取参数失败。")
            return None # 无法继续
        logger.info("信息提取成功。")

        # --- 插入：提取后的人工核对 ---
        extracted_path = settings.OUTPUT_DIR / f"{input_file_path.stem}_extracted_parameters.json"
        verified_extracted_data = save_and_verify_json(extracted_data, extracted_path, "提取的参数")
        if verified_extracted_data is None:
            logger.error("提取的参数核对失败或被中止。")
            return None # 核对失败，中止流程
        logger.info("提取的参数核对完成。")
        # 使用核对后的数据进行下一步
        extracted_data = verified_extracted_data
        # --- 结束：提取后的人工核对 ---

    except KeyboardInterrupt: # 捕获用户在核对中中止信号
        logger.warning("流程在提取核对阶段被用户中止。")
        return None # 中止整个处理
    except Exception as e:
        logger.exception(f"信息提取或首次核对过程中发生意外错误: {e}")
        return None

    # --- 3. 适配新的 JSON 结构并仅标准化共用参数 ---
    # 读取设备组列表
    device_group_list = extracted_data.get("设备列表", [])
    if not device_group_list:
        logger.warning("提取的 JSON 数据中未找到 '设备列表' 或列表为空。")
        return None # 返回 None 表示处理失败

    logger.info(f"开始标准化 {len(device_group_list)} 个设备组的共用参数...")
    all_processed_devices = [] # 重新初始化列表，用于存储处理后的设备组

    for idx, device_group in enumerate(device_group_list):
        # 获取位号列表和共用参数
        tag_nos = device_group.get("位号", [f"未知组_{idx+1}"]) # Use list for consistency
        common_params = device_group.get("共用参数", {})
        # Use the first tag number for logging, or a default name
        group_log_name = tag_nos[0] if tag_nos else f"未知组_{idx+1}"

        logger.info(f"--- 开始标准化设备组: {group_log_name} (包含 {len(tag_nos)} 个位号) ({idx+1}/{len(device_group_list)}) ---")

        # 创建处理后的设备组信息，包含位号
        processed_group = {"位号": tag_nos}
        actual_params = common_params # 直接使用共用参数进行标准化

        if not actual_params:
            logger.warning(f"设备组 '{group_log_name}' 没有共用参数信息，跳过标准化。")
            processed_group['标准化共用参数'] = {} # Use a distinct key
            all_processed_devices.append(processed_group)
            continue

        # a. 参数标准化 (仅针对共用参数)
        standardized_params_result: Dict[str, str] = {}
        logger.debug(f"开始标准化设备组 '{group_log_name}' 的 {len(actual_params)} 个共用参数...")
        for actual_name, actual_value in actual_params.items():
            # 保持原有的值类型检查
            if not isinstance(actual_value, (str, int, float)):
                 logger.warning(f"共用参数 '{actual_name}' 的值类型不支持标准化 ({type(actual_value)})，跳过。")
                 continue
            actual_value_str = str(actual_value)

            # 标准化逻辑保持不变
            search_result: Optional[Tuple[str, str, str]] = None
            try:
                search_result = search_service.search(actual_name, actual_value_str)
            except Exception as e:
                 logger.error(f"为共用参数 '{actual_name}'='{actual_value_str}' 调用 SearchService 时出错: {e}", exc_info=True)

            if search_result:
                std_name, _, _ = search_result
                # 保留原始值，键使用标准名和原始名组合 (逻辑不变)
                output_key = f"{std_name} ({actual_name})"
                standardized_params_result[output_key] = actual_value_str
                logger.debug(f"  '{actual_name}':'{actual_value_str}' -> 标准化键:'{output_key}', 值:'{actual_value_str}'")
            else:
                logger.warning(f"未能为共用参数 '{actual_name}':'{actual_value_str}' 找到标准匹配，保留原始参数。")
                # 保留原始键值对 (逻辑不变)
                standardized_params_result[actual_name] = actual_value_str

        if not standardized_params_result:
             logger.warning(f"设备组 '{group_log_name}' 处理后共用参数列表为空。")
             processed_group['标准化共用参数'] = {} # Use a distinct key
        else:
             logger.info(f"设备组 '{group_log_name}' 共用参数标准化完成，共 {len(standardized_params_result)} 个标准参数。")
             processed_group['标准化共用参数'] = standardized_params_result # Use a distinct key

        all_processed_devices.append(processed_group) # 添加处理后的设备组
        logger.info(f"--- 设备组 {group_log_name} 标准化处理完毕 ---")

    # --- 保存标准化结果 ---
    if all_processed_devices:
        # 使用之前定义的 combined_standardized_path
        final_standardized_data = {"标准化设备组列表": all_processed_devices} # Use a more descriptive key

        try:
            combined_standardized_path.parent.mkdir(parents=True, exist_ok=True)
            with open(combined_standardized_path, 'w', encoding='utf-8') as f:
                json.dump(final_standardized_data, f, ensure_ascii=False, indent=4)
            logger.info(f"标准化参数数据已成功保存至: {combined_standardized_path}")
            logger.info(f"===== 文档处理完成: {input_file_path.name} =====")
            return combined_standardized_path # 成功保存，返回文件路径
        except Exception as e:
            logger.error(f"保存标准化 JSON 数据到 {combined_standardized_path} 时出错: {e}", exc_info=True)
            print(f"错误：无法保存标准化 JSON 文件。")
            return None # 保存失败
    else:
        logger.warning("没有设备组被成功处理并标准化，无法保存标准化文件。")
        return None # 没有数据可保存


def main():
    """主函数：解析命令行参数并启动处理流程。"""
    # --- 0. 配置日志 ---
    try:
        logging_config.setup_logging()
    except Exception as e:
        print(f"致命错误：无法配置日志系统: {e}")
        sys.exit(1)

    # --- 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="一体式温度变送器参数提取与标准化系统")
    parser.add_argument(
        "input_file",
        type=str,
        help="要处理的输入文档路径 (例如 'data/input/温变规格书.pdf')"
    )
    # 移除 --output-json 参数
    # parser.add_argument(
    #     "--output-json",
    #     type=str,
    #     default=None,
    #     help="可选：保存最终结果 (位号到产品代码映射) 的 JSON 文件路径。"
    # )
    parser.add_argument(
        "--skip-extraction",
        action="store_true", # 设置为布尔标志
        help="如果指定，则跳过信息提取和参数标准化步骤，尝试加载已存在的标准化文件。"
    )
    # 可以添加更多参数，例如 --log-level 来覆盖 settings.py 中的设置

    args = parser.parse_args()

    input_file = Path(args.input_file)

    # 检查输入文件是否存在
    if not input_file.is_file():
        logger.critical(f"输入文件未找到: {input_file}")
        print(f"错误: 输入文件未找到: {input_file}")
        sys.exit(1)

    # --- 启动处理 ---
    # 传递 skip_extraction 参数
    # 接收返回的路径或 None
    standardized_file_path = process_document(input_file, skip_extraction=args.skip_extraction)

    # --- 输出结果 ---
    if standardized_file_path is not None:
        # 处理成功
        print("\n--- 处理成功 ---")
        print(f"标准化参数已保存至文件:")
        print(standardized_file_path)
        # 正常退出
        sys.exit(0)
    else:
        # 处理失败
        print("\n处理过程中发生错误或未生成标准化文件。请检查日志文件获取详细信息。")
        sys.exit(1) # 以错误码退出

if __name__ == "__main__":
    main()
