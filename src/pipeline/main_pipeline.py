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
    from src.standard_matcher.matcher import generate_product_code

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

def process_document(input_file_path: Path) -> Optional[Dict[str, str]]:
    """
    处理单个输入文档的完整流水线。

    Args:
        input_file_path: 输入文档的路径 (例如 PDF)。

    Returns:
        Optional[Dict[str, str]]: 一个字典，键是设备位号，值是生成的产品代码字符串。
                                  如果处理失败则返回 None。
    """
    logger.info(f"===== 开始处理文档: {input_file_path.name} =====")
    final_results: Dict[str, str] = {} # 存储最终结果 {位号: 产品代码}

    # --- 1. 初始化服务 ---
    try:
        logger.info("初始化 InfoExtractor...")
        info_extractor = InfoExtractor()
        logger.info("初始化 SearchService...")
        search_service = SearchService()
        # StandardMatcher 是函数式的，不需要初始化实例

        if not search_service.is_ready():
             logger.error("SearchService 未就绪，无法继续处理。")
             return None

    except Exception as e:
        logger.exception(f"初始化服务时出错: {e}")
        return None

    # --- 2. 文档转换 (例如 PDF -> Markdown) ---
    md_file_path: Optional[Path] = None
    try:
        # 使用输入文件名作为输出 MD 文件名的一部分
        md_output_filename = f"{input_file_path.stem}_extracted.md"
        md_file_path = info_extractor.md_conv.convert_to_md(input_file_path, md_output_filename)
        if md_file_path is None:
            logger.error("文档转换为 Markdown 失败。")
            return None # 无法继续
    except Exception as e:
        logger.exception(f"文档转换过程中发生意外错误: {e}")
        return None

    # --- 3. 信息提取 (Markdown -> JSON) ---
    extracted_data: Optional[Dict[str, Any]] = None
    try:
        extracted_data = info_extractor.json_proc.md_to_json(md_file_path)
        if extracted_data is None:
            logger.error("从 Markdown 提取 JSON 数据失败。")
            # 可选：删除临时的 md 文件
            # if md_file_path and md_file_path.exists(): md_file_path.unlink()
            return None # 无法继续

        # 可选：保存提取的 JSON 数据到文件
        json_output_filename = f"{input_file_path.stem}_extracted.json"
        json_output_path = settings.OUTPUT_DIR / json_output_filename
        try:
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, ensure_ascii=False, indent=4)
            logger.info(f"提取的 JSON 数据已保存至: {json_output_path}")
        except Exception as e:
            logger.error(f"保存提取的 JSON 数据时出错: {e}", exc_info=True)
            # 即使保存失败，也继续处理内存中的数据

    except Exception as e:
        logger.exception(f"信息提取过程中发生意外错误: {e}")
        return None

    # --- 4. (可选) JSON 验证 ---
    # try:
    #     logger.info("开始验证提取的 JSON 数据...")
    #     validation_result = info_extractor.json_proc.json_check(extracted_data)
    #     extracted_data = validation_result["data"] # 使用可能被修正的数据
    #     if validation_result["issues"]:
    #         logger.warning(f"JSON 验证发现 {len(validation_result['issues'])} 个问题。详情请查看日志或备注。")
    #         # 可以选择是否在验证失败时中止
    #     if validation_result["modified"]:
    #         # 如果数据被修改，可以选择保存验证后的版本
    #         validated_json_path = settings.OUTPUT_DIR / f"{input_file_path.stem}_validated.json"
    #         try:
    #             with open(validated_json_path, 'w', encoding='utf-8') as f:
    #                 json.dump(extracted_data, f, ensure_ascii=False, indent=4)
    #             logger.info(f"验证并修正后的 JSON 数据已保存至: {validated_json_path}")
    #         except Exception as e:
    #             logger.error(f"保存验证后的 JSON 数据时出错: {e}", exc_info=True)
    # except Exception as e:
    #     logger.exception(f"JSON 验证过程中发生意外错误: {e}")
        # 根据需要决定是否中止

    # --- 5. 参数标准化与代码生成 (逐个设备处理) ---
    device_list = extracted_data.get("设备列表", [])
    if not device_list:
        logger.warning("提取的 JSON 数据中未找到 '设备列表' 或列表为空。")
        return {} # 返回空结果

    logger.info(f"开始处理 {len(device_list)} 个设备...")
    for i, device_info in enumerate(device_list):
        device_tag = device_info.get("位号", f"未知设备_{i+1}")
        actual_params = device_info.get("参数", {})
        logger.info(f"--- 处理设备: {device_tag} ({i+1}/{len(device_list)}) ---")

        if not actual_params:
            logger.warning(f"设备 '{device_tag}' 没有参数信息，跳过处理。")
            final_results[device_tag] = "无参数信息"
            continue

        # a. 参数标准化
        standardized_params: Dict[str, str] = {} # {标准参数名: 标准参数值}
        logger.debug(f"开始标准化设备 '{device_tag}' 的 {len(actual_params)} 个参数...")
        for actual_name, actual_value in actual_params.items():
            if not isinstance(actual_value, (str, int, float)): # 只处理简单类型的值
                 logger.warning(f"参数 '{actual_name}' 的值类型不支持标准化 ({type(actual_value)})，跳过。")
                 continue
            actual_value_str = str(actual_value) # 确保是字符串

            # 调用 SearchService
            search_result: Optional[Tuple[str, str, str]] = None
            try:
                search_result = search_service.search(actual_name, actual_value_str)
            except Exception as e:
                 logger.error(f"为参数 '{actual_name}'='{actual_value_str}' 调用 SearchService 时出错: {e}", exc_info=True)
                 # 可以选择如何处理失败的参数：跳过、赋默认值等

            if search_result:
                std_name, std_value, std_code = search_result
                # 存储标准化结果 (标准名 -> 标准值)
                standardized_params[std_name] = std_value
                logger.debug(f"  '{actual_name}':'{actual_value_str}' -> 标准名:'{std_name}', 标准值:'{std_value}' (代码:'{std_code}')")
            else:
                logger.warning(f"未能为参数 '{actual_name}':'{actual_value_str}' 找到标准匹配。")
                # 可以选择是否将未匹配的参数也传递给下一步，或记录下来

        if not standardized_params:
             logger.error(f"设备 '{device_tag}' 未能标准化任何参数，无法生成代码。")
             final_results[device_tag] = "无标准化参数"
             continue

        logger.debug(f"设备 '{device_tag}' 标准化参数结果: {standardized_params}")

        # b. 标准匹配与代码生成
        logger.info(f"开始为设备 '{device_tag}' 生成产品代码...")
        product_code: Optional[str] = None
        try:
            product_code = generate_product_code(standardized_params)
        except Exception as e:
            logger.exception(f"为设备 '{device_tag}' 生成产品代码时出错: {e}")

        if product_code:
            logger.info(f"设备 '{device_tag}' 生成的产品代码: {product_code}")
            final_results[device_tag] = product_code
        else:
            logger.error(f"未能为设备 '{device_tag}' 生成产品代码。")
            final_results[device_tag] = "代码生成失败"

    logger.info(f"===== 文档处理完成: {input_file_path.name} =====")
    return final_results


def main():
    """主函数：解析命令行参数并启动处理流程。"""
    # --- 0. 配置日志 ---
    try:
        logging_config.setup_logging()
    except Exception as e:
        print(f"致命错误：无法配置日志系统: {e}")
        sys.exit(1)

    # --- 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="一体式温度变送器参数处理与代码生成系统")
    parser.add_argument(
        "input_file",
        type=str,
        help="要处理的输入文档路径 (例如 'data/input/温变规格书.pdf')"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="可选：保存最终结果 (位号到产品代码映射) 的 JSON 文件路径。"
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
    results = process_document(input_file)

    # --- 输出结果 ---
    if results is not None:
        print("\n--- 处理结果 ---")
        # 使用 json.dumps 来美化打印字典
        print(json.dumps(results, indent=4, ensure_ascii=False))

        # 如果指定了输出 JSON 文件，则保存
        if args.output_json:
            output_json_path = Path(args.output_json)
            try:
                # 确保目录存在
                output_json_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                logger.info(f"最终结果已保存到 JSON 文件: {output_json_path}")
                print(f"\n最终结果已保存到: {output_json_path}")
            except Exception as e:
                logger.error(f"保存最终结果到 JSON 文件时出错: {e}", exc_info=True)
                print(f"\n错误：无法保存结果到 {output_json_path}")
    else:
        print("\n处理过程中发生错误，未能生成结果。请检查日志文件获取详细信息。")
        sys.exit(1) # 以错误码退出

if __name__ == "__main__":
    main()
