# new_sensor_project/src/pipeline/main_pipeline.py
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Hashable

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
    # --- Import the new standardizer and ZhipuAI ---
    from src.parameter_standardizer.accurate_llm_standardizer import AccurateLLMStandardizer
    from zhipuai import ZhipuAI
    # standard_matcher is no longer used
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

def get_dict_hash(data: Dict[str, Any]) -> Hashable:
    """将字典转换为可哈希的表示形式（用于缓存键）。"""
    # 排序字典项以确保一致性，然后转换为元组
    return tuple(sorted(data.items()))

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
    combined_standardized_path = settings.OUTPUT_DIR / f"{input_file_path.stem}_standardized_all.json"

    # --- 尝试跳过提取和标准化 ---
    if skip_extraction:
        logger.info(f"尝试跳过提取，检查文件: {combined_standardized_path}")
        if combined_standardized_path.is_file():
            try:
                logger.info(f"找到已存在的标准化文件: {combined_standardized_path}。跳过提取和标准化步骤。")
                logger.info(f"===== 文档处理完成 (使用已存在文件): {input_file_path.name} =====")
                return combined_standardized_path
            except Exception as e:
                logger.error(f"检查已存在的标准化文件 {combined_standardized_path.name} 时出错: {e}。将继续执行完整流程。", exc_info=True)
        else:
            logger.error(f"请求跳过提取，但标准化文件 {combined_standardized_path.name} 未找到。无法继续。")
            return None

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

        # --- Initialize ZhipuAI client ---
        try:
            api_key = getattr(settings, 'ZHIPUAI_API_KEY', None)
            if not api_key:
                logger.error("配置错误：在 settings.py 中未找到 ZHIPUAI_API_KEY。")
                raise ValueError("缺少 ZHIPUAI_API_KEY 配置")
            logger.info("初始化 ZhipuAI 客户端...")
            zhipuai_client = ZhipuAI(api_key=api_key)
        except AttributeError:
             logger.error("配置错误：无法加载 settings 或 ZHIPUAI_API_KEY。")
             return None
        except ValueError as ve:
             logger.error(ve)
             return None
        except Exception as e:
             logger.exception(f"初始化 ZhipuAI 客户端时出错: {e}")
             return None

        # --- Initialize the new AccurateLLMStandardizer ---
        logger.info("初始化 AccurateLLMStandardizer...")
        standardizer = AccurateLLMStandardizer(search_service=search_service, client=zhipuai_client)

    except Exception as e:
        logger.exception(f"初始化服务时出错: {e}")
        return None

    # --- 2. 信息提取 ---
    try:
        logger.info("开始信息提取...")
        extracted_data = info_extractor.extract_parameters_from_pdf(input_file_path)
        if extracted_data is None:
            logger.error("从PDF提取参数失败。")
            return None
        logger.info("信息提取成功。")

        remarks = info_extractor.json_proc.extract_remarks(extracted_data)
        if remarks:
            logger.info(f"提取到的备注信息: {remarks}")
        else:
            remarks = {}

        # --- 插入：提取后的人工核对 (包含备注) ---
        extracted_path = settings.OUTPUT_DIR / f"{input_file_path.stem}_extracted_parameters_with_remarks.json"
        verified_extracted_data = save_and_verify_json(extracted_data, extracted_path, "提取的参数和备注")
        if verified_extracted_data is None:
            logger.error("提取的参数和备注核对失败或被中止。")
            return None # 如果核对失败或中止，返回 None
        # 移除此处错误的 return None
        logger.info("提取的参数和备注核对完成。")

    except KeyboardInterrupt:
        logger.warning("流程在提取或核对阶段被用户中止。")
        return None
    except Exception as e:
        logger.exception(f"信息提取或核对过程中发生意外错误: {e}")
        return None

    # --- 3. 参数标准化 (处理提取并核对后的完整分组数据) ---
    logger.info("开始使用 AccurateLLMStandardizer 标准化提取并核对后的完整分组数据...")
    # 调用修改后的 standardizer.standardize 方法，输入是人工核对后的完整数据
    standardized_grouped_data = standardizer.standardize(verified_extracted_data)

    if standardized_grouped_data is None:
        logger.error("完整分组数据标准化失败。")
        return None
    logger.info("完整分组数据标准化完成。")

    # --- 4. 参数合并 (合并标准化后的分组数据) ---
    logger.info("开始合并标准化后的设备组参数...")
    # 调用修改后的 info_extractor.json_proc.merge_parameters 方法，输入是标准化后的分组数据
    final_merged_data = info_extractor.json_proc.merge_parameters(standardized_grouped_data)

    if final_merged_data is None or "设备列表" not in final_merged_data:
        logger.error("合并标准化后的参数失败或结果格式不正确。")
        return None
    logger.info(f"标准化参数合并完成，得到 {len(final_merged_data.get('设备列表', []))} 个独立设备。")

    # --- 5. 保存最终的合并标准化结果 ---
    logger.info(f"准备保存最终的合并标准化结果...")
    final_output_data = final_merged_data # 最终数据就是合并后的结果

    try:
        combined_standardized_path.parent.mkdir(parents=True, exist_ok=True)
        with open(combined_standardized_path, 'w', encoding='utf-8') as f:
            json.dump(final_output_data, f, ensure_ascii=False, indent=4)
        logger.info(f"最终合并标准化参数数据已成功保存至: {combined_standardized_path}")
        logger.info(f"===== 文档处理完成: {input_file_path.name} =====")
        return combined_standardized_path
    except Exception as e:
        logger.error(f"保存最终合并标准化 JSON 数据到 {combined_standardized_path} 时出错: {e}", exc_info=True)
        print(f"错误：无法保存最终的 JSON 文件。")
        return None


def main():
    """主函数：解析命令行参数并启动处理流程。"""
    try:
        logging_config.setup_logging()
    except Exception as e:
        print(f"致命错误：无法配置日志系统: {e}")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="一体式温度变送器参数提取与标准化系统")
    parser.add_argument(
        "input_file",
        type=str,
        help="要处理的输入文档路径 (例如 'data/input/温变规格书.pdf')"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="如果指定，则跳过信息提取和参数标准化步骤，尝试加载已存在的标准化文件。"
    )

    args = parser.parse_args()
    input_file = Path(args.input_file)

    if not input_file.is_file():
        logger.critical(f"输入文件未找到: {input_file}")
        print(f"错误: 输入文件未找到: {input_file}")
        sys.exit(1)

    standardized_file_path = process_document(input_file, skip_extraction=args.skip_extraction)

    if standardized_file_path is not None:
        print("\n--- 处理成功 ---")
        print(f"标准化参数已保存至文件:")
        print(standardized_file_path)
        sys.exit(0)
    else:
        print("\n处理过程中发生错误或未生成标准化文件。请检查日志文件获取详细信息。")
        sys.exit(1)

if __name__ == "__main__":
    main()
