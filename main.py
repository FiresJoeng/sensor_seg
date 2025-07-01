# main.py (formerly new_sensor_project/src/pipeline/main_pipeline.py)
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
    # When main.py is at the project root:
    project_root = Path(__file__).resolve().parent 
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        # Logger might not be fully configured here, basic log is fine
        logging.info(f"Added project root to sys.path: {project_root}")

    from config import settings
    # Import necessary modules using absolute paths from project root
    from src.utils import logging_config
    from src.info_extractor.extractor import InfoExtractor
    # from src.parameter_standardizer.search_service import SearchService # 已移除
    # --- Import the new standardizer ---
    from src.parameter_standardizer.accurate_llm_standardizer import AccurateLLMStandardizer
    # Import OpenAI for Gemini compatibility
    from openai import OpenAI
    # Import the new CodeAssembler
    from src.code_assembler.assembler import CodeAssembler
    # The refactored standard matching function (execute_standard_matching) will be replaced
    # from src.standard_matcher.standard_matcher import execute_standard_matching

    # Now setup the full logging configuration from the utils module
    # Re-get logger after full setup to apply configured handlers/formatters
    logging_config.setup_logging() # This will reconfigure based on settings
    logger = logging.getLogger(__name__) # Get the fully configured logger
    logger.info("Module imports successful and full logging configured.")

except ImportError as e:
    # Use basic logging if full config failed
    logging.exception(f"CRITICAL: Failed to import necessary modules: {e}. Check PYTHONPATH and project structure.")
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

def process_document(input_file_path: Path) -> Optional[Path]: # 移除了 skip_extraction 参数
    """
    处理单个输入文档，生成标准化参数文件。
    (始终执行提取和标准化流程)

    Args:
        input_file_path: 输入文档的路径 (例如 PDF)。

    Returns:
        Optional[Path]: 成功时返回标准化参数文件的 Path 对象，失败时返回 None。
    """
    logger.info(f"===== 开始处理文档: {input_file_path.name} (始终执行完整提取流程) =====")
    combined_standardized_path = settings.OUTPUT_DIR / f"{input_file_path.stem}_standardized_all.json"

    # --- 跳过提取的逻辑已移除，始终执行完整提取和标准化 ---
    # --- 如果不跳过或跳过失败，则执行完整提取和标准化 ---
    # --- 1. 初始化服务 ---
    try:
        logger.info("初始化 InfoExtractor...")
        info_extractor = InfoExtractor()
        # logger.info("初始化 SearchService...") # 已移除
        # search_service = SearchService() # 已移除

        # if not search_service.is_ready(): # 已移除
        #      logger.error("SearchService 未就绪，无法继续处理。") # 已移除
        #      return None # 已移除

        # --- Initialize LLM client (using OpenAI for Gemini compatibility) ---
        try:
            api_key = settings.LLM_API_KEY
            base_url = settings.LLM_API_URL
            timeout = settings.LLM_REQUEST_TIMEOUT

            if not api_key:
                logger.error("配置错误：在 settings.py 中未找到 LLM_API_KEY。")
                raise ValueError("缺少 LLM_API_KEY 配置")
            if not base_url:
                 logger.warning("配置警告：在 settings.py 中未找到 LLM_API_URL，使用默认值。")

            logger.info("初始化 LLM 客户端 (OpenAI 兼容)...")
            llm_client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        except AttributeError:
             logger.error("配置错误：无法加载 settings 或 LLM 配置。")
             return None
        except ValueError as ve:
             logger.error(ve)
             return None
        except Exception as e:
             logger.exception(f"初始化 LLM 客户端时出错: {e}")
             return None

        # --- Initialize the AccurateLLMStandardizer with the new client ---
        logger.info("初始化 AccurateLLMStandardizer...")
        standardizer = AccurateLLMStandardizer(client=llm_client) # 不再传递 search_service

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
            logger.error("提取的参数和备注核de失败或被中止。")
            return None # 如果核对失败或中止，返回 None
        # 移除此处错误的 return None
        logger.info("提取的参数和备注核对完成。")
        # 添加阶段一完成打印
        print(f"\n--- 阶段一：信息提取与人工核对 完成 ---")
        print(f"已提取并核对的文件: {extracted_path}")

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
        
        # --- 人工检查点：在此处插入，因为文件已保存 ---
        print(f"\n--- 人工检查点：检查标准化文件 ---")
        print(f"标准化参数已保存至: {combined_standardized_path}")
        print(f"请在继续前检查或修改此文件。")
        input("检查或修改完毕后，请按 Enter键 继续...")
        logger.info(f"用户已确认或修改了标准化文件: {combined_standardized_path}")
        # 尝试重新加载文件，以防用户修改了它并引入了JSON错误
        try:
            with open(combined_standardized_path, 'r', encoding='utf-8') as f_reload:
                json.load(f_reload) # 仅用于验证JSON格式
            logger.info(f"重新加载确认后的文件 {combined_standardized_path} (验证通过)。")
        except Exception as e_reload:
            logger.error(f"重新加载文件 {combined_standardized_path} 时出错: {e_reload}。可能JSON格式已损坏。将使用原始保存版本。", exc_info=True)
            # 考虑是否应该在此处中止或强制用户修复
            print(f"警告：重新加载文件 {combined_standardized_path} 失败。如果已修改，请确保其为有效的JSON。")
            # 让流程继续，但用户需注意文件可能未按预期更新
        
        # 添加阶段二完成打印
        print(f"\n--- 阶段二：参数标准化与人工核对 完成 ---")
        print(f"已标准化并核对的文件: {combined_standardized_path}")

        logger.info(f"===== 文档处理完成 (包含人工检查): {input_file_path.name} =====")
        return combined_standardized_path
    except Exception as e:
        logger.error(f"保存最终合并标准化 JSON 数据到 {combined_standardized_path} 时出错: {e}", exc_info=True)
        print(f"错误：无法保存最终的 JSON 文件。")
        return None


def main():
    """主函数：提示用户输入文件路径并启动处理流程。""" # Docstring updated
    try:
        # Logging is now fully configured after imports
        pass # No need to call setup_logging() again here if done after imports
    except Exception as e:
        # This basic logger will be used if full config failed
        logging.critical(f"致命错误：日志系统配置可能存在问题: {e}")
        # Fallback print if logging itself is broken
        print(f"致命错误：无法配置日志系统: {e}", file=sys.stderr)
        sys.exit(1)

    while True:
        raw_path = input("请输入要处理的输入文档路径: ").strip()
        if not raw_path:
            print("错误：文件路径不能为空。请重新输入。")
            logger.warning("用户输入了空文件路径。")
            continue
        input_file = Path(raw_path)
        if input_file.is_file():
            logger.info(f"用户输入文件路径: {input_file}")
            break
        else:
            logger.error(f"用户提供的输入文件未找到: {input_file}")
            print(f"错误: 输入文件 '{input_file}' 未找到或不是一个文件。请检查路径并重新输入。")
            retry_choice = input("是否重试输入路径？(y/n，默认为 y): ").lower().strip()
            if retry_choice == 'n':
                logger.info("用户选择不重试文件路径输入，程序中止。")
                sys.exit(1)
            # 默认为重试 (空输入或其他非 'n' 输入)
    
    # 用户要求始终处理，不再询问是否跳过提取，也不再需要 skip_extraction 标志
    logger.info(f"将始终执行提取和标准化流程。")

    standardized_file_path = process_document(input_file)

    # Get the base name from the initial input file for cleanup later
    input_file_stem = input_file.stem
    base_name_for_cleanup = input_file_stem.replace('_analysis', '').replace('_standardized_all', '')
    extracted_file_to_clean = settings.OUTPUT_DIR / f"{base_name_for_cleanup}_extracted_parameters_with_remarks.json"

    if standardized_file_path is not None:
        logger.info(f"\n--- 开始阶段三：型号代码组装 ---")
        print(f"\n--- 开始阶段三：型号代码组装 ---")
        print(f"输入文件进行代码组装: {standardized_file_path}")

        try:
            # 重新加载标准化后的数据
            logger.info(f"加载标准化文件进行代码组装: {standardized_file_path}")
            with open(standardized_file_path, 'r', encoding='utf-8') as f:
                standardized_data = json.load(f)
            logger.info("标准化文件加载成功。")

            # 重新初始化 LLM 客户端，确保在 main 函数作用域内可用
            llm_client = OpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_API_URL, timeout=settings.LLM_REQUEST_TIMEOUT)
            code_assembler = CodeAssembler(client=llm_client)
            logger.info("CodeAssembler 初始化成功。")

            # 调用 CodeAssembler 进行代码组装
            logger.info("开始调用LLM组装型号代码...")
            assembled_results = code_assembler.assemble_code(standardized_data)
            logger.info("LLM型号代码组装完成。")

            # --- 人工输入缺失代码 (新版) ---
            final_results = []
            for result in assembled_results:
                # 确保位号是字符串
                tag_no_raw = result.get("位号")
                tag_no = ", ".join(tag_no_raw) if isinstance(tag_no_raw, list) else str(tag_no_raw)

                assembled_code = result.get("型号代码", "组装失败")
                missing_params = result.get("缺失参数", [])
                
                # 检查是否有占位符 '□' 并且有缺失参数列表
                if "□" in assembled_code and missing_params:
                    print(f"\n--- 人工输入：位号 {tag_no} 型号代码不完整 ---")
                    
                    # 循环直到所有占位符都被处理
                    while "□" in assembled_code:
                        print(f"当前型号代码: {assembled_code}")
                        
                        # 找出当前第一个 '□' 对应的缺失参数
                        # 注意：这个逻辑假设 '□' 的出现顺序与 missing_params 列表的顺序一致
                        # 这是一个合理的简化假设
                        current_missing_param = next((p for p in missing_params if p), None)
                        
                        if not current_missing_param:
                            # 如果没有更多具体的缺失参数名，但仍有 '□'，说明逻辑可能不匹配
                            # 此时退回到旧的通用提示
                            print(f"警告: 无法确定下一个缺失参数的名称。")
                            prompt_text = f"请为位号 '{tag_no}' 输入下一个缺失的代码: "
                        else:
                            prompt_text = f"请输入 '{current_missing_param}' 的代码 (留空则使用'-'): "

                        # 打印出所有剩余的缺失参数
                        print(f"剩余待输入参数: {', '.join(p for p in missing_params if p)}")
                        
                        user_input = input(prompt_text).strip()
                        
                        # 根据用户输入进行替换
                        replacement = user_input if user_input else "-"
                        assembled_code = assembled_code.replace("□", replacement, 1)
                        
                        # 从 missing_params 列表中移除已处理的参数
                        if current_missing_param:
                            try:
                                missing_params.remove(current_missing_param)
                            except ValueError:
                                # 如果参数不在列表中，忽略错误，继续
                                pass
                        
                        print(f"更新后型号代码: {assembled_code}\n")

                final_results.append({
                    "位号": tag_no,
                    "最终型号代码": assembled_code,
                    "剩余未填参数": [p for p in missing_params if p] # 记录最终还缺哪些
                })

            # 保存最终结果
            final_result_path = settings.OUTPUT_DIR / f"{input_file_stem}_results.json" # 统一命名为 _results.json
            with open(final_result_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=4)
            
            logger.info(f"--- 阶段三：型号代码组装与人工输入 完成 ---")
            print(f"\n--- 阶段三：型号代码组装与人工输入 完成 ---")
            logger.info(f"最终型号代码结果已保存至: {final_result_path}")
            print(f"最终型号代码结果已保存至: {final_result_path}")

            logger.info(f"\n===== 完整流程处理成功 =====")
            print(f"\n===== 完整流程处理成功 =====")
            sys.exit(0)

        except Exception as e:
            logger.error(f"型号代码组装流程失败: {e}", exc_info=True)
            print("\n--- 型号代码组装流程失败 ---")
            print("未能完成型号代码组装。请检查日志文件获取详细信息。")
            sys.exit(1)
    else:
        logger.error("标准化流程失败，无法进行型号代码组装。")
        print("\n--- 标准化流程失败 ---")
        print("未能完成标准化。请检查日志文件获取详细信息。")
        sys.exit(1)

    # --- 清理临时文件 ---
    logger.info("开始清理临时文件...")
    files_to_clean = [extracted_file_to_clean, standardized_file_path] # 清理提取阶段和标准化阶段的中间文件
    for file_path in files_to_clean:
        try:
            if file_path and file_path.is_file(): # 检查 file_path 是否为 None
                file_path.unlink()
                logger.info(f"成功删除临时文件: {file_path}")
            else:
                logger.debug(f"临时文件不存在或路径无效，无需删除: {file_path}")
        except OSError as e:
            logger.error(f"删除临时文件 '{file_path}' 时出错: {e}", exc_info=True)
    logger.info("临时文件清理完成。")


if __name__ == "__main__":
    main()
