# -*- coding: utf-8 -*-
"""
主程序入口：整合传感器选型流程
"""

import argparse
import json
import logging
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Hashable

# --- 日志和路径设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 将项目根目录添加到 sys.path
try:
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logger.info(f"已将项目根目录添加到 sys.path: {project_root}")

    # --- 核心模块导入 ---
    from config import settings
    from src.utils import logging_config
    from src.info_extractor.extractor import InfoExtractor
    from src.parameter_standardizer.search_service import SearchService
    from src.parameter_standardizer.vector_store_manager import VectorStoreManager # 用于检查 KB
    from src.parameter_standardizer.accurate_llm_standardizer import AccurateLLMStandardizer
    from src.standard_matcher.standard_matcher import ModelMatcher, FetchCsvlist, CodeSelector, CodeGenerator # <--- 添加 CodeSelector, CodeGenerator
    # from zhipuai import ZhipuAI # 改用 OpenAI
    from openai import OpenAI # <--- 添加 OpenAI 导入

    # 配置完整的日志记录
    logging_config.setup_logging()
    logger = logging.getLogger(__name__) # 重新获取 logger 以应用配置
    logger.info("模块导入成功，日志系统已配置。")

except ImportError as e:
    logger.exception(f"关键模块导入失败: {e}。请检查 PYTHONPATH 和项目结构。")
    print(f"错误：关键模块导入失败: {e}。请检查 PYTHONPATH 和项目结构。", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    logger.exception(f"初始化过程中发生未知错误: {e}")
    print(f"错误：初始化过程中发生未知错误: {e}", file=sys.stderr)
    sys.exit(1)


# --- 辅助函数：人工核对 (来自 main_pipeline.py，功能更全) ---
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
    # 注意：这里的 prompt_for_manual_check 返回 False 代表 skip
    # 对于整个 JSON 文件的核对，skip 通常意味着不想继续使用这个文件，可以视为一种中止
    if not prompt_for_manual_check(prompt):
        logger.error(f"{prompt_prefix} 数据核对未通过或被跳过/中止。")
        return None # 返回 None 表示核对失败或跳过/中止

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

# --- 知识库检查与构建 ---
def check_and_build_kb():
    """检查知识库是否存在，如果不存在则尝试构建。"""
    try:
        # 假设 VectorStoreManager 有方法可以检查 KB 是否就绪
        # 或者直接检查 settings.VECTOR_STORE_PATH 是否存在且不为空
        kb_path = Path(settings.VECTOR_STORE_PATH)
        # 简单的检查：路径存在且是个目录（FAISS 通常是目录）
        if kb_path.exists() and kb_path.is_dir() and any(kb_path.iterdir()):
             logger.info(f"知识库已存在于: {kb_path}")
             return True
        else:
             logger.warning(f"知识库未找到或为空于: {kb_path}。尝试自动构建...")
             build_script_path = project_root / "scripts" / "build_kb.py"
             if not build_script_path.is_file():
                 logger.error(f"构建脚本未找到: {build_script_path}")
                 return False

             try:
                 # 使用 subprocess 运行构建脚本
                 logger.info(f"开始执行知识库构建脚本: {build_script_path}...")
                 # 注意：确保 Python 解释器路径正确，或者直接使用 sys.executable
                 result = subprocess.run([sys.executable, str(build_script_path)],
                                         capture_output=True, text=True, check=True, cwd=project_root)
                 logger.info("知识库构建脚本执行成功。")
                 logger.info(f"脚本输出:\n{result.stdout}")
                 # 再次检查 KB 是否已生成
                 if kb_path.exists() and kb_path.is_dir() and any(kb_path.iterdir()):
                     logger.info("知识库构建成功确认。")
                     return True
                 else:
                     logger.error("脚本执行后知识库仍然未找到或为空。")
                     if result.stderr:
                         logger.error(f"脚本错误输出:\n{result.stderr}")
                     return False
             except subprocess.CalledProcessError as e:
                 logger.error(f"执行知识库构建脚本失败。返回码: {e.returncode}")
                 logger.error(f"错误输出:\n{e.stderr}")
                 logger.error(f"标准输出:\n{e.stdout}")
                 return False
             except Exception as e:
                 logger.error(f"运行知识库构建脚本时发生意外错误: {e}", exc_info=True)
                 return False
    except AttributeError:
        logger.error("配置错误：settings.py 中未找到 VECTOR_STORE_PATH。无法检查或构建知识库。")
        return False
    except Exception as e:
        logger.error(f"检查或构建知识库时发生错误: {e}", exc_info=True)
        return False

# --- 新的核心提取与标准化函数 (源自 main_pipeline.py) ---
def run_extraction_and_standardization(
    input_file_path: Path,
    info_extractor: InfoExtractor,
    standardizer: AccurateLLMStandardizer
) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """
    执行信息提取、人工核对、参数标准化和参数合并。

    Args:
        input_file_path: 输入文档的路径。
        info_extractor: InfoExtractor 的实例。
        standardizer: AccurateLLMStandardizer 的实例。

    Returns:
        一个元组，包含最终合并后的数据字典 (如果成功) 和提取参数核对后的文件路径，
        或者 (None, None) 如果过程中发生错误或中止。
    """
    logger.info(f"--- 开始对文件 {input_file_path.name} 进行提取与标准化 ---")
    
    verified_extracted_data: Optional[Dict[str, Any]] = None
    path_to_verified_extracted_data: Optional[Path] = None

    # 1. 信息提取
    try:
        logger.info("开始信息提取...")
        extracted_data_raw = info_extractor.extract_parameters_from_pdf(input_file_path)
        if extracted_data_raw is None:
            logger.error("从PDF提取参数失败。")
            return None, None
        logger.info("信息提取成功。")

        # 备注提取逻辑 (如果需要，可以从 main_pipeline.py 移植过来，但当前 InfoExtractor 可能已包含)
        # remarks = info_extractor.json_proc.extract_remarks(extracted_data_raw)
        # if remarks:
        #     logger.info(f"提取到的备注信息: {remarks}")

        # 提取后的人工核对
        # 文件名可以自定义，例如 _extracted_verified.json
        extracted_checkpoint_path = settings.OUTPUT_DIR / f"{input_file_path.stem}_extracted_verified.json"
        verified_extracted_data = save_and_verify_json(extracted_data_raw, extracted_checkpoint_path, "提取的参数（待核对）")
        
        if verified_extracted_data is None:
            logger.error("提取的参数核对失败或被中止。")
            return None, None
        logger.info("提取的参数核对完成。")
        path_to_verified_extracted_data = extracted_checkpoint_path

    except KeyboardInterrupt:
        logger.warning("流程在提取或核对阶段被用户中止。")
        return None, None
    except Exception as e:
        logger.exception(f"信息提取或核对过程中发生意外错误: {e}")
        return None, None

    # 2. 参数标准化 (使用核对后的 verified_extracted_data)
    logger.info("开始使用 AccurateLLMStandardizer 标准化提取并核对后的完整分组数据...")
    standardized_grouped_data = standardizer.standardize(verified_extracted_data)

    if standardized_grouped_data is None:
        logger.error("完整分组数据标准化失败。")
        return None, path_to_verified_extracted_data # 即使标准化失败，也返回已核对的提取文件路径
    logger.info("完整分组数据标准化完成。")

    # (可选) 标准化后的人工核对 - main_pipeline.py 中没有这一步，但如果需要可以添加
    # standardized_checkpoint_path = settings.OUTPUT_DIR / f"{input_file_path.stem}_standardized_verified.json"
    # verified_standardized_data = save_and_verify_json(standardized_grouped_data, standardized_checkpoint_path, "标准化后的参数（待核对）")
    # if verified_standardized_data is None:
    #     logger.error("标准化参数核对失败或被中止。")
    #     return None, path_to_verified_extracted_data
    # logger.info("标准化参数核对完成.")
    # final_data_for_merging = verified_standardized_data
    final_data_for_merging = standardized_grouped_data # 如果没有标准化后核对

    # 3. 参数合并
    logger.info("开始合并标准化后的设备组参数...")
    final_merged_data = info_extractor.json_proc.merge_parameters(final_data_for_merging)

    if final_merged_data is None or "设备列表" not in final_merged_data:
        # 尝试处理单个设备的情况，如果顶层就是参数字典
        if isinstance(final_data_for_merging, dict) and not any(key.startswith("设备") for key in final_data_for_merging.keys()):
            logger.warning("合并结果似乎是单个设备，尝试包装成设备列表。")
            final_merged_data = {"设备列表": [final_data_for_merging]}
        else:
            logger.error("合并标准化后的参数失败或结果格式不正确。")
            return None, path_to_verified_extracted_data
            
    logger.info(f"标准化参数合并完成，得到 {len(final_merged_data.get('设备列表', []))} 个独立设备。")
    
    # 此函数不负责保存最终的 _standardized_all.json，这由调用者决定
    # 或者，如果需要，可以在这里保存一个类似 _merged_for_matching.json 的文件
    logger.info(f"--- 文件 {input_file_path.name} 的提取与标准化完成 ---")
    return final_merged_data, path_to_verified_extracted_data


# --- 主处理流程 ---
def process_order_file(input_file_path: Path, output_file_path: Path):
    """处理单个订单文件，执行完整的提取、标准化和匹配流程。"""
    logger.info(f"===== 开始处理订单文件: {input_file_path.name} =====")

    # --- 1. 初始化服务 (已更新，使用 OpenAI 客户端) ---
    info_extractor_service: Optional[InfoExtractor] = None
    standardizer_service: Optional[AccurateLLMStandardizer] = None
    try:
        logger.info("初始化 InfoExtractor...")
        info_extractor_service = InfoExtractor()
        
        logger.info("初始化 SearchService (供 Standardizer 使用)...")
        search_service_instance = SearchService()
        if not search_service_instance.is_ready():
            logger.error("SearchService 未就绪，无法继续处理。")
            return

        logger.info("初始化 LLM 客户端 (OpenAI 兼容)...")
        api_key = settings.LLM_API_KEY
        base_url = settings.LLM_API_URL
        timeout = settings.LLM_REQUEST_TIMEOUT

        if not api_key:
            logger.error("配置错误：在 settings.py 中未找到 LLM_API_KEY。")
            raise ValueError("缺少 LLM_API_KEY 配置")
        if not base_url:
            logger.warning("配置警告：在 settings.py 中未找到 LLM_API_URL，将使用 OpenAI 默认 API 地址。")

        llm_client_instance = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        
        logger.info("初始化 AccurateLLMStandardizer (使用 OpenAI 客户端)...")
        standardizer_service = AccurateLLMStandardizer(search_service=search_service_instance, client=llm_client_instance)
        logger.info("核心服务初始化完成。")

    except ValueError as ve: 
        logger.error(f"服务初始化配置错误: {ve}")
        return
    except Exception as e:
        logger.exception(f"初始化服务时发生意外错误: {e}")
        return

    # --- 2. 信息提取、标准化和合并 (调用新核心函数) ---
    standardized_data: Optional[Dict[str, Any]] = None
    path_to_verified_extracted_data_for_csv: Optional[Path] = None
    
    try:
        logger.info(f"开始对文件 {input_file_path.name} 执行提取、标准化和合并流程...")
        standardized_data, path_to_verified_extracted_data_for_csv = run_extraction_and_standardization(
            input_file_path,
            info_extractor_service,
            standardizer_service
        )

        if standardized_data is None:
            logger.error(f"未能为文件 {input_file_path.name} 生成最终的标准化和合并数据。处理中止。")
            return
        if path_to_verified_extracted_data_for_csv is None:
             logger.error(f"未能获取提取并核对后的数据文件路径 ({input_file_path.name})，无法进行后续的 CSV 列表获取。处理中止。")
             return
        logger.info(f"文件 {input_file_path.name} 的参数提取、标准化和合并完成。")

    except KeyboardInterrupt:
        logger.warning(f"流程在提取、标准化或合并阶段被用户中止 ({input_file_path.name})。")
        return
    except Exception as e:
        logger.exception(f"提取、标准化或合并过程中发生意外错误 ({input_file_path.name}): {e}")
        return

    # --- 3. 获取 CSV 文件列表映射 (使用 FetchCsvlist) ---
    csv_list_map: Optional[Dict[str, List[str]]] = None
    try:
        logger.info("开始通过 FetchCsvlist 获取 CSV 文件列表映射...")
        fetcher = FetchCsvlist()
        index_json_path = project_root / "libs" / "standard" / "index.json"

        logger.info(f"FetchCsvlist 将使用输入文件: {path_to_verified_extracted_data_for_csv}")
        csv_list_map = fetcher.fetch_csv_lists(
            input_json_path=path_to_verified_extracted_data_for_csv,
            index_json_path=index_json_path
        )
        
        if not csv_list_map:
            logger.error(f"未能通过 FetchCsvlist 获取标准库 CSV 文件列表映射 ({input_file_path.name})，无法进行后续匹配。")
            return 
        logger.info(f"通过 FetchCsvlist 成功为 {input_file_path.name} 获取 CSV 列表映射: {len(csv_list_map)} 个产品类型。")

    except Exception as e_fetch_csv:
        logger.exception(f"通过 FetchCsvlist 获取 CSV 列表映射时发生错误 ({input_file_path.name}): {e_fetch_csv}")
        return

    # --- 4. 数据准备与选型匹配 ---
    # final_matching_results = {} # 这个变量似乎没有在后续被有效使用来保存整体结果到文件
    try:
        logger.info("开始准备数据并进行选型匹配...")
        
        if standardized_data is None or csv_list_map is None:
            logger.error(f"标准化数据或CSV列表映射为空 ({input_file_path.name})，无法进行匹配。")
            return

        device_list = standardized_data.get("设备列表")
        if not isinstance(device_list, list):
            logger.error(f"标准化数据格式错误 ({input_file_path.name})：未找到'设备列表'或其不是列表。")
            if isinstance(standardized_data, dict) and "设备列表" not in standardized_data: # 已在 run_extraction_and_standardization 中处理
                 logger.warning(f"标准化数据顶层 ({input_file_path.name}) 似乎是单个设备参数，尝试按单个设备处理。")
                 device_list = standardized_data.get("设备列表") 
                 if not isinstance(device_list, list): 
                    logger.error(f"无法将单个设备数据 ({input_file_path.name}) 转换为列表进行匹配。")
                    return
            else:
                 logger.error(f"无法解析标准化数据 ({input_file_path.name}) 以进行匹配。")
                 return

        logger.info(f"准备为 {input_file_path.name} 中的 {len(device_list)} 个设备/参数组进行匹配...")
        # all_devices_matches = [] # 这个变量似乎未被有效使用
        all_device_final_codes = [] # 用于收集每个设备的最终代码结果

        for i, device_params_raw in enumerate(device_list):
            device_name = device_params_raw.get("设备名称", f"设备_{input_file_path.stem}_{i+1}") 
            logger.info(f"--- 开始匹配 {device_name} ---")
            current_device_result = {"设备名称": device_name, "产品型号": "处理失败或无有效代码"}


            if not isinstance(device_params_raw, dict):
                logger.error(f"{device_name} 的参数格式不是字典，跳过匹配。数据: {device_params_raw}")
                continue
            
            temp_device_param_file_path = settings.OUTPUT_DIR / f"{input_file_path.stem}_device_{i+1}_for_match.json"
            input_for_matcher_path: Optional[Path] = None
            try:
                with open(temp_device_param_file_path, 'w', encoding='utf-8') as temp_f:
                    json.dump(device_params_raw, temp_f, ensure_ascii=False, indent=4)
                logger.info(f"为设备 {device_name} 创建了临时匹配输入文件: {temp_device_param_file_path}")
                input_for_matcher_path = temp_device_param_file_path
            except Exception as e_temp_save:
                logger.error(f"为设备 {device_name} 创建临时匹配输入文件失败: {e_temp_save}", exc_info=True)
                continue 

            try:
                logger.info(f"ModelMatcher 将使用输入文件: {input_for_matcher_path} (对应设备 {device_name})")
                matcher = ModelMatcher(csv_list_map=csv_list_map, input_json_path=str(input_for_matcher_path))
                device_matches = matcher.match() 
                logger.info(f"{device_name} 匹配完成。匹配到 {len(device_matches)} 项。")
                if matcher.unmatched_inputs:
                     logger.warning(f"{device_name} 有 {len(matcher.unmatched_inputs)} 个参数未匹配。")

                # --- 步骤 4.1: 代码选择 ---
                logger.info(f"开始为 {device_name} 进行代码选择...")
                selected_codes_for_device: Optional[Dict[str, Dict[str, Any]]] = None
                if not device_matches:
                    logger.warning(f"{device_name}: 模型匹配未产生任何结果，跳过代码选择。")
                    selected_codes_for_device = {}
                else:
                    try:
                        code_selector = CodeSelector(matched_models_dict=device_matches)
                        selected_codes_for_device = code_selector.select_codes()
                        logger.info(f"{device_name}: 代码选择完成。选择了 {len(selected_codes_for_device)} 个代码。")
                    except Exception as cs_err:
                        logger.error(f"{device_name}: 代码选择过程中发生错误: {cs_err}", exc_info=True)
                        selected_codes_for_device = {} 

                # --- 步骤 4.2: 代码生成 ---
                logger.info(f"开始为 {device_name} 生成最终代码...")
                final_code_str = f"产品型号生成失败（{device_name}）：无有效代码可选。"
                if not selected_codes_for_device:
                    logger.warning(f"{device_name}: 没有代码选择结果，无法生成最终代码。")
                else:
                    try:
                        code_generator = CodeGenerator()
                        logger.info(f"为 {device_name} 调用 CodeGenerator，可能需要用户输入...")
                        final_code_str = code_generator.generate_final_code(
                            csv_list_map=csv_list_map, 
                            selected_codes_data=selected_codes_for_device
                        )
                        logger.info(f"{device_name}: 最终代码生成成功。")
                    except Exception as cg_err:
                        logger.error(f"{device_name}: 代码生成过程中发生错误: {cg_err}", exc_info=True)
                        final_code_str = f"产品型号生成失败（{device_name}）：{cg_err}"
                
                print(f"\n--- {device_name} 的最终结果 ---")
                print(f"产品型号生成：{final_code_str}")
                print("=" * 70)
                current_device_result["产品型号"] = final_code_str

            except Exception as match_err: 
                logger.error(f"为 {device_name} 执行匹配、选择或生成代码时出错: {match_err}", exc_info=True)
                print(f"\n--- {device_name} 的处理失败 ---")
                print(f"错误信息：{match_err}")
                print("=" * 70)
                current_device_result["产品型号"] = f"处理失败: {match_err}"
            finally:
                # 清理为当前设备创建的临时文件
                if input_for_matcher_path and input_for_matcher_path.exists():
                    try:
                        input_for_matcher_path.unlink()
                        logger.info(f"已删除临时匹配文件: {input_for_matcher_path}")
                    except OSError as e_del_temp:
                        logger.warning(f"删除临时匹配文件 {input_for_matcher_path} 失败: {e_del_temp}")
            all_device_final_codes.append(current_device_result)
        
        logger.info(f"所有设备的选型、代码选择和生成流程完成 ({input_file_path.name})。")

        # --- 保存最终选型结果到文件 ---
        if all_device_final_codes:
            # output_file_path 是在 main 函数中基于 settings.OUTPUT_DIR 和输入文件名构造的
            # 例如: data/output/温变规格书_output.json
            final_output_data_to_save = {
                "输入文件": str(input_file_path.name),
                "选型结果": all_device_final_codes
            }
            try:
                # 确保 output_file_path 的父目录存在 (虽然 main 函数中已创建 settings.OUTPUT_DIR)
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file_path, 'w', encoding='utf-8') as f_out:
                    json.dump(final_output_data_to_save, f_out, ensure_ascii=False, indent=4)
                logger.info(f"最终选型结果已保存至: {output_file_path}")
                print(f"\n最终选型结果已保存至: {output_file_path}")
            except Exception as e_save_final:
                logger.error(f"保存最终选型结果到 {output_file_path} 失败: {e_save_final}", exc_info=True)
                print(f"\n错误：无法保存最终选型结果文件到 {output_file_path}")


    except KeyboardInterrupt:
        logger.warning(f"流程在选型匹配、代码选择或生成阶段被用户中止 ({input_file_path.name})。")
        return 
    except Exception as e:
        logger.exception(f"选型匹配、代码选择或生成过程中发生意外错误 ({input_file_path.name}): {e}")
        return 

    # --- 5. 删除中间文件 ---
    logger.info(f"处理完 {input_file_path.name}，开始删除其相关的中间 JSON 文件...")
    files_to_delete_main = []
    if path_to_verified_extracted_data_for_csv: # 这是 _extracted_verified.json
        files_to_delete_main.append(path_to_verified_extracted_data_for_csv)

    for file_to_del_path in files_to_delete_main:
        if file_to_del_path and file_to_del_path.exists(): # Added check for file_to_del_path not being None
            try:
                file_to_del_path.unlink()
                logger.info(f"已删除中间文件: {file_to_del_path}")
            except OSError as e_del:
                logger.warning(f"删除中间文件 {file_to_del_path} 失败: {e_del}")
        elif file_to_del_path: # If path was provided but file doesn't exist
            logger.info(f"中间文件未找到，无需删除: {file_to_del_path}")
            
    logger.info(f"中间文件删除操作完成 ({input_file_path.name})。")
    logger.info(f"===== 订单文件处理完成: {input_file_path.name} =====")


# --- 主函数 ---
def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description="传感器智能选型系统")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-file", type=str, help="要处理的单个输入文件路径。")
    group.add_argument("--input-dir", type=str, help="包含订单文件的输入目录路径。")
    # output_dir 参数被移除，将使用 settings.OUTPUT_DIR
    # parser.add_argument("output_dir", type=str, help="存放最终选型结果 JSON 文件的输出目录路径") 
    args = parser.parse_args()

    # 输出目录固定为 settings.OUTPUT_DIR
    output_directory = settings.OUTPUT_DIR
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录已确认/创建: {output_directory}")
    except Exception as e:
        logger.critical(f"无法创建或访问固定输出目录: {output_directory} - {e}")
        print(f"错误: 无法创建或访问固定输出目录: {output_directory} - {e}")
        sys.exit(1)

    # 检查并构建知识库
    logger.info("检查知识库状态...")
    if not check_and_build_kb():
        logger.critical("知识库检查或构建失败，无法继续。")
        print("错误：知识库准备失败，请检查日志。")
        sys.exit(1)
    logger.info("知识库已准备就绪。")

    input_files_processed = 0
    input_files_failed = 0
    files_to_process: List[Path] = []

    if args.input_file:
        logger.info(f"指定了单个输入文件: {args.input_file}")
        single_input_file = Path(args.input_file)
        if not single_input_file.is_file():
            logger.critical(f"指定的输入文件未找到或不是一个文件: {single_input_file}")
            print(f"错误: 指定的输入文件未找到或不是一个文件: {single_input_file}")
            sys.exit(1)
        files_to_process.append(single_input_file)
    elif args.input_dir:
        logger.info(f"指定了输入目录: {args.input_dir}")
        input_directory = Path(args.input_dir)
        if not input_directory.is_dir():
            logger.critical(f"输入目录未找到或不是一个目录: {input_directory}")
            print(f"错误: 输入目录未找到或不是一个目录: {input_directory}")
            sys.exit(1)
        
        logger.info(f"开始扫描输入目录: {input_directory} 中的所有文件...")
        files_to_process = [f for f in input_directory.iterdir() if f.is_file()]
        if not files_to_process:
            logger.warning(f"在目录 {input_directory} 中没有找到任何文件。")
            print(f"提示: 在目录 {input_directory} 中没有找到任何文件。")
            sys.exit(0)
        logger.info(f"找到 {len(files_to_process)} 个文件，准备处理...")

    for input_file_path in files_to_process:
        logger.info(f"--- 开始处理输入文件: {input_file_path.name} ---")
        # 输出文件名保持不变，但路径使用固定的 output_directory
        output_file_name = f"{input_file_path.stem}_output.json" # 之前 process_order_file 不使用这个 output_file_path 来写文件
                                                                # 而是打印最终型号。如果需要保存匹配结果，则需要调整 process_order_file
                                                                # 当前假设 process_order_file 的第二个参数 output_file_path 仅用于日志或未来扩展
        # 实际上，process_order_file 并没有使用 output_file_path 参数来写入任何文件。
        # 它的输出是打印到控制台的最终产品型号。
        # 中间文件（如 _extracted_verified.json）会保存到 settings.OUTPUT_DIR。
        # 如果用户期望每个输入文件都有一个最终的 JSON 输出（例如包含型号的结果），
        # 那么 process_order_file 需要修改以保存这样的文件。
        # 目前，我将保持 output_file_path 的构造，但提醒 process_order_file 当前不使用它写最终结果。
        dummy_output_path_for_logging = output_directory / output_file_name


        try:
            process_order_file(input_file_path, dummy_output_path_for_logging) # 第二个参数当前主要用于日志或结构占位
            input_files_processed += 1
        except KeyboardInterrupt:
            logger.warning(f"用户中止了文件 {input_file_path.name} 的处理流程。")
            print(f"\n文件 {input_file_path.name} 的处理流程已中止。")
            input_files_failed +=1
            if len(files_to_process) > 1: # 仅当处理多个文件时询问
                user_choice = input("是否中止处理所有剩余文件? (yes/no): ").strip().lower()
                if user_choice == 'yes':
                    logger.warning("用户选择中止整个批处理流程。")
                    print("整个批处理流程已中止。")
                    sys.exit(1)
                else:
                    logger.info("继续处理下一个文件（如果还有）。")
            else: # 如果是单个文件，中止就直接退出了
                sys.exit(1)
        except Exception as e:
            logger.exception(f"处理文件 {input_file_path.name} 过程中发生未捕获的严重错误。")
            print(f"\n处理文件 {input_file_path.name} 过程中发生严重错误，请检查日志。")
            input_files_failed += 1
            # 可以选择是否因单个文件失败而中止，当前是继续

    logger.info(f"--- 所有文件处理完毕 ---")
    logger.info(f"成功处理文件数: {input_files_processed}")
    logger.info(f"处理失败文件数: {input_files_failed}")

    if input_files_processed == 0 and input_files_failed == 0 and not files_to_process:
        # 这个条件可能需要调整，因为 files_to_process 在单文件模式下也会有内容
        if args.input_dir: # 仅当是目录模式且未找到文件时显示此消息
             logger.info(f"输入目录 {args.input_dir} 中没有找到符合条件的文件进行处理。")

if __name__ == "__main__":
    main()
