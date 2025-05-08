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
    from zhipuai import ZhipuAI

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


# --- 辅助函数：人工核对 ---
def prompt_for_manual_check(prompt_message: str) -> bool:
    """通用的人工确认提示函数"""
    while True:
        print(f"\n--- 人工核对 ---")
        print(prompt_message)
        response = input("完成后请按 Enter 继续，或输入 'abort' 中止整个流程: ").lower().strip()
        if response == '':
            logger.info("人工确认：继续处理。")
            return True # 继续
        elif response == 'abort':
            logger.error("人工确认：中止流程。")
            raise KeyboardInterrupt("用户中止流程") # 使用异常中断流程
        else:
            print("无效输入。请按 Enter 或输入 'abort'。")

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
        return None # 保存失败

    prompt = f"{prompt_prefix} 数据已保存至文件，请检查或修改:\n{file_path}"
    if not prompt_for_manual_check(prompt):
        # 如果用户选择 abort (abort 会抛异常)
        logger.error(f"{prompt_prefix} 数据核对被中止。")
        return None # 理论上不会执行到这里，因为 abort 会抛异常

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


# --- 主处理流程 ---
def process_order_file(input_file_path: Path, output_file_path: Path):
    """处理单个订单文件，执行完整的提取、标准化和匹配流程。"""
    logger.info(f"===== 开始处理订单文件: {input_file_path.name} =====")

    # --- 1. 初始化服务 ---
    try:
        logger.info("初始化服务...")
        info_extractor = InfoExtractor()
        search_service = SearchService() # Standardizer 会用到
        if not search_service.is_ready():
            logger.error("SearchService (用于标准化) 未就绪，无法继续处理。")
            return

        # 初始化 ZhipuAI 客户端
        api_key = settings.ZHIPUAI_API_KEY # 直接访问模块变量
        if not api_key:
            logger.error("配置错误：未能从 settings.py 或 .env 文件加载 ZHIPUAI_API_KEY。")
            raise ValueError("缺少 ZHIPUAI_API_KEY 配置")
        zhipuai_client = ZhipuAI(api_key=api_key)

        # 初始化 AccurateLLMStandardizer
        standardizer = AccurateLLMStandardizer(search_service=search_service, client=zhipuai_client)

        # FetchCsvlist 实例化移到获取 extracted_data 之后
        # index_json_path 定义移到 FetchCsvlist 使用处

        logger.info("核心服务初始化完成 (CSV列表映射将在数据提取后获取)。")

    except ValueError as ve:
        logger.error(ve)
        return
    except Exception as e:
        logger.exception(f"初始化服务时出错: {e}")
        return

    # --- 2. 信息提取 ---
    extracted_data: Optional[Dict[str, Any]] = None
    try:
        logger.info(f"开始从 {input_file_path} 提取信息...")
        # 假设 extract_parameters_from_pdf 返回包含设备列表和备注的字典
        extracted_data_raw = info_extractor.extract_parameters_from_pdf(input_file_path)
        if not extracted_data_raw:
            logger.error("信息提取失败或未返回任何数据。")
            return
        logger.info("信息提取成功。")

        # --- 检查点 1: 提取后核对 ---
        extracted_checkpoint_path = settings.OUTPUT_DIR / f"{input_file_path.stem}_1_extracted_raw.json"
        extracted_data = save_and_verify_json(extracted_data_raw, extracted_checkpoint_path, "提取的原始参数")
        if extracted_data is None:
            logger.error("提取的参数核对失败或被中止。")
            return # 如果核对失败或中止
        logger.info("提取参数核对完成。")

        # --- 获取 CSV 文件列表映射 (使用 FetchCsvlist) ---
        csv_list_map: Optional[Dict[str, List[str]]] = None
        try:
            logger.info("开始通过 FetchCsvlist 获取 CSV 文件列表映射...")
            fetcher = FetchCsvlist()
            index_json_path = project_root / "libs" / "standard" / "index.json"

            # FetchCsvlist 需要一个输入 JSON 文件路径，该文件内容应类似于 test.json
            # 我们使用核对后的 extracted_data，其路径是 extracted_checkpoint_path
            logger.info(f"FetchCsvlist 将使用输入文件: {extracted_checkpoint_path}")

            csv_list_map = fetcher.fetch_csv_lists(
                input_json_path=extracted_checkpoint_path, # 直接使用第一次核对后的文件
                index_json_path=index_json_path
            )
            
            if not csv_list_map:
                logger.error(f"未能通过 FetchCsvlist 获取标准库 CSV 文件列表映射，无法进行后续匹配。")
                return # 流程中止
            logger.info(f"通过 FetchCsvlist 成功获取 CSV 列表映射: {len(csv_list_map)} 个产品类型。")

        except Exception as e_fetch_csv:
            logger.exception(f"通过 FetchCsvlist 获取 CSV 列表映射时发生错误: {e_fetch_csv}")
            return # 流程中止


    except KeyboardInterrupt:
        logger.warning("流程在信息提取或CSV列表获取阶段被用户中止。")
        return
    except Exception as e:
        logger.exception(f"信息提取或CSV列表获取过程中发生意外错误: {e}")
        return

    # --- 3. 参数标准化 ---
    standardized_data: Optional[Dict[str, Any]] = None
    try:
        logger.info("开始参数标准化...")
        # 使用 AccurateLLMStandardizer 对 *核对后* 的提取数据进行标准化
        standardized_data_raw = standardizer.standardize(extracted_data) 
        if not standardized_data_raw:
            logger.error("参数标准化失败或未返回任何数据。")
            return
        logger.info("参数标准化成功。")

        # --- 检查点 2: 标准化后核对 ---
        standardized_checkpoint_path = settings.OUTPUT_DIR / f"{input_file_path.stem}_2_standardized.json"
        standardized_data = save_and_verify_json(standardized_data_raw, standardized_checkpoint_path, "标准化后的参数")
        if standardized_data is None:
            logger.error("标准化参数核对失败或被中止。")
            return 
        logger.info("标准化参数核对完成。")

    except KeyboardInterrupt:
        logger.warning("流程在参数标准化或核对阶段被用户中止。")
        return
    except Exception as e:
        logger.exception(f"参数标准化或核对过程中发生意外错误: {e}")
        return

    # --- 4. 数据准备与选型匹配 ---
    final_matching_results = {}
    try:
        logger.info("开始准备数据并进行选型匹配...")
        
        # standardized_data 和 csv_list_map 都应该已经准备好
        if standardized_data is None or csv_list_map is None:
            logger.error("标准化数据或CSV列表映射为空，无法进行匹配。")
            return

        device_list = standardized_data.get("设备列表")
        if not isinstance(device_list, list):
            logger.error("标准化数据格式错误：未找到'设备列表'或其不是列表。")
            if isinstance(standardized_data, dict) and "设备列表" not in standardized_data:
                 logger.warning("标准化数据顶层似乎是单个设备参数，尝试按单个设备处理。")
                 device_list = [standardized_data] 
            else:
                 logger.error("无法解析标准化数据以进行匹配。")
                 return

        logger.info(f"准备为 {len(device_list)} 个设备/参数组进行匹配...")
        all_devices_matches = [] 

        for i, device_params_raw in enumerate(device_list):
            device_name = device_params_raw.get("设备名称", f"设备_{i+1}") 
            logger.info(f"--- 开始匹配 {device_name} ---")

            if not isinstance(device_params_raw, dict):
                logger.error(f"{device_name} 的参数格式不是字典，跳过匹配。数据: {device_params_raw}")
                all_devices_matches.append({"设备名称": device_name, "匹配结果": None, "错误": "参数格式错误"})
                continue

            params_to_match = {k: v for k, v in device_params_raw.items() if isinstance(v, (str, int, float))}
            if not params_to_match:
                 logger.warning(f"{device_name} 没有找到可用于匹配的参数，跳过。")
                 all_devices_matches.append({"设备名称": device_name, "匹配结果": None, "错误": "无有效参数"})
                 continue

            logger.debug(f"为 {device_name} 准备的匹配参数: {params_to_match}")
            # ModelMatcher 将直接使用第二次核对后的文件 standardized_checkpoint_path
            # 前提是 standardized_data (即 standardized_checkpoint_path 的内容)
            # 在被包装成 device_list = [standardized_data] 后，
            # device_params_raw 就是 ModelMatcher 期望的单个设备的参数字典。
            # 并且 params_to_match 应该与 standardized_checkpoint_path 的内容一致（或其核心部分）。
            # 为了确保 ModelMatcher 读取的是完整的、未经 params_to_match 过滤的文件内容，
            # 我们直接将 standardized_checkpoint_path 传递给 ModelMatcher。
            
            # 检查 device_params_raw 是否就是 standardized_data (当只有一个设备时)
            # 如果是，并且 params_to_match 是从 device_params_raw 过滤的，
            # 那么直接用 standardized_checkpoint_path 是合理的，因为它包含了完整的未过滤数据。
            
            input_for_matcher_path = standardized_checkpoint_path
            logger.info(f"ModelMatcher 将使用输入文件: {input_for_matcher_path} (对应设备 {device_name})")

            try:
                # 实例化并执行匹配 (使用从 FetchCsvlist 获取的 csv_list_map)
                # 并且 ModelMatcher 直接读取 standardized_checkpoint_path
                matcher = ModelMatcher(csv_list_map=csv_list_map, input_json_path=str(input_for_matcher_path))
                device_matches = matcher.match() 
                logger.info(f"{device_name} 匹配完成。匹配到 {len(device_matches)} 项。")
                if matcher.unmatched_inputs:
                     logger.warning(f"{device_name} 有 {len(matcher.unmatched_inputs)} 个参数未匹配。")

                all_devices_matches.append({ # 这个列表可能不再直接用于最终输出了
                    "设备名称": device_name,
                    # "匹配结果": device_matches, # 可能不需要存储这个了
                    # "未匹配输入": matcher.unmatched_inputs
                })

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
                        selected_codes_for_device = {} # 出错则为空

                # --- 步骤 4.2: 代码生成 ---
                logger.info(f"开始为 {device_name} 生成最终代码...")
                final_code_str = f"产品型号生成失败（{device_name}）：无有效代码可选。"
                if not selected_codes_for_device:
                    logger.warning(f"{device_name}: 没有代码选择结果，无法生成最终代码。")
                else:
                    try:
                        code_generator = CodeGenerator()
                        # CodeGenerator 可能需要用户输入，确保日志或提示能显示
                        logger.info(f"为 {device_name} 调用 CodeGenerator，可能需要用户输入...")
                        final_code_str = code_generator.generate_final_code(
                            csv_list_map=csv_list_map, # 这个是从 FetchCsvlist 获取的
                            selected_codes_data=selected_codes_for_device
                        )
                        logger.info(f"{device_name}: 最终代码生成成功。")
                    except Exception as cg_err:
                        logger.error(f"{device_name}: 代码生成过程中发生错误: {cg_err}", exc_info=True)
                        final_code_str = f"产品型号生成失败（{device_name}）：{cg_err}"
                
                # --- 打印最终结果 (针对当前设备) ---
                print(f"\n--- {device_name} 的最终结果 ---")
                print(f"产品型号生成：{final_code_str}")
                print("=" * 70)


            except Exception as match_err: # ModelMatcher 或后续步骤的错误
                logger.error(f"为 {device_name} 执行匹配、选择或生成代码时出错: {match_err}", exc_info=True)
                print(f"\n--- {device_name} 的处理失败 ---")
                print(f"错误信息：{match_err}")
                print("=" * 70)
                # all_devices_matches.append({"设备名称": device_name, "匹配结果": None, "错误": str(match_err)}) # 如果还需要记录错误

        # final_matching_results = {"匹配详情": all_devices_matches} # 不再需要这个结构来保存文件
        logger.info("所有设备的选型、代码选择和生成流程完成。") # 针对当前输入文件的所有设备

    except KeyboardInterrupt:
        logger.warning("流程在选型匹配、代码选择或生成阶段被用户中止。")
        return # 中止当前文件的处理
    except Exception as e:
        logger.exception(f"选型匹配、代码选择或生成过程中发生意外错误: {e}")
        return # 中止当前文件的处理

    # --- 5. (原步骤6) 删除中间文件 ---
    # 这部分逻辑在单个文件处理成功后执行
    logger.info(f"处理完 {input_file_path.name}，开始删除其相关的中间 JSON 文件...")
    files_to_delete = [
        settings.OUTPUT_DIR / f"{input_file_path.stem}_1_extracted_raw.json",
        settings.OUTPUT_DIR / f"{input_file_path.stem}_2_standardized.json"
    ]
    for file_to_del_path in files_to_delete:
        if file_to_del_path.exists():
            try:
                file_to_del_path.unlink()
                logger.info(f"已删除中间文件: {file_to_del_path}")
            except OSError as e_del:
                logger.warning(f"删除中间文件 {file_to_del_path} 失败: {e_del}")
        else:
            logger.info(f"中间文件未找到，无需删除: {file_to_del_path}")
    logger.info("中间文件删除操作完成。")

    logger.info(f"===== 订单文件处理完成: {input_file_path.name} =====")


# --- 主函数 ---
def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description="传感器智能选型系统")
    parser.add_argument("input_dir", type=str, help="包含订单文件的输入目录路径")
    parser.add_argument("output_dir", type=str, help="存放最终选型结果 JSON 文件的输出目录路径")
    args = parser.parse_args()

    input_directory = Path(args.input_dir)
    output_directory = Path(args.output_dir)

    # 检查输入目录是否存在
    if not input_directory.is_dir():
        logger.critical(f"输入目录未找到或不是一个目录: {input_directory}")
        print(f"错误: 输入目录未找到或不是一个目录: {input_directory}")
        sys.exit(1)

    # 确保输出目录存在
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录已确认/创建: {output_directory}")
    except Exception as e:
        logger.critical(f"无法创建输出目录: {output_directory} - {e}")
        print(f"错误: 无法创建输出目录: {output_directory} - {e}")
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

    logger.info(f"开始扫描输入目录: {input_directory} 中的所有文件...")
    # 使用 iterdir() 并过滤出文件
    all_files_in_dir = [f for f in input_directory.iterdir() if f.is_file()]

    if not all_files_in_dir:
        logger.warning(f"在目录 {input_directory} 中没有找到任何文件。")
        print(f"提示: 在目录 {input_directory} 中没有找到任何文件。")
        sys.exit(0)

    logger.info(f"找到 {len(all_files_in_dir)} 个文件，准备处理...")

    for input_file_path in all_files_in_dir:
        logger.info(f"--- 开始处理输入文件: {input_file_path.name} ---")
        # 为每个输入文件确定输出文件路径
        # 保留原始扩展名，并在文件名后添加 _output，然后更改扩展名为 .json
        output_file_name = f"{input_file_path.stem}_output.json"
        output_file_path = output_directory / output_file_name

        try:
            # 假设 process_order_file 内部的 InfoExtractor 能够处理各种文件类型
            # 或者它有自己的逻辑来处理不支持的文件类型。
            # 当前的修改只关注 main.py 的输入和输出文件管理。
            process_order_file(input_file_path, output_file_path)
            input_files_processed += 1
        except KeyboardInterrupt:
            logger.warning(f"用户中止了文件 {input_file_path.name} 的处理流程。")
            print(f"\n文件 {input_file_path.name} 的处理流程已中止。")
            input_files_failed +=1
            user_choice = input("是否中止处理所有剩余文件? (yes/no): ").strip().lower()
            if user_choice == 'yes':
                logger.warning("用户选择中止整个批处理流程。")
                print("整个批处理流程已中止。")
                sys.exit(1)
            else:
                logger.info("继续处理下一个文件（如果还有）。")
        except Exception as e:
            logger.exception(f"处理文件 {input_file_path.name} 过程中发生未捕获的严重错误。")
            print(f"\n处理文件 {input_file_path.name} 过程中发生严重错误，请检查日志。")
            input_files_failed += 1
            # 可以选择是否因单个文件失败而中止，当前是继续

    logger.info(f"--- 所有文件处理完毕 ---")
    logger.info(f"成功处理文件数: {input_files_processed}")
    logger.info(f"处理失败文件数: {input_files_failed}")

    if input_files_processed == 0 and input_files_failed == 0 and not all_files_in_dir:
        logger.info(f"输入目录 {input_directory} 中没有找到符合条件的文件进行处理。")

if __name__ == "__main__":
    main()
