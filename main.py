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
    from src.standard_matcher.model_matcher import ModelMatcher
    # from src.standard_matcher.fetch_csvlist import get_csv_list_map # 不再使用此导入
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

# --- 新增辅助函数：从 index.json 加载 CSV 列表映射 ---
def load_csv_list_map_from_index(index_path: Path) -> Dict[str, List[str]]:
    """从 index.json 加载并构建 ModelMatcher 所需的 CSV 列表映射。"""
    csv_map = {}
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        if not isinstance(index_data, dict):
            logger.error(f"索引文件 {index_path} 格式错误：顶层不是字典。")
            return {}

        for product_type, keywords_dict in index_data.items():
            if not isinstance(keywords_dict, dict):
                logger.warning(f"索引文件 {index_path} 中产品类型 '{product_type}' 的值不是字典，跳过。")
                continue

            all_csv_paths_for_type = set() # 使用集合去重
            for keyword, csv_list in keywords_dict.items():
                if isinstance(csv_list, list):
                    # 将相对路径转换为相对于项目根目录的绝对路径字符串
                    for csv_path_str in csv_list:
                        # 假设 index.json 中的路径是相对于 libs/standard/ 的
                        # 或者更健壮的方式是假设它们相对于 index.json 所在的目录
                        # 这里我们先假设它们是相对于项目根目录的路径字符串
                        # full_path = project_root / csv_path_str # 如果路径已经是相对于根目录
                        # 如果路径是相对于 index.json 目录的
                        # full_path = index_path.parent / csv_path_str
                        # 为了简单起见，我们假设 index.json 中的路径已经是正确的相对或绝对路径字符串
                        # ModelMatcher 内部会处理路径
                        all_csv_paths_for_type.add(str(csv_path_str)) # 直接使用字符串路径
                else:
                    logger.warning(f"索引文件 {index_path} 中产品类型 '{product_type}' 的关键词 '{keyword}' 的值不是列表，跳过。")

            if all_csv_paths_for_type:
                csv_map[product_type] = sorted(list(all_csv_paths_for_type)) # 转换为排序列表

        logger.info(f"成功从 {index_path} 加载并构建了 {len(csv_map)} 个产品类型的 CSV 列表映射。")
        return csv_map

    except FileNotFoundError:
        logger.error(f"索引文件未找到: {index_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"解析索引 JSON 文件时出错: {index_path} - {e}")
        return {}
    except Exception as e:
        logger.error(f"加载或处理索引文件 {index_path} 时发生未知错误: {e}", exc_info=True)
        return {}


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

        # 获取 CSV 文件映射 (使用新的辅助函数)
        index_json_path = project_root / "libs" / "standard" / "index.json" # 定义 index.json 路径
        csv_list_map = load_csv_list_map_from_index(index_json_path)
        if not csv_list_map:
            logger.error(f"未能从 {index_json_path} 加载或构建标准库 CSV 文件列表映射，无法进行匹配。")
            return

        logger.info("所有服务初始化完成。")

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

    except KeyboardInterrupt:
        logger.warning("流程在信息提取或核对阶段被用户中止。")
        return
    except Exception as e:
        logger.exception(f"信息提取或核对过程中发生意外错误: {e}")
        return

    # --- 3. 参数标准化 ---
    standardized_data: Optional[Dict[str, Any]] = None
    try:
        logger.info("开始参数标准化...")
        # 使用 AccurateLLMStandardizer 对 *核对后* 的提取数据进行标准化
        # 假设 standardizer.standardize 返回与输入结构类似的字典，但值是标准化的
        standardized_data_raw = standardizer.standardize(extracted_data) # 使用核对后的 extracted_data
        if not standardized_data_raw:
            logger.error("参数标准化失败或未返回任何数据。")
            return
        logger.info("参数标准化成功。")

        # --- 检查点 2: 标准化后核对 ---
        standardized_checkpoint_path = settings.OUTPUT_DIR / f"{input_file_path.stem}_2_standardized.json"
        standardized_data = save_and_verify_json(standardized_data_raw, standardized_checkpoint_path, "标准化后的参数")
        if standardized_data is None:
            logger.error("标准化参数核对失败或被中止。")
            return # 如果核对失败或中止
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

        # 从 *核对后* 的标准化数据中提取设备列表
        # 注意：这里的结构取决于 standardizer.standardize 和 save_and_verify_json 返回的结构
        # 假设它仍然包含 '设备列表' 键
        device_list = standardized_data.get("设备列表")
        if not isinstance(device_list, list):
            logger.error("标准化数据格式错误：未找到'设备列表'或其不是列表。")
            # 尝试直接处理整个 standardized_data 字典？
            # 或者报错退出
            # 临时方案：如果顶层是字典且不含'设备列表'，尝试将其视为单个设备处理
            if isinstance(standardized_data, dict) and "设备列表" not in standardized_data:
                 logger.warning("标准化数据顶层似乎是单个设备参数，尝试按单个设备处理。")
                 device_list = [standardized_data] # 包装成列表
            else:
                 logger.error("无法解析标准化数据以进行匹配。")
                 return

        logger.info(f"准备为 {len(device_list)} 个设备/参数组进行匹配...")

        all_devices_matches = [] # 存储每个设备的匹配结果

        for i, device_params_raw in enumerate(device_list):
            device_name = device_params_raw.get("设备名称", f"设备_{i+1}") # 获取或生成设备名
            logger.info(f"--- 开始匹配 {device_name} ---")

            # 准备 ModelMatcher 的输入：需要一个扁平的参数字典 {param_key: param_value}
            # 我们需要从 device_params_raw 中提取出适合匹配的键值对
            # 假设 device_params_raw 本身就是或包含一个参数字典
            # TODO: 确定 device_params_raw 的确切结构并提取参数字典
            # 临时假设：device_params_raw 就是参数字典
            if not isinstance(device_params_raw, dict):
                logger.error(f"{device_name} 的参数格式不是字典，跳过匹配。数据: {device_params_raw}")
                all_devices_matches.append({"设备名称": device_name, "匹配结果": None, "错误": "参数格式错误"})
                continue

            # 过滤掉非参数项（如果需要）
            params_to_match = {k: v for k, v in device_params_raw.items() if isinstance(v, (str, int, float))} # 简单过滤
            if not params_to_match:
                 logger.warning(f"{device_name} 没有找到可用于匹配的参数，跳过。")
                 all_devices_matches.append({"设备名称": device_name, "匹配结果": None, "错误": "无有效参数"})
                 continue

            logger.debug(f"为 {device_name} 准备的匹配参数: {params_to_match}")

            # 为当前设备创建临时 JSON 输入文件 (或者修改 ModelMatcher)
            temp_input_path = settings.OUTPUT_DIR / f"{input_file_path.stem}_temp_match_input_{i}.json"
            try:
                with open(temp_input_path, 'w', encoding='utf-8') as f_temp:
                    json.dump(params_to_match, f_temp, ensure_ascii=False, indent=4)
                logger.debug(f"为 {device_name} 创建临时匹配输入文件: {temp_input_path}")

                # 实例化并执行匹配
                matcher = ModelMatcher(csv_list_map=csv_list_map, input_json_path=str(temp_input_path))
                device_matches = matcher.match() # 返回 {"'key': 'value'": [rows...]}
                logger.info(f"{device_name} 匹配完成。匹配到 {len(device_matches)} 项。")
                if matcher.unmatched_inputs:
                     logger.warning(f"{device_name} 有 {len(matcher.unmatched_inputs)} 个参数未匹配。")

                all_devices_matches.append({
                    "设备名称": device_name,
                    "匹配结果": device_matches,
                    "未匹配输入": matcher.unmatched_inputs
                })

            except Exception as match_err:
                logger.error(f"为 {device_name} 执行匹配时出错: {match_err}", exc_info=True)
                all_devices_matches.append({"设备名称": device_name, "匹配结果": None, "错误": str(match_err)})
            finally:
                # 清理临时文件
                if temp_input_path.exists():
                    try:
                        temp_input_path.unlink()
                        logger.debug(f"已删除临时文件: {temp_input_path}")
                    except OSError as e:
                        logger.warning(f"删除临时文件 {temp_input_path} 失败: {e}")

        # 汇总所有设备的匹配结果
        final_matching_results = {"匹配详情": all_devices_matches}
        logger.info("所有设备的选型匹配流程完成。")

    except KeyboardInterrupt:
        logger.warning("流程在选型匹配阶段被用户中止。")
        return
    except Exception as e:
        logger.exception(f"选型匹配过程中发生意外错误: {e}")
        return

    # --- 5. 保存最终结果 ---
    try:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            json.dump(final_matching_results, f_out, ensure_ascii=False, indent=4)
        logger.info(f"最终选型匹配结果已保存至: {output_file_path}")
        print(f"\n--- 处理成功 ---")
        print(f"最终选型匹配结果已保存至文件:")
        print(output_file_path)

    except Exception as e:
        logger.error(f"保存最终结果到 {output_file_path} 时出错: {e}", exc_info=True)
        print(f"\n错误：无法保存最终结果文件。")

    logger.info(f"===== 订单文件处理完成: {input_file_path.name} =====")


# --- 主函数 ---
def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description="传感器智能选型系统")
    parser.add_argument("input_file", type=str, help="输入的订单文件路径 (例如 PDF)")
    parser.add_argument("output_file", type=str, help="输出的最终选型结果 JSON 文件路径")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    # 检查输入文件是否存在
    if not input_path.is_file():
        logger.critical(f"输入文件未找到: {input_path}")
        print(f"错误: 输入文件未找到: {input_path}")
        sys.exit(1)

    # 确保输出目录存在
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.critical(f"无法创建输出目录: {output_path.parent} - {e}")
        print(f"错误: 无法创建输出目录: {output_path.parent} - {e}")
        sys.exit(1)

    # 检查并构建知识库
    logger.info("检查知识库状态...")
    if not check_and_build_kb():
        logger.critical("知识库检查或构建失败，无法继续。")
        print("错误：知识库准备失败，请检查日志。")
        sys.exit(1)
    logger.info("知识库已准备就绪。")

    # 执行核心处理流程
    try:
        process_order_file(input_path, output_path)
    except KeyboardInterrupt:
        logger.warning("用户中止了处理流程。")
        print("\n处理流程已中止。")
        sys.exit(1)
    except Exception as e:
        logger.exception("处理过程中发生未捕获的严重错误。")
        print("\n处理过程中发生严重错误，请检查日志。")
        sys.exit(1)

if __name__ == "__main__":
    main()
