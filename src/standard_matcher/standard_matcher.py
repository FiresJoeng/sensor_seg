# -*- coding: utf-8 -*-
"""
模块：标准匹配器 (Standard Matcher)
功能：整合了从获取CSV列表、模型匹配、代码选择到最终代码生成的完整流程。
"""

import json
import logging
import sys
import re
import time
import argparse
import os
import shutil
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Set

# 确保项目根目录在 sys.path 中以便导入 config, llm 和 json_processor
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import settings
    from src.standard_matcher.llm import call_llm_for_match
    # 导入 AnalysisJsonProcessor 用于文件拆分
    from src.standard_matcher.json_processor import AnalysisJsonProcessor
    # 导入拆分后的类
    from src.standard_matcher.fetch_csvlist import FetchCsvlist
    from src.standard_matcher.model_matcher import ModelMatcher
    from src.standard_matcher.code_selector import CodeSelector
    from src.standard_matcher.code_generator import CodeGenerator, SKIPPABLE_MODELS # 导入类和变量
except ImportError as e:
    logging.getLogger(__name__).critical(
        f"错误：在 standard_matcher.py 中导入模块失败 - {e}。"
        f"请检查项目结构和 PYTHONPATH。\n"
        f"项目根目录尝试设置为: {project_root}", exc_info=True)
    raise

# --- 全局配置 ---
logger = logging.getLogger(__name__)

# SKIPPABLE_MODELS 现在从 code_generator.py 导入

# --- 文件路径定义 ---
DEFAULT_INDEX_JSON_PATH = project_root / "libs" / "standard" / "index.json"
INDEX_JSON_PATH = Path(
    getattr(settings, 'INDEX_JSON_PATH', DEFAULT_INDEX_JSON_PATH))

# TEMP_OUTPUT_DIR 将用于存放由 json_processor.py 生成的单个参数文件
TEMP_OUTPUT_DIR = project_root / "data" / "output" / "temp"


# ==============================================================================
# Main Execution Logic
# ==============================================================================

def execute_standard_matching(main_input_json_path: Path) -> Optional[Path]:
    """
    执行标准的匹配、选择和代码生成流程。

    Args:
        main_input_json_path: 主输入 JSON 文件的路径 (通常是 _standardized_all.json)。

    Returns:
        Optional[Path]: 成功时返回最终结果文件的 Path 对象，失败或无结果时返回 None。
    """
    if not main_input_json_path.is_file():
        logger.error(f"错误：指定的主输入文件不存在: {main_input_json_path}")
        return None

    logger.info(f"开始执行 Standard Matcher 完整流程，输入文件: {main_input_json_path.name}")

    # --- 步骤 0: 拆分主输入文件到 temp 目录 ---
    logger.info(f"\n--- 步骤 0: 拆分主输入文件 '{main_input_json_path.name}' 到 temp 目录 ---")
    try:
        processor = AnalysisJsonProcessor(analysis_json_path=main_input_json_path)
        extracted_data = processor.extract_tag_and_common_params()

        if not extracted_data:
            logger.error(f"从主输入文件 '{main_input_json_path.name}' 未提取到任何设备数据，无法继续。")
            return None

        # 确保 temp 目录存在
        TEMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"确保 temp 目录存在: {TEMP_OUTPUT_DIR}")

        # 清空 temp 目录中的旧 .json 文件
        logger.info(f"正在清空 temp 目录中的旧 .json 文件...")
        cleared_count = 0
        for old_file in TEMP_OUTPUT_DIR.glob("*.json"):
            try:
                old_file.unlink()
                cleared_count += 1
            except OSError as e_clear:
                logger.warning(f"无法删除旧文件 '{old_file}': {e_clear}")
        logger.info(f"已清空 {cleared_count} 个旧 .json 文件。")


        # 写入拆分后的文件
        split_files_count = 0
        for item in extracted_data:
            tag_numbers = item.get("位号", [])
            common_params = item.get("共用参数", {})

            if not tag_numbers or not common_params:
                logger.warning(f"跳过无效的数据项（缺少位号或共用参数）：{item}")
                continue

            # 生成文件名 (用下划线连接位号，并替换斜杠)
            # 替换文件名中的非法字符，例如 '/'
            safe_tag_str = "_".join(tag_numbers).replace("/", "_").replace("\\", "_")
            # 进一步清理，移除其他可能不安全的字符 (只保留字母、数字、下划线、连字符)
            safe_filename_base = re.sub(r'[^\w\-]+', '', safe_tag_str)
            if not safe_filename_base: # 如果清理后为空，给个默认名
                safe_filename_base = f"unknown_tag_{split_files_count + 1}"
            temp_filename = f"{safe_filename_base}.json"
            temp_file_path = TEMP_OUTPUT_DIR / temp_filename

            try:
                with open(temp_file_path, 'w', encoding='utf-8') as f_temp:
                    # 只写入共用参数部分
                    json.dump(common_params, f_temp, indent=4, ensure_ascii=False)
                logger.info(f"成功将位号 {tag_numbers} 的参数写入到: {temp_file_path}")
                split_files_count += 1
            except IOError as e_write:
                logger.error(f"无法写入临时文件 '{temp_file_path}': {e_write}")
                return None # 写入失败，中止

        if split_files_count == 0:
             logger.error("未能成功拆分任何文件到 temp 目录，无法继续。")
             return None

        logger.info(f"成功拆分主文件为 {split_files_count} 个文件到 {TEMP_OUTPUT_DIR}")

    except FileNotFoundError: # AnalysisJsonProcessor 初始化时可能抛出
         logger.error(f"初始化 AnalysisJsonProcessor 失败：文件未找到 {main_input_json_path}")
         return None
    except Exception as e_split:
        logger.error(f"拆分主输入文件时发生错误: {e_split}", exc_info=True)
        return None


    # --- 开始处理 temp 目录中的文件 ---
    index_json = INDEX_JSON_PATH # 索引文件路径是固定的
    logger.info(f"使用索引文件: {index_json}")

    json_files_in_temp = sorted(list(TEMP_OUTPUT_DIR.glob("*.json"))) # 按名称排序
    if not json_files_in_temp:
        logger.error(f"错误: temp 目录 '{TEMP_OUTPUT_DIR}' 中没有找到拆分后的 JSON 文件。")
        return None

    logger.info(f"开始处理 temp 目录中的 {len(json_files_in_temp)} 个 JSON 文件...")

    all_results = [] # 用于存储所有文件的处理结果
    any_file_processed_successfully = False # 跟踪是否有任何文件成功处理

    for i, temp_json_file_path in enumerate(json_files_in_temp):
        logger.info(f"\n{'='*20} 开始处理文件 {i+1}/{len(json_files_in_temp)}: {temp_json_file_path.name} {'='*20}")
        current_file_successful = True

        # --- 步骤 1: 获取 CSV 列表 ---
        print(f"\n--- 文件: {temp_json_file_path.name} - 步骤 1: 获取 CSV 列表 ---")
        fetcher = FetchCsvlist()
        logger.info(f"使用输入文件: {temp_json_file_path}")
        csv_list_map_result = fetcher.fetch_csv_lists(temp_json_file_path, index_json)

        if not csv_list_map_result:
            logger.error(f"文件 {temp_json_file_path.name}: 未能获取 CSV 列表，跳过此文件。")
            current_file_successful = False
            # 不立即将 all_successful 设为 False，允许其他文件继续处理
            # continue # 处理下一个文件 # 改为记录错误并继续，最后判断是否有成功项
        else:
            print(
                f"文件 {temp_json_file_path.name}: 获取到的 CSV 列表映射: {json.dumps(csv_list_map_result, indent=2, ensure_ascii=False)}")

        # --- 步骤 2: 模型匹配 ---
        if current_file_successful:
            print(f"\n--- 文件: {temp_json_file_path.name} - 步骤 2: 模型匹配 ---")
            model_matcher = ModelMatcher(
                csv_list_map=csv_list_map_result, input_json_path=str(temp_json_file_path))
            matched_models_result = model_matcher.match()

            if not matched_models_result:
                logger.warning(f"文件 {temp_json_file_path.name}: 模型匹配未产生任何结果。")
            else:
                print(
                    f"文件 {temp_json_file_path.name}: 模型匹配结果 (部分): {json.dumps(dict(list(matched_models_result.items())[:2]), indent=2, ensure_ascii=False)}...")
        else:
            matched_models_result = {} # 如果上一步失败，则无匹配结果

        # --- 步骤 3: 代码选择 ---
        selected_codes_result = {}
        if current_file_successful and matched_models_result:
            print(f"\n--- 文件: {temp_json_file_path.name} - 步骤 3: 代码选择 ---")
            code_selector = CodeSelector(matched_models_dict=matched_models_result)
            try:
                selected_codes_result = code_selector.select_codes()
                print(
                    f"文件 {temp_json_file_path.name}: 代码选择结果 (部分): {json.dumps(dict(list(selected_codes_result.items())[:2]), indent=2, ensure_ascii=False)}...")
            except (ValueError, RuntimeError) as e:
                logger.error(f"文件 {temp_json_file_path.name}: 代码选择过程中发生错误: {e}，跳过此文件。")
                current_file_successful = False
        elif not matched_models_result and current_file_successful:
             logger.warning(f"文件 {temp_json_file_path.name}: 没有模型匹配结果，跳过代码选择。")


        # --- 步骤 4: 代码生成 ---
        final_code_result = f"产品型号生成失败（文件: {temp_json_file_path.name}）：处理链早期步骤失败或无代码可选。"
        if current_file_successful and selected_codes_result:
            print(f"\n--- 文件: {temp_json_file_path.name} - 步骤 4: 代码生成 ---")
            code_generator = CodeGenerator()
            print(f"\n文件 {temp_json_file_path.name}: 代码生成过程中可能需要您输入整数值...")
            try:
                final_code_result = code_generator.generate_final_code(
                    csv_list_map=csv_list_map_result,
                    selected_codes_data=selected_codes_result
                )
            except Exception as e_gen:
                logger.error(f"文件 {temp_json_file_path.name}: 代码生成过程中发生错误: {e_gen}", exc_info=True)
                final_code_result = f"产品型号生成失败（文件: {temp_json_file_path.name}）：错误 - {e_gen}"
                current_file_successful = False
        elif not selected_codes_result and current_file_successful:
            logger.warning(f"文件 {temp_json_file_path.name}: 没有代码选择结果，跳过代码生成。")
            final_code_result = f"产品型号生成失败（文件: {temp_json_file_path.name}）：无代码可选。"


        # --- 单个文件最终结果 ---
        print(f"\n--- 文件: {temp_json_file_path.name} - 最终结果 ---")
        print(final_code_result)
        print(f"{'='*70}")

        # --- 提取位号并聚合结果 ---
        try:
            tag_number_str_from_filename = temp_json_file_path.stem.replace("_", "/")
            tag_numbers_list = [tag_number_str_from_filename]
        except Exception as e_tag:
            logger.error(f"从文件名 {temp_json_file_path.name} 提取位号时出错: {e_tag}")
            tag_numbers_list = [f"无法解析文件名: {temp_json_file_path.name}"]

        result_entry = {
            "位号": tag_numbers_list,
            "型号代码": final_code_result
        }
        all_results.append(result_entry)

        if current_file_successful:
            any_file_processed_successfully = True
        else:
            logger.error(f"文件 {temp_json_file_path.name} 处理失败。")

        if i < len(json_files_in_temp) - 1:
            logger.info(f"处理完文件 {temp_json_file_path.name}，等待 5 秒...")
            time.sleep(5)

    # --- 所有文件处理完毕，写入最终结果文件 ---
    if not any_file_processed_successfully and all_results: # 如果有结果但没有一个成功，说明都是错误信息
        logger.error("所有拆分出的文件均处理失败。最终结果文件将包含错误详情。")
    elif not all_results: # 如果根本没有结果（例如拆分后temp为空，或所有拆分项都跳过了）
        logger.error("未能生成任何结果。")
        return None

    logger.info("\n所有文件处理循环结束。准备写入最终结果文件...")

    input_file_stem = main_input_json_path.stem
    base_name_for_output = input_file_stem.replace('_analysis', '').replace('_standardized_all', '')
    output_filename = f"{base_name_for_output}_results.json"
    output_file_path = project_root / "data" / "output" / output_filename

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            json.dump(all_results, f_out, indent=4, ensure_ascii=False)
        logger.info(f"所有结果已成功写入到文件: {output_file_path}")
        # --- 清理临时文件 ---
        logger.info(f"正在清理临时目录: {TEMP_OUTPUT_DIR}")
        try:
            if TEMP_OUTPUT_DIR.is_dir():
                # 删除目录下的所有文件和子目录
                for item in TEMP_OUTPUT_DIR.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                logger.info(f"临时目录 '{TEMP_OUTPUT_DIR}' 清理完成。")
            else:
                logger.warning(f"临时目录 '{TEMP_OUTPUT_DIR}' 不存在，无需清理。")
        except Exception as e_cleanup:
            logger.error(f"清理临时目录 '{TEMP_OUTPUT_DIR}' 时出错: {e_cleanup}", exc_info=True)
        # --- 清理临时文件结束 ---
        return output_file_path
    except Exception as e_write:
        logger.error(f"将最终结果写入文件 {output_file_path} 时出错: {e_write}", exc_info=True)
        # --- 清理临时文件 (即使文件写入失败) ---
        logger.info(f"正在清理临时目录 (文件写入失败后): {TEMP_OUTPUT_DIR}")
        try:
            if TEMP_OUTPUT_DIR.is_dir():
                # 删除目录下的所有文件和子目录
                for item in TEMP_OUTPUT_DIR.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                logger.info(f"临时目录 '{TEMP_OUTPUT_DIR}' 清理完成 (文件写入失败后)。")
            else:
                logger.warning(f"临时目录 '{TEMP_OUTPUT_DIR}' 不存在，无需清理 (文件写入失败后)。")
        except Exception as e_cleanup:
            logger.error(f"清理临时目录 '{TEMP_OUTPUT_DIR}' 时出错 (文件写入失败后): {e_cleanup}", exc_info=True)
        # --- 清理临时文件结束 ---
        return None


if __name__ == "__main__":
    # --- 命令行参数解析 ---
    parser = argparse.ArgumentParser(description="Standard Matcher: 接收主输入文件，拆分，处理并生成型号代码。")
    parser.add_argument(
        "--input-file",
        required=True,
        help="主输入 JSON 文件的完整路径 (例如 'data/output/温变规格书_standardized_all.json')。"
    )
    args = parser.parse_args()
    input_file_path_cli = Path(args.input_file)

    final_output_file = execute_standard_matching(input_file_path_cli)

    if final_output_file:
        logger.info(f"\nStandard Matcher 完整流程执行完毕。最终输出: {final_output_file}")
        sys.exit(0)
    else:
        logger.error("\nStandard Matcher 流程执行失败或未生成输出文件。请检查日志。")
        sys.exit(1)