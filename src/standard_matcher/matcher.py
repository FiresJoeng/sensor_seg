# new_sensor_project/src/standard_matcher/matcher.py
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import sys

# --- Module Imports ---
# Ensure project root is in sys.path for config import
project_root = Path(__file__).resolve().parent.parent.parent # Should be new_sensor_project
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    # print(f"DEBUG: Added {project_root} to sys.path in matcher.py")

try:
    from config import settings
    # Import utils from the same directory using relative import
    from src.standard_matcher.utils import calculate_string_similarity, sort_results_by_csv_order
except ImportError as e:
    print(f"ERROR in matcher.py: Failed to import modules - {e}. Check project structure and PYTHONPATH.", file=sys.stderr)
    raise

# 获取日志记录器实例
logger = logging.getLogger(__name__)

# --- 内部辅助函数 ---

def _find_main_model(standardized_params: Dict[str, str], index_data: Dict[str, List[str]]) -> Optional[str]:
    """
    根据标准化参数确定产品主型号。
    优先精确匹配 '温度变送器' 或 '变送器' 键，然后尝试模糊匹配。

    Args:
        standardized_params: 标准化后的参数字典 {标准参数名: 标准参数值}。
        index_data: 从 index.json 加载的数据 {主型号: [csv_file1, ...]}。

    Returns:
        Optional[str]: 匹配到的主型号名称，如果找不到则返回 None。
    """
    logger.debug("开始确定产品主型号...")
    available_models = list(index_data.keys())
    target_keys = ["温度变送器", "变送器"] # 可能表示主型号的键
    main_model_value = None
    match_method = "未找到"

    # 1. 尝试精确匹配标准化参数中的主型号键
    for key in target_keys:
        if key in standardized_params:
            value = standardized_params[key]
            if value in index_data:
                main_model_value = value
                match_method = f"精确匹配 (键: '{key}', 值: '{value}')"
                logger.info(f"确定主型号: {main_model_value} ({match_method})")
                return main_model_value
            else:
                logger.debug(f"键 '{key}' 的值 '{value}' 不在 index.json 的主型号列表中。")

    # 2. 如果精确匹配失败，尝试模糊匹配
    logger.info("主型号精确匹配失败，尝试模糊匹配...")
    best_match_key = None
    best_match_model = None
    best_score = 0.0

    # 遍历输入参数，寻找与目标键和可用型号最相似的组合
    for input_key, input_value in standardized_params.items():
        if not input_key or not input_value: continue

        # 计算输入键与目标键 ("温度变送器", "变送器") 的最大相似度
        key_score = 0.0
        for target_key in target_keys:
            key_score = max(key_score, calculate_string_similarity(target_key, input_key))

        # 检查输入值与 index.json 中的主型号的相似度
        for model in available_models:
            model_score = calculate_string_similarity(input_value, model)

            # 组合得分 (可以调整权重)
            combined_score = (key_score * 0.4) + (model_score * 0.6)

            if combined_score > best_score:
                best_score = combined_score
                best_match_key = input_key
                best_match_model = model

    # 检查最佳模糊匹配得分是否达到阈值
    threshold = settings.MAIN_MODEL_SIMILARITY_THRESHOLD
    if best_score >= threshold:
        main_model_value = best_match_model
        match_method = f"模糊匹配 (最佳匹配键: '{best_match_key}', 值: '{standardized_params.get(best_match_key)}', 匹配型号: '{best_match_model}', 得分: {best_score:.4f})"
        logger.info(f"确定主型号: {main_model_value} ({match_method})")
        return main_model_value
    else:
        logger.error(f"无法确定主型号。最佳模糊匹配得分 {best_score:.4f} 低于阈值 {threshold}。")
        logger.error(f"(最佳模糊匹配详情: 键='{best_match_key}', 值='{standardized_params.get(best_match_key)}', 型号='{best_match_model}')")
        return None


def _load_and_combine_csvs(csv_files: List[Path]) -> Optional[pd.DataFrame]:
    """加载并合并指定的 CSV 文件列表到一个 DataFrame。"""
    if not csv_files:
        logger.error("没有提供要加载的 CSV 文件列表。")
        return None

    dfs = []
    logger.debug(f"开始加载和合并 {len(csv_files)} 个 CSV 文件...")
    for file_path in csv_files:
        try:
            if not file_path.is_file():
                 logger.warning(f"标准库 CSV 文件未找到，跳过: {file_path}")
                 continue
            df = pd.read_csv(file_path, dtype=str) # 读取所有列为字符串，避免类型推断问题
            # 可以添加列名验证等
            dfs.append(df)
            logger.debug(f"已加载 CSV: {file_path.name} ({len(df)} 行)")
        except Exception as e:
            logger.error(f"加载 CSV 文件 '{file_path.name}' 时出错: {e}", exc_info=True)
            # 根据需要决定是否继续或中止
            continue # 跳过错误的文件

    if not dfs:
        logger.error("未能成功加载任何指定的 CSV 文件。")
        return None

    try:
        combined_df = pd.concat(dfs, ignore_index=True)
        # 填充 NaN 值为空字符串，以便进行字符串比较
        combined_df.fillna('', inplace=True)
        logger.info(f"成功合并 {len(dfs)} 个 CSV 文件，总计 {len(combined_df)} 行。")
        return combined_df
    except Exception as e:
        logger.error(f"合并 DataFrame 时出错: {e}", exc_info=True)
        return None


def _find_candidate_rows(standard_param_name: str, combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    在合并的 DataFrame 中查找与标准参数名匹配的候选行。
    使用精确匹配，然后是模糊匹配。

    Args:
        standard_param_name: 标准参数名称 (例如 '输出信号')。
        combined_df: 合并后的标准库 DataFrame。

    Returns:
        pd.DataFrame: 包含匹配行的 DataFrame，如果找不到则为空 DataFrame。
    """
    logger.debug(f"为参数 '{standard_param_name}' 查找候选行...")
    # 确保 'model' 列存在
    if 'model' not in combined_df.columns:
        logger.error("合并的 DataFrame 中缺少 'model' 列。无法查找候选行。")
        return pd.DataFrame()

    # 1. 精确匹配 'model' 列
    exact_matches = combined_df[combined_df['model'] == standard_param_name]
    if not exact_matches.empty:
        logger.debug(f"找到 {len(exact_matches)} 个精确匹配行。")
        return exact_matches

    # 2. 模糊匹配 'model' 列
    logger.debug("未找到精确匹配，尝试模糊匹配 'model' 列...")
    fuzzy_matches_models = []
    threshold = settings.FUZZY_MATCH_THRESHOLD
    # 遍历 DataFrame 中唯一的 'model' 值进行比较
    for model_in_df in combined_df['model'].unique():
        similarity = calculate_string_similarity(standard_param_name, model_in_df)
        if similarity >= threshold:
            fuzzy_matches_models.append(model_in_df)
            logger.debug(f"  - 模糊匹配到 model: '{model_in_df}' (相似度: {similarity:.4f})")

    if fuzzy_matches_models:
        fuzzy_matches_df = combined_df[combined_df['model'].isin(fuzzy_matches_models)]
        logger.debug(f"找到 {len(fuzzy_matches_df)} 个模糊匹配行 (基于 model 列)。")
        return fuzzy_matches_df
    else:
        logger.warning(f"参数 '{standard_param_name}' 在标准库的 'model' 列中未找到精确或模糊匹配项。")
        return pd.DataFrame() # 返回空 DataFrame


def _select_best_match(standard_param_value: str, candidate_rows: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    从候选行中选择与标准参数值最匹配的一行。
    比较 'description', 'code', 'model' 字段。

    Args:
        standard_param_value: 标准参数的值 (例如 '4~20mA DC')。
        candidate_rows: 包含候选匹配行的 DataFrame。

    Returns:
        Optional[Dict[str, Any]]: 最佳匹配行的字典表示，如果找不到则返回 None。
    """
    if candidate_rows.empty:
        logger.debug("没有候选行可供选择最佳匹配。")
        return None

    best_item_dict: Optional[Dict[str, Any]] = None
    best_score = -1.0
    match_field = None # 记录哪个字段产生了最佳匹配

    logger.debug(f"为值 '{standard_param_value}' 从 {len(candidate_rows)} 个候选中选择最佳匹配...")

    # 遍历候选行的索引和数据
    for index, row in candidate_rows.iterrows():
        # 尝试与 'description', 'code', 'model' 三个字段进行模糊匹配
        for field in ('description', 'code', 'model'):
            # 确保字段存在于行中
            if field in row:
                text_in_df = str(row[field]) # 转换为字符串以防万一
                score = calculate_string_similarity(standard_param_value, text_in_df)
                # logger.debug(f"  - 比较 '{standard_param_value}' vs '{field}':'{text_in_df}' -> 得分: {score:.4f}")

                # 如果找到更高的分数，更新最佳匹配
                if score > best_score:
                    best_score = score
                    best_item_dict = row.to_dict() # 将最佳行转换为字典
                    match_field = field
                    logger.debug(f"  * 新的最佳匹配: 行索引 {index}, 字段 '{field}', 得分 {score:.4f}")

            # else:
            #     logger.warning(f"候选行索引 {index} 缺少字段 '{field}'。")

    # 检查最佳得分是否达到阈值 (可选，但推荐)
    threshold = settings.FUZZY_MATCH_THRESHOLD
    if best_score >= threshold and best_item_dict is not None:
        logger.debug(f"选择最佳匹配完成。最佳得分 {best_score:.4f} (来自字段 '{match_field}') >= 阈值 {threshold}。")
        return best_item_dict
    elif best_item_dict is not None:
         logger.warning(f"选择了最佳匹配，但得分 {best_score:.4f} 低于阈值 {threshold} (来自字段 '{match_field}')。匹配可能不准确。")
         return best_item_dict # 仍然返回得分最高的，但发出警告
    else:
        logger.warning(f"未能为值 '{standard_param_value}' 在候选行中找到足够相似的匹配项 (最高得分: {best_score:.4f})。")
        return None


# --- 主生成函数 ---

def generate_product_code(standardized_params: Dict[str, str]) -> Optional[str]:
    """
    根据标准化参数生成最终的产品型号代码。

    Args:
        standardized_params: 标准化后的参数字典 {标准参数名: 标准参数值}。

    Returns:
        Optional[str]: 生成的完整产品型号代码字符串，如果过程中出错则返回 None。
    """
    logger.info("--- 开始标准匹配和代码生成 ---")
    if not standardized_params:
        logger.error("输入参数为空，无法生成代码。")
        return None

    # 1. 加载标准库索引文件
    index_path = settings.STANDARD_INDEX_JSON
    if not index_path.is_file():
        logger.error(f"标准库索引文件未找到: {index_path}")
        return None
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        logger.debug(f"成功加载标准库索引: {index_path.name}")
    except Exception as e:
        logger.error(f"加载或解析标准库索引 '{index_path.name}' 时出错: {e}", exc_info=True)
        return None

    # 2. 确定主型号
    main_model = _find_main_model(standardized_params, index_data)
    if main_model is None:
        # _find_main_model 内部已记录错误
        return None

    # 3. 获取关联的 CSV 文件路径
    relative_csv_files = index_data.get(main_model)
    if not relative_csv_files:
        logger.error(f"索引文件中未找到主型号 '{main_model}' 关联的 CSV 文件列表。")
        return None

    # 将相对路径转换为绝对路径 (相对于 index.json 所在的目录)
    base_dir = index_path.parent
    absolute_csv_files = [base_dir / Path(f) for f in relative_csv_files]
    logger.debug(f"找到与主型号 '{main_model}' 关联的 CSV 文件: {[p.name for p in absolute_csv_files]}")

    # 4. 加载并合并 CSV 数据
    combined_df = _load_and_combine_csvs(absolute_csv_files)
    if combined_df is None or combined_df.empty:
        logger.error("加载或合并标准库 CSV 文件失败。")
        return None

    # 5. 匹配每个标准化参数并选择最佳代码
    final_matches: Dict[str, Optional[Dict[str, Any]]] = {}
    logger.info("开始匹配每个标准化参数到标准库...")
    for std_param_name, std_param_value in standardized_params.items():
        # 跳过可能用于确定主型号的参数（避免重复匹配）
        # TODO: 需要更可靠的方法来识别哪些是主型号参数，可能需要配置
        if std_param_name in ["温度变送器", "变送器"] and std_param_value == main_model:
             logger.debug(f"跳过主型号参数 '{std_param_name}' 的匹配。")
             continue

        logger.debug(f"--- 匹配参数: '{std_param_name}' = '{std_param_value}' ---")
        # a. 查找候选行 (基于标准参数名)
        candidate_rows_df = _find_candidate_rows(std_param_name, combined_df)

        # b. 从候选行中选择最佳匹配 (基于标准参数值)
        best_match_row_dict = _select_best_match(std_param_value, candidate_rows_df)

        # c. 存储最佳匹配结果 (可能是 None)
        final_matches[std_param_name] = best_match_row_dict
        if best_match_row_dict:
            logger.debug(f"参数 '{std_param_name}' 的最佳匹配代码: '{best_match_row_dict.get('code', 'N/A')}'")
        else:
            logger.warning(f"参数 '{std_param_name}' 未能找到合适的标准代码。")

    # 6. 根据参考 CSV 顺序排序结果
    # 使用列表中的第一个 CSV 作为排序参考
    reference_csv = absolute_csv_files[0] if absolute_csv_files else None
    if reference_csv:
        logger.info(f"使用 '{reference_csv.name}' 作为参考对结果进行排序...")
        sorted_matches = sort_results_by_csv_order(final_matches, reference_csv)
    else:
        logger.warning("没有找到参考 CSV 文件进行排序，将使用原始匹配顺序。")
        sorted_matches = final_matches

    # 7. 拼接最终代码
    product_code_parts: List[str] = []
    logger.info("开始拼接最终产品代码...")
    for std_param_name, match_info in sorted_matches.items():
        if match_info and 'code' in match_info:
            code_part = match_info['code']
            # 处理空的 code 值 (例如 NaN 转换后)
            if pd.isna(code_part) or code_part == '':
                 logger.warning(f"参数 '{std_param_name}' 匹配到的标准代码为空，将跳过。")
            else:
                 product_code_parts.append(str(code_part)) # 确保是字符串
                 logger.debug(f"添加代码部分: '{code_part}' (来自参数 '{std_param_name}')")
        else:
            # 记录未找到代码的参数
            logger.warning(f"参数 '{std_param_name}' 没有匹配到标准代码或匹配信息为 None，无法添加到最终代码中。")

    if not product_code_parts:
        logger.error("未能从任何参数中提取有效的代码部分，无法生成最终产品代码。")
        return None

    final_product_code = "".join(product_code_parts)
    logger.info(f"--- 产品代码生成完成 ---")
    logger.info(f"最终产品代码: {final_product_code}")

    return final_product_code

# --- 可选的测试入口 ---
if __name__ == '__main__':
    # 配置基本日志记录以进行测试
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 模拟 SearchService 的输出 (标准化参数)
    mock_standardized_params = {
        '输出信号': '4~20mA DC',
        '温度变送器': 'YTA710', # 假设这是主型号
        '说明书语言': '英语',
        '传感器输入': '双支输入',
        '壳体代码': '不锈钢',
        '接线口': '1/2 NPT 内螺纹', # 稍微修改以测试模糊匹配
        '内置指示器': '数字LCD', # 稍微修改
        '安装支架': 'SUS304 2寸管道平装', # 稍微修改
        '防爆认证': 'NEPSI 隔爆型' # 假设标准化后是这个名称
    }
    print(f"模拟输入 (标准化参数): \n{json.dumps(mock_standardized_params, indent=2, ensure_ascii=False)}")

    # 设置 libs 目录路径 (假设脚本在项目根目录下运行)
    # 注意：在实际 pipeline 中，settings.STANDARD_LIBS_DIR 会被使用
    # 这里为了独立测试，需要手动指定或确保相对路径正确
    current_dir = Path(__file__).parent
    settings.STANDARD_LIBS_DIR = current_dir.parent.parent / "libs" / "standard"
    settings.STANDARD_INDEX_JSON = settings.STANDARD_LIBS_DIR / "transmitter" / "index.json"
    print(f"测试使用的标准库索引: {settings.STANDARD_INDEX_JSON}")

    # 检查 libs/standard/transmitter 是否存在
    if not settings.STANDARD_INDEX_JSON.parent.exists():
         print(f"\n错误：测试需要 'libs/standard/transmitter' 目录及其内容。请确保该目录存在于项目根目录。")
         print("请从 sensor_seg/libs/standard/transmitter 复制内容到 new_sensor_project/libs/standard/transmitter")
    else:
        # 调用生成函数
        final_code = generate_product_code(mock_standardized_params)

        if final_code:
            print(f"\n生成的最终产品代码: {final_code}")
        else:
            print("\n未能生成产品代码。请检查日志获取详细信息。")
