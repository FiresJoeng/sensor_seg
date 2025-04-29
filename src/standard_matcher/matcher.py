# new_sensor_project/src/standard_matcher/matcher.py
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import sys

# --- 模块导入 ---
# 确保项目根目录在 sys.path 中以便导入 config
project_root = Path(__file__).resolve(
).parent.parent.parent  # 应该是 new_sensor_project
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import settings
    # 使用相对导入从同一目录导入 utils
    from src.standard_matcher.utils import calculate_string_similarity, sort_results_by_csv_order
    # 导入 LLM 工具函数
    from src.utils.llm_utils import call_llm_for_match
except ImportError as e:
    print(
        f"错误 (matcher.py): 导入模块失败 - {e}。请检查项目结构和 PYTHONPATH。", file=sys.stderr)
    raise

# 获取日志记录器实例
logger = logging.getLogger(__name__)

# ==============================================================================
# 类 1: 标准参数匹配阶段
# ==============================================================================


class StandardParamMatcher:
    """
    处理将输入参数与标准库定义进行匹配的初始阶段。
    - 确定主要产品型号。
    - 加载并合并相关的标准库 CSV 文件。
    - 根据标准参数名称在合并数据中查找候选行。
    """

    def __init__(self, index_data: Dict[str, List[str]], base_dir: Path):
        """
        初始化 StandardParamMatcher。

        Args:
            index_data: 从标准库 index.json 加载的数据。
            base_dir: 包含 index.json 文件的目录（用于解析相对 CSV 路径）。
        """
        self.index_data = index_data
        self.base_dir = base_dir
        self.combined_df: Optional[pd.DataFrame] = None
        self.main_model: Optional[str] = None
        self.absolute_csv_files: List[Path] = []

    def find_main_model(self, standardized_params: Dict[str, str]) -> Optional[str]:
        """
        根据标准化参数确定主要产品型号。
        优先考虑像 '温度变送器' 这样的键的精确匹配，然后尝试模糊匹配。

        Args:
            standardized_params: 标准化参数字典 {standard_param_name: standard_param_value}。

        Returns:
            Optional[str]: 匹配的主要型号名称，如果未找到则为 None。
        """
        logger.debug("开始确定产品主型号...")
        available_models = list(self.index_data.keys())
        target_keys = ["温度变送器", "变送器", "YTA",
                       "HZ", "HR", "SZ", "HR"]  # 可能指示主要型号的潜在键
        main_model_value = None
        match_method = "未找到"

        # 1. 尝试使用 standardized_params 中的主要型号键进行精确匹配
        for key in target_keys:
            if key in standardized_params:
                value = standardized_params[key]
                if value in self.index_data:
                    main_model_value = value
                    match_method = f"精确匹配 (键: '{key}', 值: '{value}')"
                    logger.info(f"确定主型号: {main_model_value} ({match_method})")
                    self.main_model = main_model_value
                    return main_model_value
                else:
                    logger.debug(
                        f"键 '{key}' 的值 '{value}' 不在 index.json 的主型号列表中。")

        # 2. 如果精确匹配失败，尝试模糊匹配
        logger.info("主型号精确匹配失败，尝试模糊匹配...")
        best_match_key = None
        best_match_model = None
        best_score = 0.0

        for input_key, input_value in standardized_params.items():
            if not input_key or not input_value:
                continue

            key_score = 0.0
            for target_key in target_keys:
                key_score = max(key_score, calculate_string_similarity(
                    target_key, input_key))

            for model in available_models:
                model_score = calculate_string_similarity(input_value, model)
                combined_score = (key_score * 0.4) + \
                    (model_score * 0.6)  # 组合键和值的相似度

                if combined_score > best_score:
                    best_score = combined_score
                    best_match_key = input_key
                    best_match_model = model

        threshold = settings.MAIN_MODEL_SIMILARITY_THRESHOLD
        if best_score >= threshold:
            main_model_value = best_match_model
            match_method = f"模糊匹配 (最佳匹配键: '{best_match_key}', 值: '{standardized_params.get(best_match_key)}', 匹配型号: '{best_match_model}', 得分: {best_score:.4f})"
            logger.info(f"确定主型号: {main_model_value} ({match_method})")
            self.main_model = main_model_value
            return main_model_value
        else:
            logger.error(
                f"无法确定主型号。最佳模糊匹配得分 {best_score:.4f} 低于阈值 {threshold}。")
            logger.error(
                f"(最佳模糊匹配详情: 键='{best_match_key}', 值='{standardized_params.get(best_match_key)}', 型号='{best_match_model}')")
            self.main_model = None
            return None

    def load_and_combine_data(self) -> bool:
        """
        加载并合并与确定的主要型号关联的 CSV 文件。

        Returns:
            bool: 如果数据加载和合并成功则为 True，否则为 False。
        """
        if not self.main_model:
            logger.error("主型号未确定，无法加载 CSV 数据。")
            return False

        relative_csv_files = self.index_data.get(self.main_model)
        if not relative_csv_files:
            logger.error(f"索引文件中未找到主型号 '{self.main_model}' 关联的 CSV 文件列表。")
            return False

        self.absolute_csv_files = [self.base_dir /
                                   Path(f) for f in relative_csv_files]
        logger.debug(
            f"找到与主型号 '{self.main_model}' 关联的 CSV 文件: {[p.name for p in self.absolute_csv_files]}")

        if not self.absolute_csv_files:
            logger.error("没有提供要加载的 CSV 文件列表。")
            return False

        dfs = []
        logger.debug(f"开始加载和合并 {len(self.absolute_csv_files)} 个 CSV 文件...")
        for file_path in self.absolute_csv_files:
            try:
                if not file_path.is_file():
                    logger.warning(f"标准库 CSV 文件未找到，跳过: {file_path}")
                    continue
                df = pd.read_csv(file_path, dtype=str)  # 将所有内容读取为字符串
                dfs.append(df)
                logger.debug(f"已加载 CSV: {file_path.name} ({len(df)} 行)")
            except Exception as e:
                logger.error(
                    f"加载 CSV 文件 '{file_path.name}' 时出错: {e}", exc_info=True)
                continue  # 跳过有错误的文件

        if not dfs:
            logger.error("未能成功加载任何指定的 CSV 文件。")
            self.combined_df = None
            return False

        try:
            self.combined_df = pd.concat(dfs, ignore_index=True)
            self.combined_df.fillna('', inplace=True)  # 用空字符串填充 NaN
            logger.info(
                f"成功合并 {len(dfs)} 个 CSV 文件，总计 {len(self.combined_df)} 行。")
            return True
        except Exception as e:
            logger.error(f"合并 DataFrame 时出错: {e}", exc_info=True)
            self.combined_df = None
            return False

    def find_candidate_rows(self, standard_param_name: str) -> pd.DataFrame:
        """
        在合并的 DataFrame 中查找与标准参数名称匹配的候选行。
        首先使用精确匹配，然后对 'model' 列进行模糊匹配。

        Args:
            standard_param_name: 标准参数名称（例如，'输出信号'）。

        Returns:
            pd.DataFrame: 包含匹配行的 DataFrame，如果未找到则为空 DataFrame。
        """
        if self.combined_df is None or self.combined_df.empty:
            logger.error("合并后的 DataFrame 为空或未加载，无法查找候选行。")
            return pd.DataFrame()

        logger.debug(f"为参数 '{standard_param_name}' 查找候选行...")
        if 'model' not in self.combined_df.columns:
            logger.error("合并的 DataFrame 中缺少 'model' 列。无法查找候选行。")
            return pd.DataFrame()

        # 1. 对 'model' 列进行精确匹配
        exact_matches = self.combined_df[self.combined_df['model']
                                         == standard_param_name]
        if not exact_matches.empty:
            logger.debug(f"找到 {len(exact_matches)} 个精确匹配行。")
            return exact_matches

        # 2. 对 'model' 列进行模糊匹配
        logger.debug("未找到精确匹配，尝试模糊匹配 'model' 列...")
        fuzzy_matches_models = []
        threshold = settings.FUZZY_MATCH_THRESHOLD
        for model_in_df in self.combined_df['model'].unique():
            similarity = calculate_string_similarity(
                standard_param_name, model_in_df)
            if similarity >= threshold:
                fuzzy_matches_models.append(model_in_df)
                logger.debug(
                    f"  - 模糊匹配到 model: '{model_in_df}' (相似度: {similarity:.4f})")

        if fuzzy_matches_models:
            fuzzy_matches_df = self.combined_df[self.combined_df['model'].isin(
                fuzzy_matches_models)]
            logger.debug(f"找到 {len(fuzzy_matches_df)} 个模糊匹配行 (基于 model 列)。")
            return fuzzy_matches_df
        else:
            logger.warning(
                f"参数 '{standard_param_name}' 在标准库的 'model' 列中未找到精确或模糊匹配项。尝试 LLM 检索...")

            # 3. LLM 后备方案
            unique_models = list(self.combined_df['model'].unique())
            if not unique_models:
                logger.warning("合并后的 DataFrame 中没有唯一的 'model' 值可供 LLM 检索。")
                return pd.DataFrame()

            system_prompt = """
            你是一位专门研究工业传感器规格的专家助手。
            你的任务是将输入的参数名称与提供的列表中最可能的标准参数名称（'model'）进行匹配。
            分析输入参数名称和标准名称列表。
            仅以 JSON 对象响应，其中包含列表中最佳匹配的标准名称。
            响应示例：{"matched_model": "标准参数名称"}
            如果未找到合适的匹配项，请响应：{"error": "未找到合适的匹配项"}
            请勿在 JSON 对象之外包含任何解释或对话性文本。
            """
            user_prompt = f"""
            输入参数名称："{standard_param_name}"

            标准参数名称列表（'model' 列的值）：
            {json.dumps(unique_models, ensure_ascii=False)}

            从列表中为输入参数名称识别出唯一的最佳匹配项。
            """

            llm_result = call_llm_for_match(system_prompt, user_prompt)

            if llm_result and "matched_model" in llm_result and llm_result["matched_model"] in unique_models:
                matched_model_name = llm_result["matched_model"]
                logger.info(
                    f"LLM 匹配到 model: '{matched_model_name}' (来自输入参数 '{standard_param_name}')")
                llm_matches_df = self.combined_df[self.combined_df['model']
                                                  == matched_model_name]
                return llm_matches_df
            else:
                logger.warning(
                    f"LLM 未能为参数 '{standard_param_name}' 找到匹配的 model。LLM 结果: {llm_result}")
                return pd.DataFrame()  # 如果 LLM 失败或未找到匹配项，则返回空 DataFrame

    def get_reference_csv(self) -> Optional[Path]:
        """返回第一个 CSV 文件的路径，用于排序。"""
        return self.absolute_csv_files[0] if self.absolute_csv_files else None

# ==============================================================================
# 类 2: 规格代码匹配规则
# ==============================================================================


class SpecCodeMatcher:
    """
    根据标准参数值从候选行中选择最佳匹配行。
    使用模糊匹配比较 'description'、'code' 和 'model' 字段。
    """

    def select_best_match(self, standard_param_value: str, candidate_rows: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        根据标准参数值从候选行中选择最佳匹配行。

        Args:
            standard_param_value: 标准参数的值（例如，'4~20mA DC'）。
            candidate_rows: 包含候选匹配行的 DataFrame。

        Returns:
            Optional[Dict[str, Any]]: 最佳匹配行的字典表示，如果未找到合适的匹配项则为 None。
        """
        if candidate_rows.empty:
            logger.debug("没有候选行可供选择最佳匹配。")
            return None

        best_item_dict: Optional[Dict[str, Any]] = None
        best_score = -1.0
        match_field = None  # 跟踪哪个字段产生了最佳匹配

        logger.debug(
            f"为值 '{standard_param_value}' 从 {len(candidate_rows)} 个候选中选择最佳匹配...")

        for index, row in candidate_rows.iterrows():
            for field in ('description', 'code', 'model'):  # 检查的字段顺序
                if field in row:
                    text_in_df = str(row[field])  # 确保进行字符串比较
                    score = calculate_string_similarity(
                        standard_param_value, text_in_df)

                    if score > best_score:
                        best_score = score
                        best_item_dict = row.to_dict()  # 将最佳行转换为字典
                        match_field = field
                        logger.debug(
                            f"  * 新的最佳匹配: 行索引 {index}, 字段 '{field}', 得分 {score:.4f}")

        threshold = settings.FUZZY_MATCH_THRESHOLD
        if best_score >= threshold and best_item_dict is not None:
            logger.debug(
                f"选择最佳匹配完成。最佳得分 {best_score:.4f} (来自字段 '{match_field}') >= 阈值 {threshold}。")
            return best_item_dict
        # 如果找到了匹配项但得分低于阈值，或者根本没有找到匹配项，则尝试 LLM
        elif best_score < threshold:
            if best_item_dict is not None:
                logger.warning(
                    f"模糊匹配得分 {best_score:.4f} 低于阈值 {threshold} (来自字段 '{match_field}')。尝试 LLM 检索...")
            else:
                logger.warning(
                    f"未能为值 '{standard_param_value}' 在候选行中找到模糊匹配项。尝试 LLM 检索...")

            # 3. LLM 后备方案，用于根据值选择最佳行
            # 将候选行转换为字典列表以用于提示，如有必要限制大小
            candidate_dicts = candidate_rows.to_dict('records')
            # TODO: 如果 candidate_dicts 对于 LLM 提示来说太大，则实现限制其大小的逻辑

            system_prompt = """
            你是一位专门研究工业传感器规格的专家助手。
            你的任务是根据目标参数值从候选列表中选择最佳匹配的规格行。
            分析目标值以及每个候选行的 'description'、'code' 和 'model' 字段。
            仅以 JSON 对象响应，其中包含最佳匹配候选行的字典表示。
            响应示例：{"best_match": {"model": "型号名称", "description": "描述", "code": "代码", ...}}
            如果在候选者中未找到合适的匹配项，请响应：{"error": "未找到合适的匹配项"}
            请勿在 JSON 对象之外包含任何解释或对话性文本。
            """
            user_prompt = f"""
            目标参数值："{standard_param_value}"

            候选行：
            {json.dumps(candidate_dicts, ensure_ascii=False, indent=2)}

            根据目标参数值从候选者中选择唯一的最佳匹配行。
            """

            llm_result = call_llm_for_match(system_prompt, user_prompt)

            if llm_result and "best_match" in llm_result and isinstance(llm_result["best_match"], dict):
                best_match_dict_llm = llm_result["best_match"]
                # 基本验证：检查返回的字典是否看起来合理（例如，是否包含 'code'）
                if 'code' in best_match_dict_llm:
                    logger.info(
                        f"LLM 选择了最佳匹配行 (来自值 '{standard_param_value}'). Code: '{best_match_dict_llm.get('code')}'")
                    # 确保所有原始列都存在，如果需要则填充缺失的列？还是信任 LLM 的格式？
                    # 目前，按从 LLM 收到的原样返回字典。
                    return best_match_dict_llm
                else:
                    logger.warning(
                        f"LLM 返回的最佳匹配字典缺少 'code' 字段: {best_match_dict_llm}")
                    return None
            else:
                logger.warning(
                    f"LLM 未能为值 '{standard_param_value}' 选择最佳匹配行。LLM 结果: {llm_result}")
                return None  # 如果 LLM 失败或未找到匹配项，则返回 None

# ==============================================================================
# 类 3: 规格代码占位符格式（排序）
# ==============================================================================


class SpecCodeFormatter:
    """
    根据参考 CSV 文件中定义的顺序对匹配的规格代码进行排序。
    """

    def sort_matches(self, final_matches: Dict[str, Optional[Dict[str, Any]]], reference_csv: Optional[Path]) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        根据参考 CSV 中 'model' 条目的顺序对最终匹配项进行排序。

        Args:
            final_matches: {standard_param_name: best_match_row_dict} 的字典。
            reference_csv: 用作排序参考的 CSV 文件的路径。

        Returns:
            Dict[str, Optional[Dict[str, Any]]]: 已排序的匹配项字典。
        """
        if reference_csv and reference_csv.is_file():
            logger.info(f"使用 '{reference_csv.name}' 作为参考对结果进行排序...")
            try:
                # 使用从 utils 导入的工具函数
                sorted_matches = sort_results_by_csv_order(
                    final_matches, reference_csv)
                return sorted_matches
            except Exception as e:
                logger.error(
                    f"使用 '{reference_csv.name}' 排序时出错: {e}. 将返回原始顺序。", exc_info=True)
                return final_matches
        elif reference_csv:
            logger.warning(f"参考 CSV 文件 '{reference_csv.name}' 未找到，将使用原始匹配顺序。")
            return final_matches
        else:
            logger.warning("没有提供参考 CSV 文件进行排序，将使用原始匹配顺序。")
            return final_matches

# ==============================================================================
# 类 4: 型号代码推荐与理由生成
# ==============================================================================


class SpecCodeGenerator:
    """
    协调推荐最终产品型号代码及生成推荐理由的过程。
    使用 StandardParamMatcher、SpecCodeMatcher 和 SpecCodeFormatter。
    """

    def __init__(self):
        """初始化 SpecCodeGenerator。"""
        self.param_matcher: Optional[StandardParamMatcher] = None
        self.code_matcher = SpecCodeMatcher()
        self.formatter = SpecCodeFormatter()

    def _initialize_param_matcher(self) -> bool:
        """加载索引数据并初始化 StandardParamMatcher。"""
        index_path = settings.STANDARD_INDEX_JSON
        if not index_path.is_file():
            logger.error(f"标准库索引文件未找到: {index_path}")
            return False
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            logger.debug(f"成功加载标准库索引: {index_path.name}")
            base_dir = index_path.parent
            self.param_matcher = StandardParamMatcher(index_data, base_dir)
            return True
        except Exception as e:
            logger.error(
                f"加载或解析标准库索引 '{index_path.name}' 时出错: {e}", exc_info=True)
            return False

    def generate(self, standardized_params: Dict[str, str]) -> Optional[Tuple[str, str]]:
        """
        根据标准化参数推荐产品型号代码并生成推荐理由。

        Args:
            standardized_params: 标准化参数字典 {standard_param_name: standard_param_value}。

        Returns:
            Optional[Tuple[str, str]]: 包含推荐型号代码和推荐理由的元组 (code, reason)，
                                       如果发生错误则为 None。
        """
        logger.info("--- 开始型号代码推荐与理由生成 ---")
        if not standardized_params:
            logger.error("输入参数为空，无法推荐型号代码。")
            return None

        # 1. 初始化参数匹配器（加载 index.json）
        if not self._initialize_param_matcher() or self.param_matcher is None:
            # 错误已在 _initialize_param_matcher 中记录
            return None

        # 2. 确定主要型号
        main_model = self.param_matcher.find_main_model(standardized_params)
        if main_model is None:
            # 错误已在 find_main_model 中记录
            return None

        # 3. 加载并合并主要型号的 CSV 数据
        if not self.param_matcher.load_and_combine_data():
            # 错误已在 load_and_combine_data 中记录
            return None

        # 4. 匹配每个标准化参数
        final_matches: Dict[str, Optional[Dict[str, Any]]] = {}
        logger.info("开始匹配每个标准化参数到标准库...")
        for std_param_name, std_param_value in standardized_params.items():
            # 跳过可能用于主要型号识别的参数（启发式）
            # TODO: 改进主要型号参数的识别方法
            if std_param_name in ["温度变送器", "变送器"] and std_param_value == main_model:
                logger.debug(f"跳过主型号参数 '{std_param_name}' 的匹配。")
                continue

            logger.debug(
                f"--- 匹配参数: '{std_param_name}' = '{std_param_value}' ---")
            # a. 查找候选行（基于标准参数名称）
            candidate_rows_df = self.param_matcher.find_candidate_rows(
                std_param_name)

            # b. 从候选者中选择最佳匹配（基于标准参数值）
            best_match_row_dict = self.code_matcher.select_best_match(
                std_param_value, candidate_rows_df)

            # c. 存储最佳匹配结果（可能为 None）
            final_matches[std_param_name] = best_match_row_dict
            if best_match_row_dict:
                logger.debug(
                    f"参数 '{std_param_name}' 的最佳匹配代码: '{best_match_row_dict.get('code', 'N/A')}'")
            else:
                logger.warning(f"参数 '{std_param_name}' 未能找到合适的标准代码。")

        # 5. 根据参考 CSV 顺序对结果进行排序
        reference_csv = self.param_matcher.get_reference_csv()
        sorted_matches = self.formatter.sort_matches(
            final_matches, reference_csv)

        # 6. 组装推荐型号代码
        product_code_parts: List[str] = []
        logger.info("开始拼接推荐型号代码...")
        for std_param_name, match_info in sorted_matches.items():
            if match_info and 'code' in match_info:
                code_part = match_info['code']
                if pd.isna(code_part) or code_part == '':
                    logger.warning(f"参数 '{std_param_name}' 匹配到的标准代码为空，将跳过。")
                else:
                    product_code_parts.append(str(code_part))  # 确保是字符串
                    logger.debug(
                        f"添加代码部分: '{code_part}' (来自参数 '{std_param_name}')")
            else:
                logger.warning(
                    f"参数 '{std_param_name}' 没有匹配到标准代码或匹配信息为 None，无法添加到推荐型号代码中。")

        if not product_code_parts:
            logger.error("未能从任何参数中提取有效的代码部分，无法生成推荐型号代码。")
            return None

        recommended_code = f"{main_model}" + "".join(product_code_parts)
        logger.info(f"--- 型号代码推荐完成 ---")
        logger.info(f"推荐型号代码: {recommended_code}")

        # 7. 生成推荐理由 (新增步骤)
        recommendation_reason = self._generate_recommendation_reason(
            standardized_params, sorted_matches, recommended_code)

        return recommended_code, recommendation_reason

    def _generate_recommendation_reason(self, user_requirements: Dict[str, str], matched_details: Dict[str, Optional[Dict[str, Any]]], recommended_code: str) -> str:
        """
        调用 LLM 生成推荐理由。

        Args:
            user_requirements: 用户输入的标准化参数字典。
            matched_details: 匹配到的每个参数的详细信息字典。
            recommended_code: 最终推荐的型号代码。

        Returns:
            str: LLM 生成的推荐理由，如果失败则返回默认提示信息。
        """
        logger.info("开始生成推荐理由...")
        try:
            # 准备 LLM 输入
            relevant_details = []
            for param, details in matched_details.items():
                if details:
                    # 提取关键信息用于生成理由，可以根据需要调整提取的字段
                    detail_str = f"参数 '{param}': 匹配值 '{details.get('description', 'N/A')}' (代码: {details.get('code', 'N/A')})"
                    if details.get('remark'):
                        detail_str += f", 备注: {details['remark']}"
                    relevant_details.append(detail_str)

            system_prompt = """
            你是一位专业的工业自动化产品选型顾问，尤其擅长温度变送器。
            你的任务是根据用户提供的需求参数和系统匹配到的产品规格细节，为推荐的产品型号代码生成一段简洁、专业且易于理解的推荐理由。
            重点突出推荐型号的关键特性如何满足了用户的核心需求。
            语言风格应专业、客观、自信。避免口语化表达。
            直接输出推荐理由文本，不要包含任何额外的前缀或解释性文字 (例如不要说 "推荐理由如下：")。
            """
            user_prompt = f"""
            用户需求 (标准化参数):
            {json.dumps(user_requirements, indent=2, ensure_ascii=False)}

            系统匹配到的规格细节:
            {chr(10).join(relevant_details)}

            最终推荐型号代码: {recommended_code}

            请基于以上信息，生成推荐理由：
            """
            # 调用 LLM，明确告知不需要 JSON 格式
            llm_response = call_llm_for_match(system_prompt, user_prompt, expect_json=False)

            reason = "未能生成有效的推荐理由。" # 默认值

            if isinstance(llm_response, str):
                # 成功获取到字符串响应
                reason = llm_response.strip()
                if reason:
                    logger.info(f"成功生成推荐理由 (纯文本): {reason[:100]}...")
                else:
                    logger.warning("LLM 返回了空的推荐理由字符串。")
                    reason = "未能生成有效的推荐理由 (LLM 返回空)。" # 提供更具体的默认信息
            elif isinstance(llm_response, dict) and "error" in llm_response:
                # LLM 调用或处理过程中发生错误 (例如 API 错误)
                logger.error(f"LLM 生成推荐理由时返回错误字典: {llm_response}")
                reason = f"无法生成推荐理由：LLM 调用出错 ({llm_response.get('error', '未知错误')}: {llm_response.get('details', '无详情')})"
            elif llm_response is None:
                 # LLM 调用被跳过 (例如，没有 API 密钥) 或其他原因返回 None
                 logger.error("LLM 生成推荐理由调用未返回任何结果 (返回 None)。")
                 reason = "无法生成推荐理由：LLM 调用未返回结果。"
            else:
                # 收到意外的类型
                logger.error(f"LLM 生成推荐理由时返回了意外的类型: {type(llm_response)} - {llm_response}")
                reason = f"无法生成推荐理由：LLM 返回类型无效 ({type(llm_response)})。"

            return reason

        except Exception as e:
            logger.error(f"生成推荐理由时发生异常: {e}", exc_info=True)
            return f"无法生成推荐理由：内部错误 ({e})"


# --- 主执行 / 测试块 ---
if __name__ == '__main__':
    # 配置基本日志记录以进行测试
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 模拟标准化参数
    mock_standardized_params = {
        '输出信号': '4~20mA DC BRAIN通信型',
        '温度变送器': 'YTA710',  # 假定的主要型号
        '说明书语言': '英语',
        '传感器输入': '双支输入',
        '壳体代码': '不锈钢',
        '接线口': '1/2 NPT 内螺纹',  # 为模糊测试稍作修改
        '内置指示器': '数字LCD',  # 稍作修改
        '安装支架': 'SUS304 2寸管道平装',  # 稍作修改
        '防爆认证': 'NEPSI 隔爆型'  # 假定的标准化名称
    }
    print(
        f"模拟输入 (标准化参数): \n{json.dumps(mock_standardized_params, indent=2, ensure_ascii=False)}")

    # 设置测试路径（假设脚本从项目根目录或类似上下文运行）
    # 在实际流水线中，设置将从外部配置
    current_dir = Path(__file__).parent
    # 如果需要，根据实际执行上下文进行调整
    settings.STANDARD_LIBS_DIR = project_root / "libs" / "standard"
    settings.STANDARD_INDEX_JSON = settings.STANDARD_LIBS_DIR / \
        "transmitter" / "index.json"
    print(f"测试使用的标准库索引: {settings.STANDARD_INDEX_JSON}")

    # 检查所需目录是否存在
    if not settings.STANDARD_INDEX_JSON.parent.exists():
        print(f"\n错误：测试需要 '{settings.STANDARD_INDEX_JSON.parent}' 目录及其内容。")
        print("请确保该目录存在于项目结构中。")
    else:
        # 实例化生成器并运行过程
        generator = SpecCodeGenerator()
        # Now returns a tuple or None
        result = generator.generate(mock_standardized_params)

        if result:
            recommended_code, recommendation_reason = result
            print(f"\n推荐型号代码: {recommended_code}")
            print(f"\n推荐理由:\n{recommendation_reason}")
        else:
            print("\n未能推荐型号代码或生成理由。请检查日志获取详细信息。")
