# new_sensor_project/src/standard_matcher/matcher.py
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
import pandas as pd
import sys

# --- 模块导入 ---
# 确保项目根目录在 sys.path 中以便导入 config
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import settings
    # Import the new utility function
    from src.standard_matcher.utils import calculate_string_similarity, get_model_order_from_csv
    from src.utils.llm_utils import call_llm_for_match
except ImportError as e:
    print(
        f"错误 (matcher.py): 导入模块失败 - {e}。请检查项目结构和 PYTHONPATH。", file=sys.stderr)
    raise

# 获取日志记录器实例
logger = logging.getLogger(__name__)

# ==============================================================================
# 类 1: 标准库加载器 (修改后 - 添加 category_to_keywords)
# ==============================================================================

class StandardLoader:
    """
    负责加载所有标准库 CSV 文件，并维护类别/关键词到关联标准库的映射。
    """

    def __init__(self):
        """初始化 StandardLoader。"""
        self.all_standards: Dict[str, pd.DataFrame] = {} # standard_key -> DataFrame
        self.standard_param_names: Dict[str, Set[str]] = {} # standard_key -> set(param_names)
        # self.standard_main_models: Dict[str, Optional[str]] = {} # 不再直接使用主型号
        # 存储关键词到其关联的标准键列表的映射 (standard_key = category/stem)
        self.keyword_to_keys: Dict[str, List[str]] = {}
        # 存储关键词到其关联的原始 CSV 文件路径列表的映射 (用于排序)
        self.keyword_to_csv_paths: Dict[str, List[str]] = {}
        # 新增：存储类别到其包含的关键词列表的映射
        self.category_to_keywords: Dict[str, List[str]] = {}

    def load_all(self) -> bool:
        """
        加载 libs/standard/index.json 定义的所有标准库 CSV 文件。
        并构建类别/关键词到标准键/CSV路径的映射。

        Returns:
            bool: 如果至少成功加载了一个标准则为 True，否则为 False。
        """
        base_path = settings.STANDARD_LIBS_DIR
        index_path = base_path / "index.json"

        if not index_path.is_file():
            logger.error(f"主索引文件未找到: {index_path}")
            return False

        logger.info(f"开始从主索引 {index_path} 加载所有标准库...")
        loaded_count = 0
        # 清空旧数据
        self.all_standards.clear()
        self.standard_param_names.clear()
        self.keyword_to_keys.clear()
        self.keyword_to_csv_paths.clear()
        self.category_to_keywords.clear()

        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            logger.debug(f"已加载主索引: {index_path.name}")

            if not isinstance(index_data, dict):
                logger.error(f"主索引文件格式错误，根元素必须是字典: {index_path.name}")
                return False

            # 遍历索引中的顶级键 (分类, e.g., 'tg', 'sensor', 'transmitter')
            for category, category_data in index_data.items():
                if not isinstance(category_data, dict):
                    logger.warning(
                        f"索引 '{index_path.name}' 中的分类 '{category}' 的值不是字典，跳过。")
                    continue

                # 存储该类别的关键词列表
                self.category_to_keywords[category] = list(category_data.keys())
                logger.debug(f"为类别 '{category}' 存储了关键词: {self.category_to_keywords[category]}")

                # 遍历分类下的关键词 (原主型号)
                for keyword, relative_csv_files in category_data.items():
                    if not isinstance(relative_csv_files, list):
                        logger.warning(
                            f"索引 '{index_path.name}' -> '{category}' -> '{keyword}' 的值不是列表，跳过。")
                        continue

                    # 初始化该关键词的列表
                    self.keyword_to_keys[keyword] = []
                    self.keyword_to_csv_paths[keyword] = []

                    # 遍历关键词关联的 CSV 文件列表
                    for relative_csv in relative_csv_files:
                        if not isinstance(relative_csv, str):
                            logger.warning(
                                f"索引 '{index_path.name}' -> '{category}' -> '{keyword}' 中的文件名不是字符串，跳过: {relative_csv}")
                            continue

                        csv_path = base_path / Path(relative_csv)
                        standard_key = f"{category}/{csv_path.stem}" # 使用 category/stem

                        # 加载单个 CSV
                        if self._load_single_standard(standard_key, csv_path):
                            loaded_count += 1
                            self.keyword_to_keys[keyword].append(standard_key)
                            self.keyword_to_csv_paths[keyword].append(relative_csv)
                        else:
                            logger.warning(
                                f"未能加载标准 '{standard_key}' (关键词: {keyword})，将不会包含在映射中。")

        except Exception as e:
            logger.error(
                f"处理主索引文件 '{index_path.name}' 时出错: {e}", exc_info=True)
            return False

        if loaded_count > 0:
            logger.info(f"成功加载 {loaded_count} 个标准 CSV 文件。")
            logger.debug(f"构建的类别到关键词映射: {self.category_to_keywords}")
            logger.debug(f"构建的关键词到标准键映射: {self.keyword_to_keys}")
            logger.debug(f"构建的关键词到 CSV 路径映射: {self.keyword_to_csv_paths}")
            return True
        else:
            logger.error("未能成功加载任何标准 CSV 文件。")
            return False

    def _load_single_standard(self, standard_key: str, csv_path: Path) -> bool:
        """加载单个 CSV 文件并存储其信息。"""
        try:
            if not csv_path.is_file():
                logger.warning(f"标准库 CSV 文件未找到，跳过: {csv_path}")
                return False
            df = pd.read_csv(csv_path, dtype=str)
            df.fillna('', inplace=True)

            if 'model' not in df.columns:
                logger.error(f"CSV 文件 '{csv_path.name}' 缺少必需的 'model' 列，无法加载。")
                return False

            self.all_standards[standard_key] = df
            self.standard_param_names[standard_key] = set(df['model'].unique()) - {''}
            logger.debug(f"已加载标准 '{standard_key}' ({len(df)} 行)")
            return True
        except Exception as e:
            logger.error(
                f"加载 CSV 文件 '{csv_path.name}' 时出错: {e}", exc_info=True)
            return False

    def find_candidate_rows(self, standard_param_name: str, target_df: pd.DataFrame) -> pd.DataFrame:
        """
        在 *目标* DataFrame 中查找与标准参数名称匹配的候选行。
        策略：精确匹配 -> 模糊匹配 -> LLM 匹配。
        (此方法逻辑不变)
        """
        if target_df is None or target_df.empty:
            logger.debug(f"目标 DataFrame 为空，无法为 '{standard_param_name}' 查找候选行。")
            return pd.DataFrame()

        logger.debug(f"在 DataFrame 中为参数 '{standard_param_name}' 查找候选行...")
        if 'model' not in target_df.columns:
            logger.error("目标 DataFrame 缺少 'model' 列。")
            return pd.DataFrame()

        # 1. 精确匹配
        exact_matches = target_df[target_df['model'] == standard_param_name]
        if not exact_matches.empty:
            logger.info(f"找到 {len(exact_matches)} 个精确匹配行 (参数名: '{standard_param_name}')。")
            return exact_matches

        # 2. 模糊匹配
        unique_models = target_df['model'].dropna().unique()
        best_fuzzy_score = -1.0
        best_fuzzy_models = []
        for model_in_df in unique_models:
            if isinstance(model_in_df, str) and model_in_df:
                score = calculate_string_similarity(standard_param_name, model_in_df)
                if score > best_fuzzy_score:
                    best_fuzzy_score = score
                    best_fuzzy_models = [model_in_df]
                elif score == best_fuzzy_score:
                    best_fuzzy_models.append(model_in_df)

        threshold = settings.FUZZY_MATCH_THRESHOLD
        if best_fuzzy_score >= threshold:
            logger.info(f"找到 {len(best_fuzzy_models)} 个模糊匹配模型 (得分: {best_fuzzy_score:.4f})，模型: {best_fuzzy_models}。")
            fuzzy_matches = target_df[target_df['model'].isin(best_fuzzy_models)]
            return fuzzy_matches

        # 3. LLM 匹配
        candidate_model_names = [m for m in unique_models if isinstance(m, str) and m]
        if not candidate_model_names:
            logger.warning(f"参数 '{standard_param_name}': 模糊匹配失败，无候选用于 LLM。")
            return pd.DataFrame()

        logger.debug(f"准备调用 LLM 从 {len(candidate_model_names)} 个候选模型中为 '{standard_param_name}' 选择...")
        system_prompt = """
        你是一位专门研究工业传感器规格的专家助手。
        你的任务是根据目标参数名称，从候选模型名称列表中选择最匹配的一个模型名称。
        仅以 JSON 对象响应，其中包含最佳匹配的模型名称。
        响应示例：{"best_match_model_name": "选中的模型名称"}
        如果在候选者中未找到合适的匹配项，请响应：{"error": "未找到合适的匹配项"}
        请勿在 JSON 对象之外包含任何解释或对话性文本。
        """
        user_prompt = f"""
        目标参数名称："{standard_param_name}"
        候选模型名称列表：
        {json.dumps(candidate_model_names, ensure_ascii=False, indent=2)}
        根据目标参数名称从候选列表中选择唯一的最佳匹配模型名称。
        """
        llm_result = call_llm_for_match(system_prompt, user_prompt)

        if llm_result and "best_match_model_name" in llm_result and isinstance(llm_result["best_match_model_name"], str):
            best_match_model_llm = llm_result["best_match_model_name"]
            if best_match_model_llm in candidate_model_names:
                logger.info(f"LLM 选择了最佳匹配模型名称: '{best_match_model_llm}'。")
                llm_matches = target_df[target_df['model'] == best_match_model_llm]
                return llm_matches
            else:
                logger.warning(f"LLM 返回的模型名称 '{best_match_model_llm}' 不在候选列表中。")
        else:
            logger.warning(f"LLM 未能为参数 '{standard_param_name}' 选择最佳匹配模型。LLM 结果: {llm_result}")

        return pd.DataFrame()

# ==============================================================================
# 类 2: 规格代码匹配器 (逻辑不变)
# ==============================================================================

class SpecCodeMatcher:
    """
    根据标准参数值从候选行中选择最佳匹配行。
    使用模糊匹配比较 'description'、'code'、'model' 和 'remark' 字段。
    """
    def select_best_match(self, standard_param_value: str, candidate_rows: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        根据标准参数值从候选行中选择最佳匹配行。
        优化：如果候选行只有一行，则直接返回该行，无需进一步匹配。
        """
        if candidate_rows.empty:
            logger.debug("没有候选行可供选择最佳匹配。")
            return None

        # --- 优化：如果只有一个候选行，直接返回 ---
        if len(candidate_rows) == 1:
            unique_match_dict = candidate_rows.iloc[0].to_dict()
            logger.info(f"参数 '{unique_match_dict.get('model', '未知')}' 只有一个候选规格，直接选用。Code: '{unique_match_dict.get('code', 'N/A')}'")
            return unique_match_dict
        # --- 优化结束 ---

        # 如果有多于一个候选行，才进行后续匹配
        best_item_dict: Optional[Dict[str, Any]] = None
        best_score = -1.0
        match_field = None

        logger.debug(f"为值 '{standard_param_value}' 从 {len(candidate_rows)} 个候选中选择...")

        for index, row in candidate_rows.iterrows():
            for field in ('description', 'code', 'model', 'remark'): # 包含 remark
                if field in row:
                    text_in_df = str(row[field])
                    score = calculate_string_similarity(standard_param_value, text_in_df)
                    if score > best_score:
                        best_score = score
                        best_item_dict = row.to_dict()
                        match_field = field
                        logger.debug(f"  * 新最佳: 行 {index}, 字段 '{field}', 得分 {score:.4f}")

        threshold = settings.FUZZY_MATCH_THRESHOLD
        if best_score >= threshold and best_item_dict is not None:
            logger.debug(f"选择完成。最佳得分 {best_score:.4f} (来自 '{match_field}') >= {threshold}。")
            return best_item_dict
        elif best_score < threshold:
            logger.warning(f"模糊匹配得分 {best_score:.4f} 低于阈值 {threshold} (来自 '{match_field}')。尝试 LLM...")

            candidate_dicts = candidate_rows.to_dict('records')
            system_prompt = """
            你是一位专门研究工业传感器规格的专家助手。
            你的任务是根据目标参数值从候选列表中选择最佳匹配的规格行。
            分析目标值以及每个候选行的 'description'、'code'、'model' 和 'remark' 字段。
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
                if 'code' in best_match_dict_llm:
                    logger.info(f"LLM 选择了最佳匹配行 (来自值 '{standard_param_value}'). Code: '{best_match_dict_llm.get('code')}'")
                    return best_match_dict_llm
                else:
                    logger.warning(f"LLM 返回的最佳匹配字典缺少 'code' 字段: {best_match_dict_llm}")
            else:
                logger.warning(f"LLM 未能为值 '{standard_param_value}' 选择最佳匹配行。LLM 结果: {llm_result}")

        return None # 模糊和 LLM 都失败

# ==============================================================================
# 类 3: 型号代码生成器 (重构后)
# ==============================================================================

class SpecCodeGenerator:
    """
    协调为多个产品类别（如 transmitter, sensor, tg）生成最终产品型号代码及推荐理由的过程。
    """

    def __init__(self):
        """初始化 SpecCodeGenerator。"""
        self.loader = StandardLoader()
        self.code_matcher = SpecCodeMatcher()
        # 定义需要处理的产品类别顺序
        self.target_categories = ["transmitter", "sensor", "tg"]

    def _find_best_keyword_for_category(self, category: str, standardized_params: Dict[str, str]) -> Optional[str]:
        """
        根据输入参数的 *值* 与指定类别下的“关键词”进行匹配，选择关联度最高的关键词。

        Args:
            category: 要查找关键词的产品类别 (e.g., "transmitter").
            standardized_params: 标准化参数字典 {standard_param_name: standard_param_value}。

        Returns:
            Optional[str]: 最佳匹配的关键词，如果未找到则为 None。
        """
        candidate_keywords = self.loader.category_to_keywords.get(category)
        if not candidate_keywords:
            logger.error(f"类别 '{category}' 没有可用的候选关键词。请检查 index.json。")
            return None

        input_param_values = [v for v in standardized_params.values() if isinstance(v, str) and v]
        if not input_param_values:
            logger.error(f"输入参数值列表为空，无法为类别 '{category}' 进行关键词匹配。")
            return None

        logger.info(f"开始为类别 '{category}' 根据输入值 {input_param_values} 选择最佳关键词...")

        best_overall_score = -1.0
        best_keyword: Optional[str] = None

        # 1. 模糊匹配
        keyword_scores: Dict[str, float] = {}
        for keyword in candidate_keywords:
            max_score_for_keyword = 0.0
            for value in input_param_values:
                score = calculate_string_similarity(keyword, value)
                if score > max_score_for_keyword:
                    max_score_for_keyword = score
            keyword_scores[keyword] = max_score_for_keyword
            logger.debug(f"  类别 '{category}', 关键词 '{keyword}' 最高模糊匹配得分: {max_score_for_keyword:.4f}")
            if max_score_for_keyword > best_overall_score:
                best_overall_score = max_score_for_keyword
                best_keyword = keyword

        threshold = settings.FUZZY_MATCH_THRESHOLD
        if best_keyword and best_overall_score >= threshold:
            logger.info(f"类别 '{category}' 通过模糊匹配选择的最佳关键词: '{best_keyword}' (得分: {best_overall_score:.4f})")
            return best_keyword
        else:
            logger.warning(f"类别 '{category}' 模糊匹配未能找到足够高的关键词 (最高分: {best_overall_score:.4f})。尝试 LLM...")

            # 2. LLM 匹配
            system_prompt = f"""
            你是一位专门分析工业产品参数的专家，当前专注于产品类别 '{category}'。
            你的任务是根据一系列输入参数值，从给定的该类别候选“关键词”列表中，选择一个与这些参数值整体最相关的关键词。
            这些关键词通常代表该类别下的产品主型号或子类别。
            请仔细分析输入参数值和候选关键词的含义。
            仅以 JSON 对象响应，其中包含最佳匹配的关键词。
            响应示例：{{"best_match_keyword": "选中的关键词"}}
            如果在候选者中未找到合适的匹配项，请响应：{{"error": "未找到合适的匹配项"}}
            请勿在 JSON 对象之外包含任何解释或对话性文本。
            """
            user_prompt = f"""
            输入参数值列表:
            {json.dumps(input_param_values, ensure_ascii=False, indent=2)}

            类别 '{category}' 的候选关键词列表:
            {json.dumps(candidate_keywords, ensure_ascii=False, indent=2)}

            根据输入参数值，从候选列表中选择最相关的一个关键词。
            """
            llm_result = call_llm_for_match(system_prompt, user_prompt)

            if llm_result and "best_match_keyword" in llm_result and isinstance(llm_result["best_match_keyword"], str):
                best_keyword_llm = llm_result["best_match_keyword"]
                if best_keyword_llm in candidate_keywords:
                    logger.info(f"类别 '{category}' LLM 选择了最佳匹配关键词: '{best_keyword_llm}'")
                    return best_keyword_llm
                else:
                    logger.warning(f"类别 '{category}' LLM 返回的关键词 '{best_keyword_llm}' 不在候选列表中。")
            else:
                logger.error(f"类别 '{category}' LLM 未能选择最佳匹配关键词。LLM 结果: {llm_result}")

            return None # 模糊和 LLM 都失败

    def _generate_code_for_keyword(self, standardized_params: Dict[str, str], selected_keyword: str, associated_standard_keys: List[str], csv_order_list: List[str]) -> Tuple[Optional[str], Dict[str, Optional[Dict[str, Any]]]]:
        """
        为选定的关键词及其关联的标准库生成单个产品型号代码字符串。
        应用详细的排序和占位符规则。

        Args:
            standardized_params: 标准化参数字典。
            selected_keyword: 已选定的关键词。
            associated_standard_keys: 与关键词关联的标准键列表 (category/stem)。
            csv_order_list: 与关键词关联的原始 CSV 路径列表 (用于排序)。

        Returns:
            Tuple[Optional[str], Dict[str, Optional[Dict[str, Any]]]]:
                生成的代码字符串 (如果无法生成则为 None) 和该类别匹配详情的字典。
        """
        logger.info(f"--- 开始为关键词 '{selected_keyword}' 生成代码 ---")
        # 存储从 CSV 匹配到的 (code, match_details)
        product_code_parts: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        # 存储最终匹配结果 {param_name: match_details}, None表示未匹配, dict表示匹配成功
        final_matches_for_keyword: Dict[str, Optional[Dict[str, Any]]] = {}
        # 新增：存储通过用户输入解决的参数及其匹配详情
        user_input_matches: Dict[str, Dict[str, Any]] = {}
        # 新增：存储最终未能匹配（包括用户跳过或输入后仍失败）的参数名
        final_unmatched_models: Set[str] = set()

        matched_param_names = set() # 跟踪已通过标准化参数匹配的参数名

        # --- 1. 遍历所有输入参数，在关联的标准库中查找匹配 ---
        for std_param_name, std_param_value in standardized_params.items():
            if not std_param_name or not std_param_value: continue # 跳过空参数
            if std_param_name in matched_param_names: continue # 跳过已匹配

            logger.debug(f"  尝试匹配参数: '{std_param_name}' = '{std_param_value}'")
            found_match_for_param = False
            best_match_row_dict: Optional[Dict[str, Any]] = None

            # 遍历与关键词关联的所有标准键 (CSV 文件)
            for standard_key in associated_standard_keys:
                logger.debug(f"    在标准 '{standard_key}' 中查找...")
                target_df = self.loader.all_standards.get(standard_key)
                if target_df is None or target_df.empty: continue

                # a. 查找候选行 (基于参数名称)
                candidate_rows_df = self.loader.find_candidate_rows(std_param_name, target_df)

                if not candidate_rows_df.empty:
                    # b. 从候选者中选择最佳匹配 (基于参数值)
                    current_best_match = self.code_matcher.select_best_match(std_param_value, candidate_rows_df)
                    if current_best_match:
                        best_match_row_dict = current_best_match
                        best_match_row_dict['_source_standard_key'] = standard_key # 记录来源
                        logger.debug(f"    在 '{standard_key}' 中为 '{std_param_name}' 找到匹配。")
                        found_match_for_param = True
                        break # 找到后不再查找此参数

            # c. 处理匹配结果
            if found_match_for_param and best_match_row_dict:
                final_matches_for_keyword[std_param_name] = best_match_row_dict
                code_part = best_match_row_dict.get('code')
                if pd.notna(code_part) and str(code_part).strip():
                    code_value = str(code_part).strip()
                    product_code_parts.append((code_value, best_match_row_dict))
                    matched_param_names.add(std_param_name)
                    logger.debug(f"    参数 '{std_param_name}' 匹配代码: '{code_value}' (来自: {standard_key})")
                else:
                    logger.warning(f"    参数 '{std_param_name}' 匹配代码为空或无效 ('{code_part}')，跳过。")
                    final_matches_for_keyword[std_param_name] = None # 标记未找到有效代码
            else:
                # 只有当参数确实在某个关联的标准库的 model 列中存在时，才记录为未匹配成功
                # 检查参数名是否存在于任何关联标准库的 model 列中
                param_exists_in_models = False
                for sk in associated_standard_keys:
                    if std_param_name in self.loader.standard_param_names.get(sk, set()):
                        param_exists_in_models = True
                        break
                if param_exists_in_models:
                    logger.warning(f"  参数 '{std_param_name}' (值: '{std_param_value}') 未能在标准库 {associated_standard_keys} 中找到匹配代码。")
                    final_matches_for_keyword[std_param_name] = None # 标记为未找到匹配
                else:
                     logger.debug(f"  参数 '{std_param_name}' 不属于关键词 '{selected_keyword}' 的标准参数，跳过匹配记录。")


        # --- 2. 应用排序和交互式占位符规则 ---
        logger.info(f"  开始为关键词 '{selected_keyword}' 应用代码排序和交互式占位符规则...")

        # 2.1 获取第一个 CSV 的路径、键、DataFrame 和 model 顺序
        first_csv_relative_path: Optional[str] = None
        first_csv_full_path: Optional[Path] = None
        first_csv_key: Optional[str] = None
        first_csv_df: Optional[pd.DataFrame] = None
        first_csv_model_order: List[str] = []
        first_csv_model_to_index: Dict[str, int] = {}
        category = "unknown" # 提取类别用于提示

        if csv_order_list:
            first_csv_relative_path = csv_order_list[0]
            first_csv_full_path = settings.STANDARD_LIBS_DIR / Path(first_csv_relative_path)
            # 从相对路径推断 standard_key 和 category
            parts = Path(first_csv_relative_path).parts
            if len(parts) >= 2:
                category = parts[0]
                stem = Path(first_csv_relative_path).stem
                first_csv_key = f"{category}/{stem}"
                first_csv_df = self.loader.all_standards.get(first_csv_key)
                # 使用新的工具函数获取 model 顺序
                model_order_result = get_model_order_from_csv(first_csv_full_path)
                if model_order_result:
                    first_csv_model_order = model_order_result
                    first_csv_model_to_index = {model: i for i, model in enumerate(first_csv_model_order)}
                    logger.debug(f"    第一个 CSV ('{first_csv_key}') 的 model 顺序: {first_csv_model_order}")
                else:
                    logger.error(f"    无法从第一个 CSV ('{first_csv_full_path}') 获取 'model' 顺序。")
                    return None, final_matches_for_keyword # 无法排序
            else:
                logger.error(f"    无法从第一个 CSV 路径 '{first_csv_relative_path}' 推断类别和键。")
                return None, final_matches_for_keyword
        else:
            logger.error(f"    关键词 '{selected_keyword}' 没有关联的 CSV 路径列表。")
            return None, final_matches_for_keyword

        # 2.2 初始化最终代码序列和后续代码列表
        final_code_sequence: List[Optional[str]] = [None] * len(first_csv_model_order)
        subsequent_codes_with_details: List[Tuple[str, Dict[str, Any]]] = []
        matched_first_csv_models: Set[str] = set()

        # 2.3 分配匹配到的代码
        for code, details in product_code_parts:
            if details and '_source_standard_key' in details:
                source_key = details['_source_standard_key']
                matched_model = details.get('model')

                if not matched_model:
                    logger.warning(f"    代码 '{code}' (来自 {source_key}) 缺少 'model' 信息，放入后续。")
                    subsequent_codes_with_details.append((code, details))
                    continue

                if source_key == first_csv_key: # 来自第一个 CSV
                    if matched_model in first_csv_model_to_index:
                        idx = first_csv_model_to_index[matched_model]
                        if final_code_sequence[idx] is None:
                            final_code_sequence[idx] = code
                            matched_first_csv_models.add(matched_model)
                            logger.debug(f"      代码 '{code}' (model: {matched_model}) 放入首个 CSV 序列位置 {idx}")
                        else:
                            logger.warning(f"      位置 {idx} (model: {matched_model}) 已被代码 '{final_code_sequence[idx]}' 占据，新代码 '{code}' 放入后续。")
                            subsequent_codes_with_details.append((code, details))
                    else:
                        logger.warning(f"      代码 '{code}' 的 model '{matched_model}' (来自首个 CSV) 不在预期顺序中，放入后续。")
                        subsequent_codes_with_details.append((code, details))
                else: # 来自后续 CSV
                    logger.debug(f"      代码 '{code}' (model: {matched_model}, 来自 {source_key}) 放入后续列表。")
                    subsequent_codes_with_details.append((code, details))
            else:
                logger.warning(f"    代码 '{code}' 缺少来源信息，放入后续。")
                subsequent_codes_with_details.append((code, details if details else {}))

        # 2.4 处理第一个 CSV 中未通过标准化参数匹配的 model (交互式输入)
        placeholder_count = 0
        user_input_success_count = 0
        if first_csv_df is not None: # 确保 DataFrame 已加载
            for i, model_name in enumerate(first_csv_model_order):
                # 检查条件：该 model 在首个CSV顺序中，且未通过标准化参数匹配成功，且当前位置为空
                if model_name not in matched_first_csv_models and final_code_sequence[i] is None:
                    logger.info(f"    参数 '{model_name}' (来自首个 CSV) 未通过标准化参数匹配。")

                    # 尝试向用户请求输入
                    try:
                        # 使用之前提取的 category
                        prompt = f"提示: 类别 '{category}' 的参数 '{model_name}' 未匹配! 请输入您的规格要求 (留空则使用占位符'?'): "
                        user_spec_value = input(prompt).strip()

                        if user_spec_value:
                            logger.info(f"      用户为 '{model_name}' 输入了规格: '{user_spec_value}'，尝试匹配...")
                            # 在第一个 CSV 中查找此 model 的行
                            # 使用 .copy() 避免 SettingWithCopyWarning
                            candidate_rows_df = first_csv_df[first_csv_df['model'] == model_name].copy()
                            if not candidate_rows_df.empty:
                                user_match_details = self.code_matcher.select_best_match(user_spec_value, candidate_rows_df)
                                if user_match_details and 'code' in user_match_details and str(user_match_details['code']).strip():
                                    code = str(user_match_details['code']).strip()
                                    final_code_sequence[i] = code
                                    # 记录匹配详情，标记为用户输入
                                    user_match_details['_source_standard_key'] = first_csv_key # 记录来源
                                    user_match_details['_matched_by_user'] = True # 标记
                                    user_input_matches[model_name] = user_match_details # 存储用户匹配结果
                                    user_input_success_count += 1
                                    logger.info(f"      成功通过用户输入为 '{model_name}' 匹配到代码: '{code}'")
                                else:
                                    logger.warning(f"      用户输入 '{user_spec_value}' 后未能为 '{model_name}' 匹配到有效代码，使用占位符 '?'。")
                                    final_code_sequence[i] = '?'
                                    final_unmatched_models.add(model_name) # 记录最终未匹配
                                    placeholder_count += 1
                            else:
                                # 理论上不应发生，因为 model_name 来自 first_csv_df['model']
                                logger.error(f"      在第一个 CSV 中找不到 model '{model_name}' 的行，无法处理用户输入，使用占位符 '?'。")
                                final_code_sequence[i] = '?'
                                final_unmatched_models.add(model_name)
                                placeholder_count += 1
                        else:
                            logger.info(f"      用户未为 '{model_name}' 输入规格，使用占位符 '?'。")
                            final_code_sequence[i] = '?'
                            final_unmatched_models.add(model_name)
                            placeholder_count += 1
                    except Exception as e:
                        logger.error(f"      处理用户输入或为 '{model_name}' 重新匹配时出错: {e}", exc_info=True)
                        final_code_sequence[i] = '?' # 出错也用占位符
                        final_unmatched_models.add(model_name)
                        placeholder_count += 1
                # else: model 已通过标准化参数匹配，或位置已被填充，无需处理

        else:
             logger.error("    第一个 CSV DataFrame 未加载，无法处理未匹配的 model。")
             # 可能需要决定如何处理这种情况，例如全部设为 '?' 或返回错误

        if user_input_success_count > 0:
            logger.info(f"    成功通过用户输入匹配了 {user_input_success_count} 个参数。")
        if placeholder_count > 0:
            logger.info(f"    为 {placeholder_count} 个最终未匹配的参数插入了占位符 '?'。")


        # 2.5 对后续代码进行排序 (按来源 CSV 顺序)
        def get_subsequent_sort_key(item: Tuple[str, Dict[str, Any]]) -> int:
            code, details = item
            if details and '_source_standard_key' in details:
                source_key = details['_source_standard_key'] # category/stem
                parts = source_key.split('/', 1)
                if len(parts) == 2:
                    stem_from_key = parts[1]
                    for csv_relative_path in csv_order_list: # 使用传入的 csv_order_list
                        if Path(csv_relative_path).stem == stem_from_key:
                            try:
                                index = csv_order_list.index(csv_relative_path)
                                # 确保后续 CSV 的索引排在第一个 CSV 之后
                                return 1 + index # 第一个 CSV 索引为 0
                            except ValueError: pass
                return 1 + len(csv_order_list) # 无法确定，放最后
            return 1 + len(csv_order_list) # 无来源，放最后

        sorted_subsequent_codes_with_details = sorted(
            subsequent_codes_with_details, key=get_subsequent_sort_key
        )
        sorted_subsequent_codes = [code for code, details in sorted_subsequent_codes_with_details]
        logger.debug(f"    排序后的后续代码: {sorted_subsequent_codes}")

        # 3. 组装代码字符串
        # 将 final_code_sequence 中的 None 替换为空字符串 (不应是 '?' 除非明确未匹配)
        final_sequence_str = [c if c is not None else '' for c in final_code_sequence]

        # --- 3. 组装代码字符串 ---
        # 将 final_code_sequence 中的 None 替换为空字符串
        final_sequence_str = [c if c is not None else '' for c in final_code_sequence]

        # 拼接时去除空字符串
        generated_code = "".join(filter(None, final_sequence_str)) + "".join(sorted_subsequent_codes)

        # 合并用户输入匹配结果到最终结果
        final_matches_for_keyword.update(user_input_matches)
        # 添加最终未匹配（使用占位符）的信息
        for unmatched_model in final_unmatched_models:
            # 只添加尚未记录的未匹配项，避免覆盖可能的 None (表示尝试过但失败)
            if unmatched_model not in final_matches_for_keyword:
                 final_matches_for_keyword[unmatched_model] = {'_status': 'unmatched_placeholder'}
            # 如果已存在且为 None，可以考虑更新状态，但当前逻辑是保留 None
            elif final_matches_for_keyword[unmatched_model] is None:
                 logger.debug(f"Model '{unmatched_model}' was already marked as None (match attempted but failed), keeping as None instead of 'unmatched_placeholder'.")


        if not generated_code.replace('?', ''): # 如果去除占位符后仍为空
             logger.warning(f"关键词 '{selected_keyword}' 未能生成任何有效代码部分。")
             # 即使代码为空，也要返回包含用户输入和占位符信息的匹配详情
             return None, final_matches_for_keyword

        logger.info(f"--- 关键词 '{selected_keyword}' 代码生成完成: {generated_code} ---")
        # 返回生成的代码和包含所有匹配/未匹配信息的字典
        return generated_code, final_matches_for_keyword


    def generate(self, standardized_params: Dict[str, str]) -> Optional[Tuple[str, str]]:
        """
        为多个产品类别生成组合型号代码和推荐理由。

        Args:
            standardized_params: 标准化参数字典。

        Returns:
            Optional[Tuple[str, str]]: 包含组合型号代码和推荐理由的元组，如果出错则为 None。
        """
        logger.info("--- 开始多类别型号代码推荐与理由生成 ---")
        if not standardized_params:
            logger.error("输入参数为空，无法推荐。")
            return None

        # 1. 加载所有标准库
        if not self.loader.load_all():
            return None

        # 2. 为每个目标类别生成代码
        category_codes: Dict[str, Optional[str]] = {}
        all_matched_details: Dict[str, Optional[Dict[str, Any]]] = {}
        selected_keywords: Dict[str, Optional[str]] = {} # 存储每个类别选中的关键词

        for category in self.target_categories:
            logger.info(f"--- 处理类别: {category} ---")
            # a. 查找该类别的最佳关键词
            keyword = self._find_best_keyword_for_category(category, standardized_params)
            selected_keywords[category] = keyword

            if keyword:
                # b. 获取关联的标准键和 CSV 路径
                associated_keys = self.loader.keyword_to_keys.get(keyword, [])
                csv_paths = self.loader.keyword_to_csv_paths.get(keyword, [])
                if not associated_keys or not csv_paths:
                    logger.error(f"类别 '{category}', 关键词 '{keyword}': 未找到关联的标准键或 CSV 路径。")
                    category_codes[category] = None
                    continue

                # c. 为该关键词生成代码
                code, matches = self._generate_code_for_keyword(
                    standardized_params, keyword, associated_keys, csv_paths
                )
                category_codes[category] = code
                # 合并匹配详情 (注意：如果不同类别的标准库有同名参数，后匹配的会覆盖)
                # 理论上参数名在整个输入中应唯一，或至少在应用到不同类别时上下文清晰
                all_matched_details.update(matches)
            else:
                logger.warning(f"类别 '{category}': 未能找到匹配的关键词，无法生成代码。")
                category_codes[category] = None

        # 3. 组合最终代码字符串
        final_code_parts = [category_codes.get(cat) or "" for cat in self.target_categories]
        # 过滤掉完全是问号的代码部分
        final_code_parts_filtered = [part for part in final_code_parts if part.replace('?', '')]

        if not final_code_parts_filtered:
             logger.error("所有类别均未能生成有效代码。")
             return None

        # 用空格连接非空的代码部分
        final_combined_code = " ".join(final_code_parts_filtered)
        logger.info(f"--- 最终组合型号代码: {final_combined_code} ---")

        # 4. 生成组合推荐理由
        recommendation_reason = self._generate_recommendation_reason(
            standardized_params, all_matched_details, final_combined_code, selected_keywords
        )

        return final_combined_code, recommendation_reason

    def _generate_recommendation_reason(self, user_requirements: Dict[str, str], all_matched_details: Dict[str, Optional[Dict[str, Any]]], recommended_code: str, selected_keywords: Dict[str, Optional[str]]) -> str:
        """
        调用 LLM 生成组合推荐理由。

        Args:
            user_requirements: 用户输入的标准化参数字典。
            all_matched_details: 所有类别匹配到的参数详细信息字典。
            recommended_code: 最终推荐的组合型号代码。
            selected_keywords: 每个类别选定的关键词字典。

        Returns:
            str: LLM 生成的推荐理由。
        """
        logger.info("开始生成组合推荐理由...")
        try:
            relevant_details = [] # 通过标准化参数匹配的
            unmatched_params = [] # 尝试过但失败的 (标记为 None)
            user_matched_params_info = [] # 通过用户输入匹配的
            placeholder_params_info = [] # 最终使用占位符的
            processed_params = set() # 跟踪已处理的参数

            # 构建关键词和来源信息
            keyword_info_parts = []
            for cat, kw in selected_keywords.items():
                if kw:
                    source_keys = self.loader.keyword_to_keys.get(kw, [])
                    keyword_info_parts.append(f"类别 '{cat}': 选定关键词 '{kw}' (基于标准库: {source_keys})")
                else:
                    keyword_info_parts.append(f"类别 '{cat}': 未能确定关键词")
            keyword_info = "\n".join(keyword_info_parts)


            # 整理匹配和未匹配信息
            for param, req_value in user_requirements.items():
                 if not param or not req_value: continue # 跳过空参数
                 if param in processed_params: continue # 避免重复处理

                 details = all_matched_details.get(param)

                 if isinstance(details, dict): # 匹配成功 (包括用户输入或占位符状态)
                     if details.get('_matched_by_user'):
                         user_matched_params_info.append(f"参数 '{param}': 基于用户输入 '{req_value}' 匹配到代码 '{details.get('code', 'N/A')}' (描述: {details.get('description', 'N/A')})")
                     elif details.get('_status') == 'unmatched_placeholder':
                          placeholder_params_info.append(f"参数 '{param}': 需求 '{req_value}', 未能匹配，使用占位符 '?'")
                     else: # 标准化参数匹配成功
                         source_key_str = details.get('_source_standard_key', '未知')
                         detail_str = f"参数 '{param}': 需求 '{req_value}', 匹配值 '{details.get('description', 'N/A')}' (代码: {details.get('code', 'N/A')}, 来源: {source_key_str})"
                         if details.get('remark'):
                             detail_str += f", 备注: {details['remark']}"
                         relevant_details.append(detail_str)
                     processed_params.add(param)

                 elif details is None and param in all_matched_details: # 显式标记为 None (尝试过但失败)
                     # 检查是否属于相关标准
                     if self._is_param_relevant(param, selected_keywords):
                         unmatched_params.append(f"参数 '{param}': 需求 '{req_value}', 尝试匹配但未成功")
                         processed_params.add(param)
                 # else: 参数不在 all_matched_details 中，忽略 (可能是无关参数)

            system_prompt = f"""
            你是一位专业的工业自动化产品选型顾问，擅长组合温度测量方案（变送器、传感器、保护套管）。
            你的任务是根据用户提供的整体需求参数、系统为每个产品类别（transmitter, sensor, tg）分别匹配到的规格细节、选定的关键词、通过用户输入补充的参数、最终未能匹配的参数（包括使用占位符'?'的）以及最终推荐的组合型号代码，生成一段简洁、专业且连贯的推荐理由。
            理由需要整合三个部分，说明组合方案如何满足用户的整体需求。
            明确指出哪些参数是基于用户输入补充的，哪些参数最终未能匹配或使用了占位符。
            语言风格应专业、客观、自信。避免口语化表达。
            直接输出推荐理由文本，不要包含任何额外的前缀或解释性文字。
            使用中文回答。
            """
            user_prompt = f"""
            用户整体需求 (标准化参数):
            {json.dumps(user_requirements, indent=2, ensure_ascii=False)}

            各类别选定的关键词及标准库:
            {keyword_info}

            系统匹配到的规格细节汇总 (来自标准化参数):
            {chr(10).join(relevant_details) if relevant_details else "无通过标准化参数成功匹配的规格细节。"}

            通过用户输入补充的参数:
            {chr(10).join(user_matched_params_info) if user_matched_params_info else "无"}

            未能匹配的相关参数 (尝试过但失败):
            {chr(10).join(unmatched_params) if unmatched_params else "无"}

            最终使用占位符'?'的参数:
            {chr(10).join(placeholder_params_info) if placeholder_params_info else "无"}

            最终推荐组合型号代码: {recommended_code}

            请基于以上信息，生成组合推荐理由：
            """
            llm_response = call_llm_for_match(system_prompt, user_prompt, expect_json=False)

            reason = "未能生成有效的推荐理由。"
            if isinstance(llm_response, str) and llm_response.strip():
                reason = llm_response.strip()
                logger.info(f"成功生成组合推荐理由: {reason[:100]}...")
            elif isinstance(llm_response, dict) and "error" in llm_response:
                 logger.error(f"LLM 生成组合理由时返回错误: {llm_response}")
                 reason = f"无法生成推荐理由：LLM 调用出错 ({llm_response.get('error', '未知错误')})"
            else:
                 logger.error(f"LLM 生成组合理由时返回意外结果: {llm_response}")
                 reason = f"无法生成推荐理由：LLM 返回无效 ({type(llm_response)})。"

            return reason

        except Exception as e:
            logger.error(f"生成组合推荐理由时发生异常: {e}", exc_info=True)
            return f"无法生成推荐理由：内部错误 ({e})"

    def _is_param_relevant(self, param_name: str, selected_keywords: Dict[str, Optional[str]]) -> bool:
        """检查参数名是否存在于任何一个选定关键词关联的标准库的 model 列表中。"""
        for cat, kw in selected_keywords.items():
            if kw:
                keys = self.loader.keyword_to_keys.get(kw, [])
                for sk in keys:
                    if param_name in self.loader.standard_param_names.get(sk, set()):
                        return True # 只要在一个相关标准中找到就认为是相关的
        return False


# --- 主执行 / 测试块 (修改后) ---
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 模拟包含多个类别参数的标准化输入
    mock_standardized_params_combined = {
                "元件类型 (仪表名称 Inst. Name)": "热电阻",
                "元件数量 (类型 Type)": "单支",
                "连接螺纹 (温度元件型号 Therm. Element Model)": "缺失（文档未提供）",
                "元件类型 (分度号 Type)": "IEC标准 Pt100",
                "过程连接（法兰等级） (允差等级 Tolerance Error Rating)": "A级",
                "元件类型 (测量端形式 Meas. End Type)": "绝缘型",
                "铠套材质 (铠装材质 Armo. Mat'l)": "316",
                "铠套外径(d) (铠装直径 Armo. Dia. (mm))": "Ф6",
                "壳体代码 (接线盒形式 Terminal Box Style)": "防水型",
                "壳体代码 (接线盒材质 Terminal Box Mat'l)": "304",
                "接线口 (电气连接 Elec. Conn.)": "1/2\" NPT (F)",
                "过程连接（法兰等级） (防护等级 Enclosure Protection)": "IP65",
                "NEPSI (防爆等级 Explosion Proof)": "Exd II BT4",
                "TG套管形式 (套管形式 Well Type)": "整体钻孔锥形保护管",
                "TG套管形式 (套管材质 Well Mat'l)": "316",
                "过程连接（法兰等级） (压力等级 Pressure Rating)": "Class150",
                "铠套外径(d) (套管外径 Well Outside Dia. (mm))": "根部不大于28,套管厚度由供货商根据振动频率和强度计算确定",
                "过程连接 (过程连接形式 Process Conn.)": "固定法兰",
                "过程连接（法兰尺寸（Fs）） (连接规格Conn. Size)": "DN40",
                "过程连接 (法兰标准 Flange STD.)": "HG/T20615-2009",
                "过程连接（法兰等级） (等级 Rating)": "Class150",
                "法兰密封面形式 (法兰材质 Flange Mat'l)": "316",
                "法兰密封面形式 (密封面形式 Facing)": "RF",
                "过程连接 (制造厂 Manufacturer)": "缺失（文档未提供）",
                "内置指示器 (备注)": "缺失（文档未提供）",
                "过程连接（法兰等级） (操作/设计压力 Oper. Press. MPa(G))": "0.3/",
                "连接螺纹 (最大流速 Max. Velocity m/s)": "缺失（文档未提供）",
                "过程连接（法兰等级） (管嘴长度 Length mm)": "150",
                "过程连接（法兰等级） (插入深度 Well Length (mm))": "250",
                "连接螺纹 (测量范围 Meas. Range (°C))": "缺失（文档未提供）"
            }
    print(f"模拟组合输入 (标准化参数): \n{json.dumps(mock_standardized_params_combined, indent=2, ensure_ascii=False)}")

    print(f"测试使用的标准库根目录 (来自 settings): {settings.STANDARD_LIBS_DIR}")

    if not settings.STANDARD_LIBS_DIR.exists():
        print(f"\n错误：测试需要 '{settings.STANDARD_LIBS_DIR}' 目录。")
    else:
        generator = SpecCodeGenerator()
        print("\n--- 运行组合测试用例 ---")
        result_combined = generator.generate(mock_standardized_params_combined)

        if result_combined:
            recommended_code, recommendation_reason = result_combined
            print(f"\n推荐组合型号代码: {recommended_code}")
            # 预期格式: "YTA710... HR... TG..." (具体代码取决于匹配和排序)
            print(f"\n组合推荐理由:\n{recommendation_reason}")
        else:
            print("\n未能推荐组合型号代码或生成理由。请检查日志。")
