import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
import pandas as pd
import sys


project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import settings, prompts # 导入 prompts
    from src.standard_matcher.utils import calculate_string_similarity, sort_results_by_csv_order
    from src.utils.llm_utils import call_llm_for_match
except ImportError as e:
    print(
        f"错误 (matcher.py): 导入模块失败 - {e}。请检查项目结构和 PYTHONPATH。", file=sys.stderr)
    raise

logger = logging.getLogger(__name__)


class StandardLoader:

    def __init__(self):

        self.all_standards: Dict[str, pd.DataFrame] = {}
        self.standard_param_names: Dict[str, Set[str]] = {}
        self.standard_original_keyword: Dict[str, Optional[str]] = {}
        self.category_keyword_to_keys: Dict[str, Dict[str, List[str]]] = {}
        self.category_keyword_csv_order: Dict[str, Dict[str, List[str]]] = {}

    def load_all(self) -> bool:

        base_path = settings.STANDARD_LIBS_DIR
        index_path = base_path / "index.json"

        if not index_path.is_file():
            logger.error(f"主索引文件未找到: {index_path}")
            return False

        logger.info(f"开始从主索引 {index_path} 加载所有标准库...")
        loaded_count = 0
        self.all_standards.clear()
        self.standard_param_names.clear()
        self.standard_original_keyword.clear()
        self.category_keyword_to_keys.clear()
        self.category_keyword_csv_order.clear()

        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            logger.debug(f"已加载主索引: {index_path.name}")

            if not isinstance(index_data, dict):
                logger.error(f"主索引文件格式错误，根元素必须是字典: {index_path.name}")
                return False

            for category, category_data in index_data.items():
                if not isinstance(category_data, dict):
                    logger.warning(
                        f"索引 '{index_path.name}' 中的分类 '{category}' 的值不是字典，跳过。")
                    continue

                self.category_keyword_to_keys.setdefault(category, {})
                self.category_keyword_csv_order.setdefault(category, {})

                for keyword, relative_csv_files in category_data.items():
                    if not isinstance(relative_csv_files, list):
                        logger.warning(
                            f"索引 '{index_path.name}' -> '{category}' -> '{keyword}' 的值不是列表，跳过。")
                        continue

                    self.category_keyword_to_keys[category].setdefault(
                        keyword, [])
                    self.category_keyword_csv_order[category].setdefault(
                        keyword, [])

                    for relative_csv in relative_csv_files:
                        if not isinstance(relative_csv, str):
                            logger.warning(
                                f"索引 '{index_path.name}' -> '{category}' -> '{keyword}' 中的文件名不是字符串，跳过: {relative_csv}")
                            continue

                        csv_path = base_path / Path(relative_csv)
                        standard_key = f"{category}/{csv_path.stem}"

                        if self._load_single_standard(standard_key, csv_path, keyword):
                            loaded_count += 1
                            self.category_keyword_to_keys[category][keyword].append(
                                standard_key)
                            self.category_keyword_csv_order[category][keyword].append(
                                relative_csv)
                        else:
                            logger.warning(
                                f"未能加载标准 '{standard_key}' (关键词: {keyword})，将不会包含在映射中。")

        except Exception as e:
            logger.error(
                f"处理主索引文件 '{index_path.name}' 时出错: {e}", exc_info=True)
            return False

        if loaded_count > 0:
            logger.info(f"成功加载 {loaded_count} 个标准 CSV 文件。")
            logger.debug(
                f"构建的分类-关键词到标准键映射: {json.dumps(self.category_keyword_to_keys, indent=2, ensure_ascii=False)}")
            logger.debug(
                f"构建的分类-关键词到 CSV 顺序映射: {json.dumps(self.category_keyword_csv_order, indent=2, ensure_ascii=False)}")
            return True
        else:
            logger.error("未能成功加载任何标准 CSV 文件 (检查索引和 CSV 文件是否存在且格式正确)。")
            return False

    def _load_single_standard(self, standard_key: str, csv_path: Path, original_keyword: Optional[str]) -> bool:

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
            self.standard_param_names[standard_key] = set(
                df['model'].unique()) - {''}
            self.standard_original_keyword[standard_key] = original_keyword
            logger.debug(
                f"已加载标准 '{standard_key}' (原始关键词: {original_keyword}, {len(df)} 行)")
            return True
        except Exception as e:
            logger.error(
                f"加载 CSV 文件 '{csv_path.name}' 时出错: {e}", exc_info=True)
            return False

    def find_candidate_rows(self, standard_param_name: str, target_df: pd.DataFrame) -> pd.DataFrame:

        if target_df is None or target_df.empty:
            logger.debug(
                f"目标 DataFrame 为空或未提供，无法为 '{standard_param_name}' 查找候选行。")
            return pd.DataFrame()

        logger.debug(
            f"在当前 DataFrame 中为参数 '{standard_param_name}' 查找候选行 (策略: 精确 -> 模糊 -> LLM)...")
        if 'model' not in target_df.columns:
            logger.error("目标 DataFrame 中缺少 'model' 列。无法查找候选行。")
            return pd.DataFrame()

        exact_matches = target_df[target_df['model'] == standard_param_name]
        if not exact_matches.empty:
            logger.info(
                f"找到 {len(exact_matches)} 个精确匹配行 (参数名: '{standard_param_name}')。")
            return exact_matches
        else:
            logger.debug(f"参数 '{standard_param_name}' 未找到精确匹配项，尝试模糊匹配...")

        unique_models = target_df['model'].dropna(
        ).unique()
        best_fuzzy_score = -1.0
        best_fuzzy_models = []

        for model_in_df in unique_models:
            if not isinstance(model_in_df, str) or not model_in_df:
                continue
            score = calculate_string_similarity(
                standard_param_name, model_in_df)
            if score > best_fuzzy_score:
                best_fuzzy_score = score
                best_fuzzy_models = [model_in_df]
            elif score == best_fuzzy_score:
                best_fuzzy_models.append(model_in_df)

        threshold = settings.FUZZY_MATCH_THRESHOLD
        if best_fuzzy_score >= threshold:
            logger.info(
                f"找到 {len(best_fuzzy_models)} 个模糊匹配模型 (得分: {best_fuzzy_score:.4f} >= {threshold})，模型: {best_fuzzy_models}。")
            fuzzy_matches = target_df[target_df['model'].isin(
                best_fuzzy_models)]
            logger.debug(f"返回 {len(fuzzy_matches)} 行模糊匹配结果。")
            return fuzzy_matches
        else:
            logger.debug(
                f"模糊匹配最高得分 {best_fuzzy_score:.4f} 低于阈值 {threshold}，尝试 LLM 匹配...")

        candidate_model_names = [
            m for m in unique_models if isinstance(m, str) and m]
        if not candidate_model_names:
            logger.warning(
                f"参数 '{standard_param_name}': 模糊匹配失败，且无有效候选模型名称用于 LLM 匹配。")
            return pd.DataFrame()

        logger.debug(
            f"准备调用 LLM 从 {len(candidate_model_names)} 个候选模型中为 '{standard_param_name}' 选择最佳匹配...")

        # 使用 config 中的 prompt
        system_prompt = prompts.LLM_MODEL_NAME_MATCH_SYSTEM_PROMPT
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
                logger.info(
                    f"LLM 选择了最佳匹配模型名称: '{best_match_model_llm}' (针对参数 '{standard_param_name}')。")
                llm_matches = target_df[target_df['model']
                                        == best_match_model_llm]
                logger.debug(f"返回 {len(llm_matches)} 行 LLM 匹配结果。")
                return llm_matches
            else:
                logger.warning(
                    f"LLM 返回的模型名称 '{best_match_model_llm}' 不在候选列表中，忽略。")
                return pd.DataFrame()
        else:
            logger.warning(
                f"LLM 未能为参数 '{standard_param_name}' 选择最佳匹配模型名称。LLM 结果: {llm_result}")
            return pd.DataFrame()


class SpecCodeMatcher:

    def select_best_match(self, standard_param_value: str, candidate_rows: pd.DataFrame) -> Optional[Dict[str, Any]]:

        if candidate_rows.empty:
            logger.debug("没有候选行可供选择最佳匹配。")
            return None

        best_item_dict: Optional[Dict[str, Any]] = None
        best_score = -1.0
        match_field = None

        logger.debug(
            f"为值 '{standard_param_value}' 从 {len(candidate_rows)} 个候选中选择最佳匹配...")

        for index, row in candidate_rows.iterrows():
            for field in ('description', 'code', 'model', 'remark'):
                if field in row:
                    text_in_df = str(row[field])
                    score = calculate_string_similarity(
                        standard_param_value, text_in_df)

                    if score > best_score:
                        best_score = score
                        best_item_dict = row.to_dict()
                        match_field = field
                        logger.debug(
                            f"  * 新的最佳匹配: 行索引 {index}, 字段 '{field}', 得分 {score:.4f}")

        threshold = settings.FUZZY_MATCH_THRESHOLD
        if best_score >= threshold and best_item_dict is not None:
            logger.debug(
                f"选择最佳匹配完成。最佳得分 {best_score:.4f} (来自字段 '{match_field}') >= 阈值 {threshold}。")
            return best_item_dict
        elif best_score < threshold:
            if best_item_dict is not None:
                logger.warning(
                    f"模糊匹配得分 {best_score:.4f} 低于阈值 {threshold} (来自字段 '{match_field}')。尝试 LLM 检索...")
            else:
                if best_score <= 0:
                    logger.warning(
                        f"未能为值 '{standard_param_value}' 在候选行中找到模糊匹配项。尝试 LLM 检索...")

            candidate_dicts = candidate_rows.to_dict('records')

            # 使用 config 中的 prompt
            system_prompt = prompts.LLM_VALUE_MATCH_SYSTEM_PROMPT
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
                    logger.info(
                        f"LLM 选择了最佳匹配行 (来自值 '{standard_param_value}'). Code: '{best_match_dict_llm.get('code')}'")
                    return best_match_dict_llm
                else:
                    logger.warning(
                        f"LLM 返回的最佳匹配字典缺少 'code' 字段: {best_match_dict_llm}")
                    return None
            else:
                logger.warning(
                    f"LLM 未能为值 '{standard_param_value}' 选择最佳匹配行。LLM 结果: {llm_result}")
                return None
        else:
            logger.debug(f"未能为值 '{standard_param_value}' 在候选行中找到任何模糊匹配项。")
            return None


class SpecCodeGenerator:

    def __init__(self):

        self.loader = StandardLoader()
        self.code_matcher = SpecCodeMatcher()
        self.main_model_param_names = {
            'transmitter': ['温度变送器', '变送器型号'],
            'sensor': ['传感器类型', '热电阻类型', '热电偶类型'],
            'tg': ['保护管类型', '套管类型']
        }

    def _find_best_keyword_set(self, standardized_params: Dict[str, str]) -> Optional[Tuple[str, str, List[str]]]:

        if not self.loader.category_keyword_to_keys:
            logger.error("分类-关键词到标准键的映射为空，无法选择。请确保 StandardLoader 已成功加载。")
            return None

        if not standardized_params:
            logger.error("输入参数字典为空，无法进行关键词匹配。")
            return None

        logger.info(f"开始根据输入参数 {standardized_params} 选择最佳匹配类别和关键词...")

        for category, main_names in self.main_model_param_names.items():
            for main_name in main_names:
                if main_name in standardized_params:
                    param_value = standardized_params[main_name]
                    if category in self.loader.category_keyword_to_keys:
                        category_keywords = self.loader.category_keyword_to_keys[category]
                        if param_value in category_keywords and param_value != "默认":
                            associated_keys = category_keywords[param_value]
                            if associated_keys:
                                logger.info(
                                    f"通过主要型号参数 '{main_name}' = '{param_value}' 直接匹配到类别 '{category}' 的关键词 '{param_value}'。")
                                return category, param_value, associated_keys
                            else:
                                logger.warning(
                                    f"直接匹配到关键词 '{param_value}' 但其关联的标准键列表为空。")
                        else:
                            logger.debug(
                                f"参数值 '{param_value}' (来自 '{main_name}') 不是类别 '{category}' 下的特定关键词。")
                    else:
                        logger.warning(
                            f"参数名 '{main_name}' 指向的类别 '{category}' 在加载的索引中不存在。")

        logger.info("未能通过主要型号参数直接匹配关键词，继续进行类别推断和模糊匹配...")

        inferred_category: Optional[str] = None
        best_category_score = -1.0

        for category, keywords_map in self.loader.category_keyword_to_keys.items():
            category_score = 0.0
            param_count_for_category = 0

            category_hint_found = False
            if category in self.main_model_param_names:
                for main_name in self.main_model_param_names[category]:
                    if main_name in standardized_params:
                        category_hint_found = True
                        break

            value_similarity_score = 0.0
            specific_keyword_count = 0
            candidate_specific_keywords = [
                k for k in keywords_map.keys() if k != "默认"]

            if candidate_specific_keywords:
                total_max_similarity = 0.0
                for keyword in candidate_specific_keywords:
                    max_score_for_keyword = 0.0
                    for param_name, param_value in standardized_params.items():
                        score = calculate_string_similarity(
                            keyword, param_value)
                        if score > max_score_for_keyword:
                            max_score_for_keyword = score
                    total_max_similarity += max_score_for_keyword
                value_similarity_score = total_max_similarity / \
                    len(candidate_specific_keywords)
                specific_keyword_count = len(candidate_specific_keywords)

            if category_hint_found:
                category_score = 0.6 + (value_similarity_score * 0.4)
                logger.debug(
                    f"  类别 '{category}': 找到参数名提示，基础分 0.6 + 值相似度 {value_similarity_score:.4f} * 0.4 = {category_score:.4f}")
            elif specific_keyword_count > 0:
                category_score = value_similarity_score
                logger.debug(
                    f"  类别 '{category}': 无参数名提示，得分来自值相似度 = {category_score:.4f}")
            else:
                logger.debug(f"  类别 '{category}': 无参数名提示且无特定关键词，无法评分。")
                category_score = 0.0

            if category_score > best_category_score:
                best_category_score = category_score
                inferred_category = category

        if not inferred_category:
            logger.error("未能根据输入参数推断出产品类别。")
            return None

        logger.info(
            f"推断出的最可能类别: '{inferred_category}' (综合得分: {best_category_score:.4f})")

        best_specific_keyword: Optional[str] = None
        best_specific_score = -1.0
        category_keywords = self.loader.category_keyword_to_keys.get(
            inferred_category, {})
        candidate_specific_keywords = [
            k for k in category_keywords.keys() if k != "默认"]

        if not candidate_specific_keywords:
            logger.warning(f"类别 '{inferred_category}' 中没有定义特定的关键词，将直接尝试默认。")
        else:
            logger.debug(
                f"在类别 '{inferred_category}' 中搜索特定关键词 (基于值匹配): {candidate_specific_keywords}")
            for keyword in candidate_specific_keywords:
                max_score_for_keyword = 0.0
                for param_name, param_value in standardized_params.items():
                    score = calculate_string_similarity(keyword, param_value)
                    if score > max_score_for_keyword:
                        max_score_for_keyword = score
                logger.debug(
                    f"  特定关键词 '{keyword}' 的最高值匹配得分: {max_score_for_keyword:.4f}")
                if max_score_for_keyword > best_specific_score:
                    best_specific_score = max_score_for_keyword
                    best_specific_keyword = keyword

            threshold = settings.FUZZY_MATCH_THRESHOLD
            fuzzy_match_found = False
            if best_specific_keyword and best_specific_score >= threshold:
                logger.info(
                    f"通过值模糊匹配在类别 '{inferred_category}' 中找到高分特定关键词: '{best_specific_keyword}' (得分: {best_specific_score:.4f} >= {threshold})")
                fuzzy_match_found = True
            else:
                logger.warning(
                    f"类别 '{inferred_category}' 中的特定关键词值模糊匹配得分过低 (最高 {best_specific_score:.4f} < {threshold})。")
                best_specific_keyword = None

            llm_match_found = False
            if not fuzzy_match_found:
                logger.info(f"尝试 LLM 匹配类别 '{inferred_category}' 的特定关键词...")
                # 使用 config 中的 prompt
                system_prompt = prompts.LLM_SPECIFIC_KEYWORD_MATCH_SYSTEM_PROMPT
                user_prompt = f"""
                 输入参数 (名称和值):
                 {json.dumps(standardized_params, ensure_ascii=False, indent=2)}

                 候选特定关键词列表 (类别: {inferred_category}):
                 {json.dumps(candidate_specific_keywords, ensure_ascii=False, indent=2)}

                 根据输入参数的名称和值，从候选列表中选择最相关的一个关键词。
                 """
                llm_result = call_llm_for_match(system_prompt, user_prompt)

                if llm_result and "best_match_keyword" in llm_result and isinstance(llm_result["best_match_keyword"], str):
                    best_keyword_llm = llm_result["best_match_keyword"]
                    if best_keyword_llm in candidate_specific_keywords:
                        logger.info(
                            f"LLM 在类别 '{inferred_category}' 中选择了特定关键词: '{best_keyword_llm}'")
                        best_specific_keyword = best_keyword_llm
                        llm_match_found = True
                    else:
                        logger.warning(
                            f"LLM 返回的关键词 '{best_keyword_llm}' 不在类别 '{inferred_category}' 的候选特定关键词列表中，忽略 LLM 结果。")
                else:
                    logger.error(
                        f"LLM 未能为类别 '{inferred_category}' 选择特定关键词。LLM 结果: {llm_result}")

            final_specific_keyword = None
            if fuzzy_match_found or llm_match_found:
                final_specific_keyword = best_specific_keyword
                logger.info(
                    f"最终确定的特定关键词: '{final_specific_keyword}' (来自 {'模糊匹配' if fuzzy_match_found else 'LLM'})")
                associated_keys = category_keywords.get(
                    final_specific_keyword, [])
                if associated_keys:
                    logger.debug(
                        f"关键词 '{final_specific_keyword}' 关联的标准键: {associated_keys}")
                    return inferred_category, final_specific_keyword, associated_keys
                else:
                    logger.error(
                        f"确定了特定关键词 '{final_specific_keyword}' 但未能获取其关联的标准键列表。")

        if final_specific_keyword is None:
            logger.warning(
                f"未能在类别 '{inferred_category}' 中找到合适的特定关键词，尝试使用 '默认' 配置...")
            default_keyword = "默认"
            if default_keyword in category_keywords:
                default_associated_keys = category_keywords.get(
                    default_keyword, [])
                if default_associated_keys:
                    logger.info(f"成功找到并使用类别 '{inferred_category}' 的 '默认' 配置。")
                    logger.debug(f"'默认' 关键词关联的标准键: {default_associated_keys}")
                    return inferred_category, default_keyword, default_associated_keys
                else:
                    logger.error(
                        f"类别 '{inferred_category}' 中存在 '默认' 关键词，但未能获取其关联的标准键列表。无法使用默认配置。")
                    return None
            else:
                logger.error(
                    f"类别 '{inferred_category}' 中未定义 '默认' 关键词，且无特定关键词匹配，无法确定标准库。")
                return None

            logger.error(f"代码逻辑异常：未能为关键词 '{final_specific_keyword}' 返回结果。") # 保留旧方法以备后用或调试
        return None

    def _find_best_keyword_llm(self, category: str, standardized_params: Dict[str, str]) -> str:
        """使用 LLM 为指定类别选择最佳关键词"""
        logger.info(f"开始为类别 '{category}' 使用 LLM 查找最佳关键词...")

        if category not in self.loader.category_keyword_to_keys:
            logger.error(f"类别 '{category}' 不在已加载的索引中，无法查找关键词。")
            return "默认" # 或者抛出异常？

        category_keywords_map = self.loader.category_keyword_to_keys[category]
        candidate_keywords = list(category_keywords_map.keys())

        if not candidate_keywords:
            logger.warning(f"类别 '{category}' 没有定义任何关键词，将使用 '默认'。")
            return "默认"

        if len(candidate_keywords) == 1 and candidate_keywords[0] == "默认":
             logger.info(f"类别 '{category}' 只有一个 '默认' 关键词，直接选用。")
             return "默认"

        # 准备 LLM 调用
        system_prompt = prompts.LLM_KEYWORD_SELECTION_SYSTEM_PROMPT # 使用 config 中的 prompt
        user_prompt = f"""
        输入参数 (名称和值):
        {json.dumps(standardized_params, ensure_ascii=False, indent=2)}

        候选关键词列表 (类别: {category}):
        {json.dumps(candidate_keywords, ensure_ascii=False, indent=2)}

        根据输入参数的名称和值，从候选列表中选择最相关的一个关键词。如果无法确定，请选择 "默认"。
        """

        llm_result = call_llm_for_match(system_prompt, user_prompt)

        selected_keyword = "默认" # 默认值

        if llm_result and "best_match_keyword" in llm_result and isinstance(llm_result["best_match_keyword"], str):
            best_keyword_llm = llm_result["best_match_keyword"]
            if best_keyword_llm in candidate_keywords:
                logger.info(f"LLM 在类别 '{category}' 中选择了关键词: '{best_keyword_llm}'")
                selected_keyword = best_keyword_llm
            else:
                logger.warning(f"LLM 返回的关键词 '{best_keyword_llm}' 不在类别 '{category}' 的候选关键词列表中，将使用 '默认'。")
        else:
            logger.error(f"LLM 未能为类别 '{category}' 选择关键词或返回格式错误。LLM 结果: {llm_result}。将使用 '默认'。")

        logger.info(f"类别 '{category}' 最终选定的关键词: '{selected_keyword}'")
        return selected_keyword

    def _generate_code_for_category(self, category: str, selected_keyword: str, associated_standard_keys: List[str], standardized_params: Dict[str, str]) -> Tuple[str, Dict[str, Optional[Dict[str, Any]]]]:
        """为指定类别和关键词生成代码段"""
        logger.info(f"--- 开始为类别 '{category}' (关键词: '{selected_keyword}') 生成代码段 ---")
        logger.debug(f"使用标准库: {associated_standard_keys}")

        product_code_parts: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        final_matches: Dict[str, Optional[Dict[str, Any]]] = {}
        matched_param_names_in_category = set() # 跟踪此类别内已匹配的参数

        # 过滤参数，只处理与当前类别相关的（这是一个优化，但可能复杂且易错，暂时匹配所有参数）
        # relevant_params = {k: v for k, v in standardized_params.items() if k not in previously_matched_params}
        relevant_params = standardized_params # 简化：暂时对所有参数进行匹配尝试

        for std_param_name, std_param_value in relevant_params.items():
            # 注意：这里不再跳过已匹配参数，因为不同类别可能需要匹配相同的参数名
            # if std_param_name in matched_param_names_in_category:
            #     logger.debug(f"参数 '{std_param_name}' 已在此类别 '{category}' 中匹配，跳过。")
            #     continue
            if not std_param_name or not std_param_value:
                logger.debug(f"跳过空参数: 名称='{std_param_name}', 值='{std_param_value}'")
                continue

            logger.debug(f"  尝试匹配参数: '{std_param_name}' = '{std_param_value}'")
            found_match_for_param = False
            best_match_row_dict: Optional[Dict[str, Any]] = None

            for standard_key in associated_standard_keys:
                logger.debug(f"    在标准 '{standard_key}' 中查找...")
                target_df = self.loader.all_standards.get(standard_key)
                if target_df is None or target_df.empty:
                    logger.debug(f"    标准 '{standard_key}' 的 DataFrame 为空或未加载，跳过。")
                    continue

                # 优先检查参数名是否是唯一的 model 值 (简化逻辑，可能需要调整)
                if 'model' in target_df.columns:
                    model_column = target_df['model']
                    if std_param_name in model_column.values:
                        count = (model_column == std_param_name).sum()
                        if count == 1:
                            logger.debug(f"    参数名 '{std_param_name}' 在标准 '{standard_key}' 中是唯一的 model，直接采用该行。")
                            unique_row_df = target_df[model_column == std_param_name]
                            best_match_row_dict = unique_row_df.iloc[0].to_dict()
                            best_match_row_dict['_source_standard_key'] = standard_key
                            found_match_for_param = True
                            break # 找到唯一 model 匹配，不再在此标准库内查找其他匹配

                # 如果不是唯一 model，则进行正常的行查找和值匹配
                if not found_match_for_param:
                    candidate_rows_df = self.loader.find_candidate_rows(std_param_name, target_df)
                    if not candidate_rows_df.empty:
                        current_best_match = self.code_matcher.select_best_match(std_param_value, candidate_rows_df)
                        if current_best_match:
                            # 比较是否比之前跨标准库找到的更好？(暂时不处理，找到第一个就用)
                            best_match_row_dict = current_best_match
                            best_match_row_dict['_source_standard_key'] = standard_key
                            logger.debug(f"    在 '{standard_key}' 中为参数 '{std_param_name}' 通过值匹配找到匹配。")
                            found_match_for_param = True
                            break # 在当前标准库找到匹配，跳到下一个参数

            # 处理参数匹配结果
            if found_match_for_param and best_match_row_dict:
                # 检查此参数是否已在此类别中匹配过，如果新的匹配更好则替换？(简化：不替换，记录第一个)
                if std_param_name not in final_matches:
                    final_matches[std_param_name] = best_match_row_dict
                    code_part = best_match_row_dict.get('code')
                    if pd.notna(code_part) and str(code_part).strip():
                        code_value = str(code_part).strip()
                        product_code_parts.append((code_value, best_match_row_dict))
                        matched_param_names_in_category.add(std_param_name) # 标记在此类别中匹配
                        logger.debug(f"  参数 '{std_param_name}' 的匹配代码: '{code_value}' (来自: {best_match_row_dict.get('_source_standard_key', '未知')})")
                    else:
                        logger.warning(f"  参数 '{std_param_name}' 匹配到的代码为空或无效 ('{code_part}')，将跳过。")
                        final_matches[std_param_name] = None # 标记为尝试过但无有效代码
                else:
                     logger.debug(f"  参数 '{std_param_name}' 已在此类别中匹配过，忽略来自 '{best_match_row_dict.get('_source_standard_key', '未知')}' 的新匹配。")

            elif std_param_name not in final_matches: # 仅当此参数之前未被记录时才标记为未匹配
                logger.debug(f"  参数 '{std_param_name}' 未能在类别 '{category}' 的关联标准库中找到合适的标准代码。")
                final_matches[std_param_name] = None # 标记为尝试过但未找到匹配

        # --- 代码组装 ---
        logger.info(f"开始为类别 '{category}' 组装代码...")

        # 获取第一个 CSV 的顺序信息
        first_csv_key: Optional[str] = None
        first_csv_model_order: List[str] = []
        first_csv_model_to_index: Dict[str, int] = {}
        if associated_standard_keys:
            first_csv_key = associated_standard_keys[0] # 使用列表中的第一个作为主顺序文件
            first_csv_df = self.loader.all_standards.get(first_csv_key)
            if first_csv_df is not None and 'model' in first_csv_df.columns:
                seen_models = set()
                for model_name in first_csv_df['model']:
                    if model_name and model_name not in seen_models:
                        first_csv_model_order.append(model_name)
                        seen_models.add(model_name)
                first_csv_model_to_index = {model: i for i, model in enumerate(first_csv_model_order)}
                logger.debug(f"  第一个 CSV ('{first_csv_key}') 的 model 顺序: {first_csv_model_order}")
            else:
                logger.error(f"  无法加载第一个 CSV ('{first_csv_key}') 或其缺少 'model' 列，无法按其顺序排序。")
                # 返回空代码和空匹配详情？
                return "", {}
        else:
            logger.error(f"  类别 '{category}' 没有关联的标准键，无法确定第一个 CSV 文件。")
            return "", {}

        # 按第一个 CSV 的 model 顺序排列代码
        final_code_sequence: List[Optional[str]] = [None] * len(first_csv_model_order)
        subsequent_codes_with_details: List[Tuple[str, Dict[str, Any]]] = []
        matched_first_csv_models: Set[str] = set()

        for code, details in product_code_parts:
            if details and '_source_standard_key' in details:
                source_key = details['_source_standard_key']
                matched_model = details.get('model')

                if not matched_model:
                    logger.warning(f"  匹配到的代码 '{code}' (来自 {source_key}) 缺少 'model' 信息，放入后续列表。")
                    subsequent_codes_with_details.append((code, details))
                    continue

                if source_key == first_csv_key:
                    if matched_model in first_csv_model_to_index:
                        idx = first_csv_model_to_index[matched_model]
                        if final_code_sequence[idx] is None:
                            final_code_sequence[idx] = code
                            matched_first_csv_models.add(matched_model)
                            logger.debug(f"    代码 '{code}' (model: {matched_model}) 放入首个 CSV 序列位置 {idx}")
                        else:
                            logger.warning(f"    位置 {idx} (model: {matched_model}) 已被代码 '{final_code_sequence[idx]}' 占据，新的代码 '{code}' 将被忽略或放入后续。")
                            # 考虑是否应该放入后续？暂时忽略重复位置的代码
                            # subsequent_codes_with_details.append((code, details))
                    else:
                        logger.warning(f"    代码 '{code}' 的 model '{matched_model}' (来自首个 CSV {source_key}) 未在预期的 model 顺序中找到，将放入后续列表。")
                        subsequent_codes_with_details.append((code, details))
                else:
                    logger.debug(f"    代码 '{code}' (model: {matched_model}, 来自 {source_key}) 放入后续列表。")
                    subsequent_codes_with_details.append((code, details))
            else:
                logger.warning(f"  代码 '{code}' 缺少来源信息，放入后续列表。")
                subsequent_codes_with_details.append((code, details if details else {}))

        # 填充占位符
        placeholder_count = 0
        for i, model_name in enumerate(first_csv_model_order):
            if model_name not in matched_first_csv_models:
                if final_code_sequence[i] is None:
                    final_code_sequence[i] = '?' # 使用 '?' 作为占位符
                    placeholder_count += 1
                    logger.debug(f"    为首个 CSV 中未匹配的 model '{model_name}' 在位置 {i} 插入占位符 '?'")
        if placeholder_count > 0:
            logger.info(f"  为第一个 CSV 中 {placeholder_count} 个未匹配的 model 插入了占位符。")

        # 对后续代码按 CSV 文件顺序排序
        category_csv_orders = self.loader.category_keyword_csv_order.get(category, {})
        csv_order_list_for_keyword = category_csv_orders.get(selected_keyword, [])
        logger.debug(f"  用于后续代码排序的 CSV 顺序 (类别: {category}, 关键词: {selected_keyword}): {csv_order_list_for_keyword}")

        def get_subsequent_sort_key(item: Tuple[str, Dict[str, Any]]) -> int:
            code, details = item
            if details and '_source_standard_key' in details:
                source_key = details['_source_standard_key']
                # 从 source_key (如 'transmitter/YTA610_addon') 提取相对路径
                relative_path_parts = source_key.split('/', 1)
                if len(relative_path_parts) == 2:
                    relative_path_stem = relative_path_parts[1] # 'YTA610_addon'
                    # 在 csv_order_list_for_keyword (如 ['transmitter/YTA610.csv', 'transmitter/YTA610_addon.csv']) 中查找
                    for i, csv_relative_path in enumerate(csv_order_list_for_keyword):
                        if Path(csv_relative_path).stem == relative_path_stem:
                            return i # 返回在顺序列表中的索引
                logger.warning(f"    无法确定代码 '{code}' (来源: {source_key}) 在 CSV 顺序列表中的位置，将排在后面。")
                return len(csv_order_list_for_keyword) # 排在最后
            logger.warning(f"    代码 '{code}' 缺少来源信息，将排在最后。")
            return len(csv_order_list_for_keyword) + 1 # 绝对排在最后

        sorted_subsequent_codes_with_details = sorted(subsequent_codes_with_details, key=get_subsequent_sort_key)
        sorted_subsequent_codes = [code for code, details in sorted_subsequent_codes_with_details]
        logger.debug(f"  排序后的后续代码部分: {sorted_subsequent_codes}")

        # 拼接最终代码
        final_sequence_str = [c if c is not None else '?' for c in final_code_sequence]
        category_code = "".join(final_sequence_str) + "".join(sorted_subsequent_codes)

        if not category_code.replace('?', ''):
            logger.warning(f"未能为类别 '{category}' 生成任何有效代码部分 (只有占位符或完全为空)。")
            category_code = "" # 返回空字符串表示失败

        logger.info(f"--- 类别 '{category}' 代码段生成完成: '{category_code}' ---")
        return category_code, final_matches


    def generate(self, standardized_params: Dict[str, str]) -> Optional[Tuple[str, str]]:
        """
        根据输入的标准化参数，为 transmitter, sensor, tg 三个类别生成组合型号代码。
        使用 LLM 为每个类别选择最佳关键词。
        """
        logger.info("--- 开始组合型号代码生成 (LLM 关键词选择) ---")
        if not standardized_params:
            logger.error("输入参数为空，无法生成型号代码。")
            return None

        # 确保标准库已加载
        if not self.loader.all_standards:
            logger.info("标准库尚未加载，尝试加载...")
            if not self.loader.load_all():
                logger.error("加载标准库失败，无法继续。")
                return None
            logger.info("标准库加载成功。")

        categories_to_process = ["transmitter", "sensor", "tg"]
        combined_code_parts: Dict[str, str] = {}
        combined_matches: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {}
        combined_keywords: Dict[str, str] = {}
        combined_source_keys: Dict[str, List[str]] = {}

        for category in categories_to_process:
            # 1. 使用 LLM 选择关键词
            selected_keyword = self._find_best_keyword_llm(category, standardized_params)
            combined_keywords[category] = selected_keyword

            # 2. 获取关联的标准库键
            category_keywords_map = self.loader.category_keyword_to_keys.get(category, {})
            associated_standard_keys = category_keywords_map.get(selected_keyword, [])
            if not associated_standard_keys:
                 # 如果特定关键词没有关联文件，尝试获取'默认'的
                 if selected_keyword != "默认":
                     logger.warning(f"关键词 '{selected_keyword}' (类别: {category}) 没有关联的标准库，尝试使用 '默认'。")
                     associated_standard_keys = category_keywords_map.get("默认", [])
                     if associated_standard_keys:
                         combined_keywords[category] = "默认" # 更新使用的关键词
                     else:
                         logger.error(f"类别 '{category}' 的关键词 '{selected_keyword}' 和 '默认' 都没有关联的标准库。无法处理此类别。")
                         combined_code_parts[category] = "" # 标记为空表示失败
                         combined_matches[category] = {}
                         combined_source_keys[category] = []
                         continue # 处理下一个类别
                 else:
                     logger.error(f"类别 '{category}' 的 '默认' 关键词没有关联的标准库。无法处理此类别。")
                     combined_code_parts[category] = ""
                     combined_matches[category] = {}
                     combined_source_keys[category] = []
                     continue

            combined_source_keys[category] = associated_standard_keys

            # 3. 为当前类别生成代码段
            category_code, category_matches = self._generate_code_for_category(
                category, combined_keywords[category], associated_standard_keys, standardized_params
            )
            combined_code_parts[category] = category_code
            combined_matches[category] = category_matches

        # 4. 拼接最终代码 (按 transmitter sensor tg 顺序)
        final_code = " ".join(combined_code_parts.get(cat, "") for cat in categories_to_process)
        final_code = final_code.strip() # 去除可能的首尾空格

        if not final_code.replace('?', '').replace(' ', ''):
             logger.error("所有类别都未能生成有效代码部分，无法生成最终组合型号代码。")
             return None

        logger.info(f"--- 组合型号代码生成完成 ---")
        logger.info(f"各部分代码: {combined_code_parts}")
        logger.info(f"最终组合代码: {final_code}")

        # 5. 生成推荐理由
        recommendation_reason = self._generate_recommendation_reason_combined(
            standardized_params, combined_matches, final_code, combined_keywords, combined_source_keys
        )

        return final_code, recommendation_reason

    def _generate_recommendation_reason_combined(self,
                                                 user_requirements: Dict[str, str],
                                                 all_matched_details: Dict[str, Dict[str, Optional[Dict[str, Any]]]],
                                                 recommended_code: str,
                                                 keywords_used: Dict[str, str],
                                                 source_keys_used: Dict[str, List[str]]) -> str:
        """为组合型号代码生成推荐理由"""
        logger.info("开始生成组合推荐理由...")
        try:
            reason_parts = []
            categories_processed = list(all_matched_details.keys())

            # 构建每个类别的匹配详情字符串
            category_details_str = ""
            all_unmatched_params = set(user_requirements.keys()) # Start with all params

            for category in categories_processed:
                matched_details = all_matched_details.get(category, {})
                keyword = keywords_used.get(category, "未知")
                source_keys = source_keys_used.get(category, [])
                category_relevant_details = []
                category_unmatched_params = [] # Params attempted but failed for this category

                category_details_str += f"\n--- {category.capitalize()} 部分 (关键词: '{keyword}', 标准库: {source_keys}) ---\n"

                processed_params_for_category = set()
                for param, details in matched_details.items():
                    processed_params_for_category.add(param)
                    all_unmatched_params.discard(param) # Remove from global unmatched if matched anywhere
                    req_value = user_requirements.get(param, "N/A")
                    if details:
                        detail_str = f"  参数 '{param}': 需求 '{req_value}', 匹配值 '{details.get('description', 'N/A')}' (代码: {details.get('code', 'N/A')}, 来源: {details.get('_source_standard_key', '未知')})"
                        if details.get('remark'):
                            detail_str += f", 备注: {details['remark']}"
                        category_relevant_details.append(detail_str)
                    else:
                        # This param was attempted for this category but failed
                        category_unmatched_params.append(f"'{param}' (需求值: '{req_value}')")


                if category_relevant_details:
                     category_details_str += "\n".join(category_relevant_details) + "\n"
                else:
                     category_details_str += "  无成功匹配的规格。\n"

                if category_unmatched_params:
                    category_details_str += f"  未能匹配的参数 (此部分尝试过): {', '.join(category_unmatched_params)}\n"

                # Check for params not even attempted for this category (maybe relevant?)
                # This logic might be too complex/unreliable for now.

            # Final list of params that were never matched in any category
            final_unmatched_list = [f"'{p}' (需求值: '{user_requirements[p]}')" for p in all_unmatched_params]


            # 构建 LLM Prompt
            # 动态填充 system prompt 中的关键词示例 (可选，但更精确)
            system_prompt = prompts.LLM_COMBINED_REASON_SYSTEM_PROMPT.format(
                transmitter_keyword=keywords_used.get('transmitter', '默认')
                # 可以为 sensor 和 tg 添加类似格式化
            )

            # 填充 user prompt 模板
            user_prompt = prompts.LLM_COMBINED_REASON_USER_PROMPT_TEMPLATE.format(
                user_requirements_json=json.dumps(user_requirements, indent=2, ensure_ascii=False),
                category_details_str=category_details_str,
                final_unmatched_list_str=', '.join(final_unmatched_list) if final_unmatched_list else "所有参数均在至少一个部分中成功匹配或找到对应代码。",
                recommended_code=recommended_code
            )

            # 调用 LLM
            llm_response = call_llm_for_match(system_prompt, user_prompt, expect_json=False)

            # 处理 LLM 响应
            reason = "未能生成有效的推荐理由。"
            if isinstance(llm_response, str):
                reason = llm_response.strip()
                if reason:
                    logger.info(f"成功生成组合推荐理由 (纯文本): {reason[:100]}...")
                else:
                    logger.warning("LLM 返回了空的组合推荐理由字符串。")
                    reason = "未能生成有效的推荐理由 (LLM 返回空)。"
            elif isinstance(llm_response, dict) and "error" in llm_response:
                logger.error(f"LLM 生成组合推荐理由时返回错误字典: {llm_response}")
                reason = f"无法生成推荐理由：LLM 调用出错 ({llm_response.get('error', '未知错误')}: {llm_response.get('details', '无详情')})"
            elif llm_response is None:
                logger.error("LLM 生成组合推荐理由调用未返回任何结果 (返回 None)。")
                reason = "无法生成推荐理由：LLM 调用未返回结果。"
            else:
                logger.error(f"LLM 生成组合推荐理由时返回了意外的类型: {type(llm_response)} - {llm_response}")
                reason = f"无法生成推荐理由：LLM 返回类型无效 ({type(llm_response)})。"

            return reason

        except Exception as e:
            logger.error(f"生成组合推荐理由时发生异常: {e}", exc_info=True)
            return f"无法生成推荐理由：内部错误 ({e})"


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    mock_standardized_params_tx = {
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
    print(
        f"模拟输入 (标准化参数): \n{json.dumps(mock_standardized_params_tx, indent=2, ensure_ascii=False)}")

    print(f"测试使用的标准库根目录 (来自 settings): {settings.STANDARD_LIBS_DIR}")

    if not settings.STANDARD_LIBS_DIR.exists():
        print(f"\n错误：测试需要 '{settings.STANDARD_LIBS_DIR}' 目录及其内容。")
        print("请确保该目录存在于项目结构中。")
    else:
        generator = SpecCodeGenerator()

        print("\n--- 运行测试用例 ---")
        result_tx = generator.generate(mock_standardized_params_tx)
        if result_tx:
            recommended_code, recommendation_reason = result_tx
            print(f"\n推荐型号代码: {recommended_code}")
            print(f"\n推荐理由:\n{recommendation_reason}")
        else:
            print("\n未能推荐型号代码或生成理由。请检查日志获取详细信息。")
