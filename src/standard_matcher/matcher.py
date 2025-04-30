# new_sensor_project/src/standard_matcher/matcher.py
import json
import logging
import os  # 新增导入 os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
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
# 类 1: 标准库加载器 (修改后)
# ==============================================================================


class StandardLoader:
    """
    负责加载所有标准库 CSV 文件，并维护主型号到关联标准库的映射。
    """

    def __init__(self):
        """初始化 StandardLoader。"""
        # 存储所有加载的标准 DataFrame，键是标准的唯一标识符 (例如 'category/csv_name_stem')
        self.all_standards: Dict[str, pd.DataFrame] = {}
        # 存储每个标准 DataFrame 中定义的参数名集合
        self.standard_param_names: Dict[str, Set[str]] = {}
        # 存储每个标准键关联的主型号名称
        self.standard_main_models: Dict[str, Optional[str]] = {}
        # 新增：存储主型号到其关联的标准键列表的映射
        self.main_model_to_keys: Dict[str, List[str]] = {}
        # 新增：存储 index.json 中定义的 CSV 文件顺序，用于后续排序
        self.main_model_csv_order: Dict[str, List[str]] = {}

    def load_all(self) -> bool:
        """
        加载 libs/standard/index.json 定义的所有标准库 CSV 文件。
        并构建主型号到标准键列表的映射。

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
        # 清空旧数据，确保每次调用都重新加载
        self.all_standards.clear()
        self.standard_param_names.clear()
        self.standard_main_models.clear()
        self.main_model_to_keys.clear()
        self.main_model_csv_order.clear()

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

                # 遍历分类下的主型号
                for main_model, relative_csv_files in category_data.items():
                    if not isinstance(relative_csv_files, list):
                        logger.warning(
                            f"索引 '{index_path.name}' -> '{category}' -> '{main_model}' 的值不是列表，跳过。")
                        continue

                    # 初始化该主型号的列表
                    self.main_model_to_keys[main_model] = []
                    # 存储原始 CSV 文件名用于排序
                    self.main_model_csv_order[main_model] = []

                    # 遍历主型号关联的 CSV 文件列表
                    for relative_csv in relative_csv_files:
                        if not isinstance(relative_csv, str):
                            logger.warning(
                                f"索引 '{index_path.name}' -> '{category}' -> '{main_model}' 中的文件名不是字符串，跳过: {relative_csv}")
                            continue

                        # 构建完整的 CSV 文件路径 (相对于 base_path)
                        # relative_csv 已经是 'category/filename.csv' 格式
                        csv_path = base_path / Path(relative_csv)
                        # 使用 'category/filename_stem' 作为标准键
                        standard_key = f"{category}/{csv_path.stem}"

                        # 加载单个 CSV
                        if self._load_single_standard(standard_key, csv_path, main_model):
                            loaded_count += 1
                            # 将有效的 standard_key 添加到主型号的映射列表中
                            self.main_model_to_keys[main_model].append(
                                standard_key)
                            self.main_model_csv_order[main_model].append(
                                relative_csv)  # 存储原始相对路径文件名
                        else:
                            logger.warning(
                                f"未能加载标准 '{standard_key}' (主型号: {main_model})，将不会包含在映射中。")

        except Exception as e:
            logger.error(
                f"处理主索引文件 '{index_path.name}' 时出错: {e}", exc_info=True)
            return False # 索引加载失败，直接返回 False

        if loaded_count > 0:
            logger.info(f"成功加载 {loaded_count} 个标准 CSV 文件。")
            logger.debug(f"构建的主型号到标准键映射: {self.main_model_to_keys}")
            logger.debug(f"构建的主型号到 CSV 顺序映射: {self.main_model_csv_order}")
            return True
        else:
            logger.error("未能成功加载任何标准 CSV 文件 (检查索引和 CSV 文件是否存在且格式正确)。")
            return False

    def _load_single_standard(self, standard_key: str, csv_path: Path, main_model: Optional[str]) -> bool:
        """
        加载单个 CSV 文件并存储其信息，包括关联的主型号。
        (代码基本不变)
        """
        try:
            if not csv_path.is_file():
                logger.warning(f"标准库 CSV 文件未找到，跳过: {csv_path}")
                return False
            df = pd.read_csv(csv_path, dtype=str)
            df.fillna('', inplace=True)  # 用空字符串填充 NaN

            if 'model' not in df.columns:
                logger.error(f"CSV 文件 '{csv_path.name}' 缺少必需的 'model' 列，无法加载。")
                return False

            self.all_standards[standard_key] = df
            # 提取 'model' 列的唯一值作为该标准的参数名集合
            self.standard_param_names[standard_key] = set(
                df['model'].unique()) - {''}  # 移除空字符串
            # 存储关联的主型号
            self.standard_main_models[standard_key] = main_model
            logger.debug(
                f"已加载标准 '{standard_key}' (主型号: {main_model}, {len(df)} 行)")
            return True
        except Exception as e:
            logger.error(
                f"加载 CSV 文件 '{csv_path.name}' 时出错: {e}", exc_info=True)
            return False

    def find_candidate_rows(self, standard_param_name: str, target_df: pd.DataFrame) -> pd.DataFrame:
        """
        在 *目标* DataFrame 中查找与标准参数名称匹配的候选行。
        采用策略：精确匹配 -> 模糊匹配 -> LLM 匹配。
        """
        if target_df is None or target_df.empty:
            logger.debug(
                f"目标 DataFrame 为空或未提供，无法为 '{standard_param_name}' 查找候选行。")
            return pd.DataFrame()

        logger.debug(
            f"在当前 DataFrame 中为参数 '{standard_param_name}' 查找候选行 (策略: 精确 -> 模糊 -> LLM)...")
        if 'model' not in target_df.columns:
            logger.error("目标 DataFrame 中缺少 'model' 列。无法查找候选行。")
            return pd.DataFrame()

        # 1. 精确匹配 'model' 列
        exact_matches = target_df[target_df['model'] == standard_param_name]
        if not exact_matches.empty:
            logger.info(
                f"找到 {len(exact_matches)} 个精确匹配行 (参数名: '{standard_param_name}')。")
            return exact_matches
        else:
            logger.debug(f"参数 '{standard_param_name}' 未找到精确匹配项，尝试模糊匹配...")

        # 2. 模糊匹配 'model' 列
        unique_models = target_df['model'].dropna(
        ).unique()  # 获取所有非空的唯一 model 值
        best_fuzzy_score = -1.0
        best_fuzzy_models = []

        for model_in_df in unique_models:
            if not isinstance(model_in_df, str) or not model_in_df:  # 跳过非字符串或空字符串
                continue
            score = calculate_string_similarity(
                standard_param_name, model_in_df)
            if score > best_fuzzy_score:
                best_fuzzy_score = score
                best_fuzzy_models = [model_in_df]  # 重置最佳列表
            elif score == best_fuzzy_score:
                best_fuzzy_models.append(model_in_df)  # 添加得分相同的模型

        threshold = settings.FUZZY_MATCH_THRESHOLD
        if best_fuzzy_score >= threshold:
            logger.info(
                f"找到 {len(best_fuzzy_models)} 个模糊匹配模型 (得分: {best_fuzzy_score:.4f} >= {threshold})，模型: {best_fuzzy_models}。")
            # 返回所有最佳模糊匹配模型对应的行
            fuzzy_matches = target_df[target_df['model'].isin(
                best_fuzzy_models)]
            logger.debug(f"返回 {len(fuzzy_matches)} 行模糊匹配结果。")
            return fuzzy_matches
        else:
            logger.debug(
                f"模糊匹配最高得分 {best_fuzzy_score:.4f} 低于阈值 {threshold}，尝试 LLM 匹配...")

        # 3. LLM 匹配 'model' 列
        candidate_model_names = [
            m for m in unique_models if isinstance(m, str) and m]  # 过滤后的候选列表
        if not candidate_model_names:
            logger.warning(
                f"参数 '{standard_param_name}': 模糊匹配失败，且无有效候选模型名称用于 LLM 匹配。")
            return pd.DataFrame()

        # TODO: 如果 candidate_model_names 对于 LLM 提示来说太大，则实现限制其大小的逻辑
        logger.debug(
            f"准备调用 LLM 从 {len(candidate_model_names)} 个候选模型中为 '{standard_param_name}' 选择最佳匹配...")

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
            if best_match_model_llm in candidate_model_names:  # 确保 LLM 返回的是候选列表中的一个
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
            return pd.DataFrame()  # 如果 LLM 失败或未找到匹配项，则返回空 DataFrame


# ==============================================================================
# 类 2: 规格代码匹配器 (代码不变)
# ==============================================================================

class SpecCodeMatcher:
    """
    根据标准参数值从候选行中选择最佳匹配行。
    使用模糊匹配比较 'description'、'code' 和 'model' 字段。
    """

    def select_best_match(self, standard_param_value: str, candidate_rows: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        根据标准参数值从候选行中选择最佳匹配行。
        (代码不变)
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
                # 只有在完全没有模糊匹配时才记录这个警告，避免重复
                if best_score <= 0:  # 确保是完全没找到任何相似的
                    logger.warning(
                        f"未能为值 '{standard_param_value}' 在候选行中找到模糊匹配项。尝试 LLM 检索...")

            # 3. LLM 后备方案，用于根据值选择最佳行
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
                return None  # 如果 LLM 失败或未找到匹配项，则返回 None
        else:  # best_score == -1.0 and best_item_dict is None
            logger.debug(f"未能为值 '{standard_param_value}' 在候选行中找到任何模糊匹配项。")
            return None


# ==============================================================================
# 类 3: 型号代码生成器 (重大修改)
# ==============================================================================

class SpecCodeGenerator:
    """
    协调推荐最终产品型号代码及生成推荐理由的过程。
    使用 StandardLoader 和 SpecCodeMatcher。
    能够处理与主型号关联的多个标准库文件。
    """

    def __init__(self):
        """初始化 SpecCodeGenerator。"""
        self.loader = StandardLoader()
        self.code_matcher = SpecCodeMatcher()
        # 假设主型号参数的键是 '温度变送器'，可以根据实际情况调整或配置化
        self.main_model_param_key = '温度变送器'  # TODO: 考虑从配置加载

    def _find_best_main_model(self, standardized_params: Dict[str, str]) -> Optional[Tuple[str, List[str]]]:
        """
        根据输入参数名称和主型号值，选择最佳匹配的主型号及其关联的标准键列表。

        Args:
            standardized_params: 标准化参数字典 {standard_param_name: standard_param_value}。

        Returns:
            Optional[Tuple[str, List[str]]]:
                包含最佳主型号名称和关联标准键列表的元组，
                如果未找到合适的主型号则为 None。
        """
        if not self.loader.main_model_to_keys:
            logger.error("主型号到标准键的映射为空，无法选择最佳主型号。请确保 StandardLoader 已成功加载。")
            return None

        input_param_names = set(standardized_params.keys())
        if not input_param_names:
            logger.error("输入参数名称集合为空，无法选择最佳主型号。")
            return None

        # 提取输入的主型号值
        input_main_model_value = standardized_params.get(
            self.main_model_param_key)
        logger.info(
            f"开始为输入参数 {input_param_names} (主型号值: {input_main_model_value}) 选择最佳匹配主型号...")

        best_score = -1.0
        best_main_model: Optional[str] = None
        best_standard_keys: List[str] = []

        # 主型号匹配的加权分数
        MAIN_MODEL_MATCH_BONUS = 1000

        # 遍历所有已识别的主型号及其关联的标准键
        for main_model, standard_keys in self.loader.main_model_to_keys.items():
            if not standard_keys:  # 如果某个主型号没有关联的 key，跳过
                continue

            # 收集该主型号下所有标准库定义的参数名称
            combined_standard_names = set()
            for key in standard_keys:
                combined_standard_names.update(
                    self.loader.standard_param_names.get(key, set()))

            if not combined_standard_names:
                logger.debug(f"主型号 '{main_model}' 关联的标准库没有有效的参数名称，跳过。")
                continue

            # 1. 计算基础得分 (参数名称交集)
            # 只考虑非主型号参数的交集，因为主型号参数用于 bonus
            non_main_model_input_names = input_param_names - \
                {self.main_model_param_key}
            intersection = non_main_model_input_names.intersection(
                combined_standard_names)
            base_score = len(intersection)

            # 2. 计算主型号匹配加分
            model_bonus = 0
            if input_main_model_value and main_model and input_main_model_value == main_model:
                model_bonus = MAIN_MODEL_MATCH_BONUS
                logger.debug(
                    f"    主型号匹配! ('{input_main_model_value}' == '{main_model}')，加分 {model_bonus}")
            # 如果输入没有主型号，但标准库有，给一个较小的 bonus？或者不给？当前不给。

            # 3. 计算总分
            total_score = base_score + model_bonus

            logger.debug(
                f"  - 主型号 '{main_model}' (关联键: {standard_keys}): 参数名 {combined_standard_names}, 交集 {intersection} (基础分 {base_score}), 主型号加分 {model_bonus}, 总分 {total_score}")

            # 更新最佳匹配
            if total_score > best_score:
                best_score = total_score
                best_main_model = main_model
                best_standard_keys = standard_keys  # 存储关联的标准键列表

        if best_main_model and best_standard_keys:
            logger.info(
                f"选择的最佳匹配主型号: '{best_main_model}' (关联标准键: {best_standard_keys}, 得分: {best_score})")
            return best_main_model, best_standard_keys
        else:
            logger.error(f"未能为输入参数找到合适的匹配主型号。最高得分: {best_score}")
            return None

    def generate(self, standardized_params: Dict[str, str]) -> Optional[Tuple[str, str]]:
        """
        根据标准化参数推荐产品型号代码并生成推荐理由。
        能够处理与主型号关联的多个标准库文件。

        Args:
            standardized_params: 标准化参数字典 {standard_param_name: standard_param_value}。

        Returns:
            Optional[Tuple[str, str]]: 包含推荐型号代码和推荐理由的元组 (code, reason)，
                                       如果发生错误则为 None。
        """
        logger.info("--- 开始型号代码推荐与理由生成 (多文件处理逻辑) ---")
        if not standardized_params:
            logger.error("输入参数为空，无法推荐型号代码。")
            return None

        # 1. 加载所有标准库 (确保映射已构建)
        if not self.loader.load_all():
            # 错误已在 load_all 中记录
            return None

        # 2. 根据输入参数选择最佳匹配的主型号及其关联的标准键列表
        best_model_result = self._find_best_main_model(standardized_params)
        if not best_model_result:
            # 错误已在 _find_best_main_model 中记录
            return None
        selected_main_model, associated_standard_keys = best_model_result

        # 3. 初始化代码列表，将主型号作为第一部分
        # 存储 (code, match_details)
        product_code_parts: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        if selected_main_model and selected_main_model.strip():
            # 主型号本身没有 match_details，用 None 占位
            product_code_parts.append((selected_main_model.strip(), None))
            logger.info(f"已添加基础型号代码: '{selected_main_model.strip()}'")
        else:
            logger.warning("未能确定有效的基础型号代码，最终代码可能不完整。")
            # 即使没有主型号，也继续尝试匹配其他参数

        # 4. 遍历 *所有* 输入参数 (除了主型号)，在关联的标准键列表中查找匹配
        # 存储最终匹配结果 {param_name: match_details}
        final_matches: Dict[str, Optional[Dict[str, Any]]] = {}
        matched_param_names = set()  # 跟踪已成功匹配的参数名

        logger.info(
            f"开始在主型号 '{selected_main_model}' 的关联标准库 {associated_standard_keys} 中匹配其余输入参数...")

        for std_param_name, std_param_value in standardized_params.items():
            # 跳过主型号参数
            if std_param_name == self.main_model_param_key:
                logger.debug(f"跳过主型号参数 '{std_param_name}' 的匹配。")
                continue
            # 跳过空的参数名或值
            if not std_param_name or not std_param_value:
                logger.debug(
                    f"跳过空参数: 名称='{std_param_name}', 值='{std_param_value}'")
                continue
            # 如果参数已匹配，跳过 (防止重复添加代码)
            if std_param_name in matched_param_names:
                logger.debug(f"参数 '{std_param_name}' 已在之前的标准库中匹配，跳过。")
                continue

            logger.debug(
                f"--- 尝试匹配参数: '{std_param_name}' = '{std_param_value}' ---")
            found_match_for_param = False
            best_match_row_dict: Optional[Dict[str, Any]] = None

            # 遍历与主型号关联的 *所有* 标准键 (CSV 文件)
            for standard_key in associated_standard_keys:
                logger.debug(f"  在标准 '{standard_key}' 中查找...")
                target_df = self.loader.all_standards.get(standard_key)
                if target_df is None or target_df.empty:
                    logger.debug(
                        f"  标准 '{standard_key}' 的 DataFrame 为空或未加载，跳过。")
                    continue

                # a. 查找候选行 (基于标准参数名称，在当前 DF 内精确匹配)
                candidate_rows_df = self.loader.find_candidate_rows(
                    std_param_name, target_df)

                if not candidate_rows_df.empty:
                    # b. 从候选者中选择最佳匹配 (基于标准参数值)
                    current_best_match = self.code_matcher.select_best_match(
                        std_param_value, candidate_rows_df)

                    if current_best_match:
                        best_match_row_dict = current_best_match
                        # 记录匹配来源的标准键，可能用于排序或调试
                        best_match_row_dict['_source_standard_key'] = standard_key
                        logger.debug(
                            f"  在 '{standard_key}' 中为参数 '{std_param_name}' 找到匹配。")
                        found_match_for_param = True
                        break  # 找到匹配后，停止在此参数的其他标准库中查找

            # c. 处理匹配结果
            if found_match_for_param and best_match_row_dict:
                final_matches[std_param_name] = best_match_row_dict  # 存储完整匹配信息
                code_part = best_match_row_dict.get('code')
                if pd.notna(code_part) and str(code_part).strip():
                    code_value = str(code_part).strip()
                    # 存储代码和匹配详情，用于后续排序和生成理由
                    product_code_parts.append(
                        (code_value, best_match_row_dict))
                    matched_param_names.add(std_param_name)  # 标记为已匹配
                    logger.debug(
                        f"参数 '{std_param_name}' 的匹配代码: '{code_value}' (来自: {best_match_row_dict.get('_source_standard_key', '未知')})")
                else:
                    logger.warning(
                        f"参数 '{std_param_name}' 匹配到的代码为空或无效 ('{code_part}')，将跳过。")
                    final_matches[std_param_name] = None  # 标记为未找到有效代码
            else:
                logger.warning(f"参数 '{std_param_name}' 未能在任何关联的标准库中找到合适的标准代码。")
                final_matches[std_param_name] = None  # 标记为未找到匹配

        # 5. 排序附加代码 (基于 index.json 中 CSV 文件的顺序)
        # 提取基础型号代码 (第一个元素)
        base_model_code = product_code_parts[0][0] if product_code_parts else ""
        # 提取附加代码及其来源信息 (从第二个元素开始)
        additional_codes_with_details = product_code_parts[1:]

        # 获取用于排序的 CSV 文件顺序列表
        csv_order_list = self.loader.main_model_csv_order.get(
            selected_main_model, [])
        logger.debug(
            f"用于排序的 CSV 文件顺序 (主型号: {selected_main_model}): {csv_order_list}")

        # 定义排序函数
        def get_sort_key(item: Tuple[str, Optional[Dict[str, Any]]]) -> int:
            code, details = item
            if details and '_source_standard_key' in details:
                # e.g., 'transmitter/YTA710_addon'
                source_key = details['_source_standard_key']
                # 从 source_key 提取原始 CSV 文件名 (需要去掉目录前缀和可能的 _stem)
                parts = source_key.split('/', 1)
                if len(parts) == 2:
                    stem = parts[1]
                    # 尝试在 csv_order_list 中找到对应的原始文件名
                    for csv_file in csv_order_list:
                        if Path(csv_file).stem == stem:
                            try:
                                index = csv_order_list.index(csv_file)
                                logger.debug(
                                    f"  排序: 代码 '{code}' (来自 {source_key}, 对应 {csv_file}) 的索引为 {index}")
                                return index
                            except ValueError:
                                pass  # 文件名不在顺序列表中
                # 如果无法确定顺序，放到最后
                logger.warning(
                    f"无法确定代码 '{code}' (来自 {source_key}) 的排序位置，将置于末尾。")
                return len(csv_order_list)  # 放到最后
            logger.warning(f"代码 '{code}' 缺少来源信息，无法排序，将置于末尾。")
            return len(csv_order_list)  # 没有来源信息，放到最后

        # 对附加代码进行排序
        sorted_additional_codes_with_details = sorted(
            additional_codes_with_details, key=get_sort_key)
        sorted_additional_codes = [
            code for code, details in sorted_additional_codes_with_details]

        logger.debug(f"排序后的附加代码: {sorted_additional_codes}")

        # 6. 组装推荐型号代码 (拼接基础型号和排序后的附加代码)
        if not base_model_code and not sorted_additional_codes:
            logger.error("未能生成任何代码部分，无法生成最终型号代码。")
            return None
        elif not sorted_additional_codes and base_model_code:
            logger.warning("只生成了基础型号代码，没有匹配到任何附加选项代码。")

        # 如果需要分隔符，在这里添加
        recommended_code = base_model_code + \
            "".join(sorted_additional_codes)  # 直接拼接
        logger.info(f"--- 型号代码推荐完成 ---")
        logger.info(f"推荐型号代码: {recommended_code}")

        # 7. 生成推荐理由 (使用 final_matches 获取所有参数的匹配信息)
        recommendation_reason = self._generate_recommendation_reason(
            standardized_params, final_matches, recommended_code, selected_main_model, associated_standard_keys)

        return recommended_code, recommendation_reason

    def _generate_recommendation_reason(self, user_requirements: Dict[str, str], matched_details: Dict[str, Optional[Dict[str, Any]]], recommended_code: str, selected_main_model: str, source_keys: List[str]) -> str:
        """
        调用 LLM 生成推荐理由。 (增加了 selected_main_model 和 source_keys 参数, 处理未匹配参数)

        Args:
            user_requirements: 用户输入的标准化参数字典。
            matched_details: 匹配到的每个参数的详细信息字典 (None 表示未匹配)。
            recommended_code: 最终推荐的型号代码。
            selected_main_model: 选定的主型号名称。
            source_keys: 用于匹配的标准库键列表。

        Returns:
            str: LLM 生成的推荐理由，如果失败则返回默认提示信息。
        """
        logger.info("开始生成推荐理由...")
        try:
            # 准备 LLM 输入
            relevant_details = []
            unmatched_params = []
            # 遍历原始需求，检查匹配结果
            for param, req_value in user_requirements.items():
                # 跳过主型号参数本身
                if param == self.main_model_param_key:
                    continue

                details = matched_details.get(param)  # 获取匹配结果
                if details:
                    # 提取关键信息用于生成理由
                    detail_str = f"参数 '{param}': 需求 '{req_value}', 匹配值 '{details.get('description', 'N/A')}' (代码: {details.get('code', 'N/A')}, 来源: {details.get('_source_standard_key', '未知')})"
                    if details.get('remark'):
                        detail_str += f", 备注: {details['remark']}"
                    relevant_details.append(detail_str)
                elif param in matched_details:  # 存在于 matched_details 但值为 None，表示尝试过但未匹配
                    unmatched_params.append(f"'{param}' (需求值: '{req_value}')")
                # else: 参数不在 matched_details 中，可能因为是空值被跳过，不计入未匹配

            system_prompt = f"""
            你是一位专业的工业自动化产品选型顾问，尤其擅长温度变送器、温度传感器和TG保护套管。
            你的任务是根据用户提供的需求参数、系统匹配到的产品规格细节（包括来源标准库）、未匹配的参数（如有）以及最终推荐的型号代码，生成一段简洁、专业且易于理解的推荐理由。
            重点突出推荐型号（基于主型号 '{selected_main_model}'）的关键特性如何满足了用户的核心需求。
            如果存在未匹配的参数，请在理由中提及，并说明当前推荐是基于已匹配参数的最佳选择。
            语言风格应专业、客观、自信。避免口语化表达。
            直接输出推荐理由文本，不要包含任何额外的前缀或解释性文字 (例如不要说 "推荐理由如下：")。
            使用中文回答。
            """
            user_prompt = f"""
            用户需求 (标准化参数):
            {json.dumps(user_requirements, indent=2, ensure_ascii=False)}

            系统匹配到的规格细节 (来自标准库: {source_keys}):
            {chr(10).join(relevant_details) if relevant_details else "无成功匹配的附加规格。"}

            未能匹配的参数:
            {', '.join(unmatched_params) if unmatched_params else "所有参数均成功匹配或找到对应代码。"}

            最终推荐型号代码: {recommended_code}

            请基于以上信息，为型号 '{selected_main_model}' 生成推荐理由：
            """
            # 调用 LLM，明确告知不需要 JSON 格式
            llm_response = call_llm_for_match(
                system_prompt, user_prompt, expect_json=False)

            reason = "未能生成有效的推荐理由。"  # 默认值

            if isinstance(llm_response, str):
                reason = llm_response.strip()
                if reason:
                    logger.info(f"成功生成推荐理由 (纯文本): {reason[:100]}...")
                else:
                    logger.warning("LLM 返回了空的推荐理由字符串。")
                    reason = "未能生成有效的推荐理由 (LLM 返回空)。"
            elif isinstance(llm_response, dict) and "error" in llm_response:
                logger.error(f"LLM 生成推荐理由时返回错误字典: {llm_response}")
                reason = f"无法生成推荐理由：LLM 调用出错 ({llm_response.get('error', '未知错误')}: {llm_response.get('details', '无详情')})"
            elif llm_response is None:
                logger.error("LLM 生成推荐理由调用未返回任何结果 (返回 None)。")
                reason = "无法生成推荐理由：LLM 调用未返回结果。"
            else:
                logger.error(
                    f"LLM 生成推荐理由时返回了意外的类型: {type(llm_response)} - {llm_response}")
                reason = f"无法生成推荐理由：LLM 返回类型无效 ({type(llm_response)})。"

            return reason

        except Exception as e:
            logger.error(f"生成推荐理由时发生异常: {e}", exc_info=True)
            return f"无法生成推荐理由：内部错误 ({e})"


# --- 主执行 / 测试块 (保持不变) ---
if __name__ == '__main__':
    # 配置基本日志记录以进行测试
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 模拟标准化参数 - 仅保留变送器测试用例，并包含基础型号参数
    mock_standardized_params_tx = {
        '输出信号': '4~20mA DC BRAIN通信型',
        '温度变送器': 'YTA710',
        '说明书语言': '英语',
        '传感器输入': '双支输入',
        '壳体代码': '不锈钢',
        '接线口': '1/2 NPT 内螺纹',
        '内置指示器': '数字LCD',
        '安装支架': 'SUS304 2寸管道平装',
        'NEPSI': 'GB3836.1-2010、GB3836.4-2010、GB3836.20-2010、GB3836.19-2010、 GB12476.1-2013、GB12476.4-2010 '
    }
    print(
        f"模拟输入 (标准化参数): \n{json.dumps(mock_standardized_params_tx, indent=2, ensure_ascii=False)}")

    # 设置测试路径 (确保 settings.py 或环境变量指向正确的 libs 目录)
    # project_root 在文件顶部已定义
    # settings.STANDARD_LIBS_DIR = project_root / "libs" / "standard" # 通常在 settings.py 中配置
    print(f"测试使用的标准库根目录 (来自 settings): {settings.STANDARD_LIBS_DIR}")

    # 检查所需目录是否存在
    if not settings.STANDARD_LIBS_DIR.exists():
        print(f"\n错误：测试需要 '{settings.STANDARD_LIBS_DIR}' 目录及其内容。")
        print("请确保该目录存在于项目结构中。")
    else:
        # 实例化生成器并运行过程
        generator = SpecCodeGenerator()

        print("\n--- 运行测试用例 ---")
        result_tx = generator.generate(mock_standardized_params_tx)
        if result_tx:
            recommended_code, recommendation_reason = result_tx
            print(f"\n推荐型号代码: {recommended_code}")
            # 预期 YTA710 的代码 + 附加代码，例如 YTA710-ES2B2DD/S2/X2 (具体代码取决于 CSV 内容和顺序)
            print(f"\n推荐理由:\n{recommendation_reason}")
        else:
            print("\n未能推荐型号代码或生成理由。请检查日志获取详细信息。")
