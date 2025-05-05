import pandas as pd
import json
import logging
from typing import Dict, List, Any, Set

# 配置日志记录 (与项目其他部分保持一致)
# 建议在项目入口或配置文件中统一配置，这里为了独立运行添加
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义可跳过的 model 名称集合 (将来可以考虑移到配置)
SKIPPABLE_MODELS = {
    "变送器附加规格",
    "传感器附加规格",
    "套管附加规格"
}


def get_model_order(csv_list_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    根据CSV列表映射和指定规则确定各产品类型下model的排序顺序。

    规则:
    1. 产品顺序: transmitter -> sensor -> tg
    2. 每个产品类型只读取其CSV列表中的第一个文件来确定该类型的model顺序。
    3. CSV内部按'model'列从上到下的首次出现顺序排序。

    Args:
        csv_list_map: 产品类型到CSV文件路径列表的映射字典。
                       例如: {"transmitter": ["path1.csv", "path2.csv"], ...}

    Returns:
        一个字典，键是产品类型 ("transmitter", "sensor", "tg")，
        值是该产品类型下按顺序排列的唯一 model 名称列表。
        如果某个产品类型的文件读取失败或列不存在，则其条目可能为空列表。
    """
    ordered_models_by_product: Dict[str, List[str]] = {
        "transmitter": [], "sensor": [], "tg": []}
    # 用于跟踪全局已添加的model，确保跨产品类型的唯一性（如果需要）
    processed_models_globally: Set[str] = set()

    # 预定义的产品处理顺序
    product_order = ["transmitter", "sensor", "tg"]

    for product_type in product_order:
        # 确保字典中有该产品类型的键
        if product_type not in ordered_models_by_product:
            ordered_models_by_product[product_type] = []

        if product_type in csv_list_map and csv_list_map[product_type]:
            # 只取该产品类型的第一个CSV文件来确定顺序
            csv_path = csv_list_map[product_type][0]
            logger.info(f"正在处理产品 '{product_type}' 的CSV文件: {csv_path}")
            try:
                # 读取CSV文件，假设分隔符是逗号
                # 注意：根据实际CSV格式可能需要调整读取参数，例如 sep=';' 或编码 encoding='gbk'
                df = pd.read_csv(csv_path)  # 尝试默认逗号分隔符

                if 'model' not in df.columns:
                    # 如果没有 model 列，尝试分号分隔符 (常见于某些中文环境导出的CSV)
                    logger.warning(
                        f"文件 {csv_path} 未找到 'model' 列 (逗号分隔)，尝试使用分号分隔符...")
                    try:
                        df = pd.read_csv(csv_path, sep=';')
                    except Exception as sep_err:
                        logger.warning(f"尝试使用分号分隔符读取 {csv_path} 失败: {sep_err}")
                        # 可以添加更多尝试，例如指定编码
                        # df = pd.read_csv(csv_path, encoding='gbk')
                        # df = pd.read_csv(csv_path, sep=';', encoding='gbk')

                if 'model' in df.columns:
                    # 提取model列，去除NaN/空值，并转换成字符串以防数字等类型
                    models_in_csv = df['model'].dropna().astype(str).tolist()
                    # 获取当前CSV中唯一的model，并保持首次出现的顺序
                    models_in_this_csv_ordered = []
                    seen_in_this_csv = set()
                    for model in models_in_csv:
                        if model not in seen_in_this_csv:
                            models_in_this_csv_ordered.append(model)
                            seen_in_this_csv.add(model)

                    # 将这些 model 添加到对应产品类型的列表中
                    # 注意：这里不再检查全局唯一性，因为顺序是按产品类型内部确定的
                    # 如果需要全局唯一性检查，需要调整逻辑
                    ordered_models_by_product[product_type].extend(
                        models_in_this_csv_ordered)
                    logger.debug(
                        f"为产品 '{product_type}' 添加 models: {models_in_this_csv_ordered}")

                else:
                    logger.warning(
                        f"尝试多种方式后，CSV文件 '{csv_path}' 中仍未找到 'model' 列。")

            except FileNotFoundError:
                logger.error(f"错误：找不到CSV文件 '{csv_path}'。")
            except pd.errors.EmptyDataError:
                logger.warning(f"警告：CSV文件 '{csv_path}' 为空。")
            except Exception as e:
                logger.error(f"读取或处理CSV文件 '{csv_path}' 时出错: {e}")
        else:
            logger.warning(
                f"在CSV列表映射中未找到产品类型 '{product_type}' 或其列表为空，无法确定其 model 顺序。")

    logger.info(f"最终确定的各产品类型 model 排序: {ordered_models_by_product}")
    return ordered_models_by_product


def generate_final_code(csv_list_map: Dict[str, List[str]], selected_codes_data: Dict[str, Dict[str, Any]]) -> str:
    """
    根据确定的各产品类型model顺序，从selected_codes_data中提取代码，
    将同一产品类型的代码连接，不同产品类型的代码块用空格分隔。

    Args:
        csv_list_map: 产品类型到CSV文件路径列表的映射字典。
        selected_codes_data: 从 code_selector.py 输出的 JSON 结构，
                             格式为 {"参数名": {"model": "模型名", "code": "代码", ...}}。

    Returns:
        最终拼接成的产品型号代码字符串，格式如 "产品型号生成：transmitter_code_block sensor_code_block tg_code_block"。
        如果无法生成代码，则返回错误信息。
    """
    logger.info("开始生成最终产品代码...")
    logger.debug(f"接收到的CSV列表映射: {csv_list_map}")
    logger.debug(f"接收到的已选代码数据: {selected_codes_data}")

    # 1. 获取按产品类型分组的 model 顺序
    model_order_by_product = get_model_order(csv_list_map)

    if not any(model_order_by_product.values()):  # 检查是否所有产品类型的 model 列表都为空
        logger.error("未能确定任何产品类型的 model 排序顺序，无法生成代码。")
        return "产品型号生成失败：无法确定任何排序。"

    # 2. 增强预加载：查找默认代码 (is_default='1')，并记录所有代码
    # model_details_map: { model_name: {"default_code": str | None, "all_codes": List[str], "product_type": str} }
    model_details_map: Dict[str, Dict[str, Any]] = {}
    product_order_for_defaults = ["transmitter",
                                  "sensor", "tg"]  # 与 get_model_order 一致

    for product_type_default in product_order_for_defaults:
        if product_type_default in csv_list_map and csv_list_map[product_type_default]:
            csv_path_default = csv_list_map[product_type_default][0]
            logger.info(
                # 日志信息更新
                f"正在为产品 '{product_type_default}' 从 {csv_path_default} 预加载模型详情...")
            try:
                # 尝试读取CSV，处理分隔符问题
                df_default = None
                try:
                    df_default = pd.read_csv(csv_path_default)
                except Exception:
                    pass  # 稍后检查列是否存在

                if df_default is None or 'model' not in df_default.columns:
                    try:
                        df_default = pd.read_csv(csv_path_default, sep=';')
                    except Exception as sep_err_default:
                        logger.warning(
                            f"无法读取 {csv_path_default} 的 'model' 列 (尝试逗号和分号后)，跳过为 {product_type_default} 生成模型详情。错误: {sep_err_default}")
                        continue  # 跳过此产品类型的默认代码生成

                # 确保 'model', 'code', 'is_default' 列存在
                if df_default is not None and all(col in df_default.columns for col in ['model', 'code', 'is_default']):
                    # 清理数据：去除 model 或 code 为空的行，并将 model, code, is_default 转为字符串
                    required_cols = ['model', 'code', 'is_default']
                    df_cleaned_default = df_default[required_cols].dropna(
                        subset=['model', 'code'])  # is_default 可以为空? 假设不为空
                    df_cleaned_default = df_cleaned_default.astype(
                        str)  # 全部转为字符串

                    # 获取每个 model 对应的所有 code 值
                    model_to_all_codes = df_cleaned_default.groupby(
                        'model')['code'].apply(list).to_dict()

                    # 查找每个 model 的默认 code (is_default == '1')
                    model_to_default_code = {}
                    default_rows = df_cleaned_default[df_cleaned_default['is_default'] == '1']
                    # 使用 drop_duplicates 确保每个 model 只取一个默认值（如果CSV中有多个标记为1）
                    model_to_default_code = default_rows.drop_duplicates(
                        subset=['model']).set_index('model')['code'].to_dict()

                    # 遍历此CSV中的唯一 model
                    for model_name_default in df_cleaned_default['model'].unique():
                        if model_name_default not in model_details_map:  # 避免被后续产品类型覆盖
                            all_codes_for_model = model_to_all_codes.get(
                                model_name_default, [])
                            default_code_for_model = model_to_default_code.get(
                                model_name_default)  # 获取默认 code

                            details = {
                                # 存储默认 code (可能是 None)
                                "default_code": default_code_for_model,
                                "all_codes": all_codes_for_model,
                                "product_type": product_type_default  # 记录来源产品类型
                            }
                            # 存储详细信息
                            model_details_map[model_name_default] = details

                            if default_code_for_model:
                                logger.debug(
                                    f"为 model '{model_name_default}' 找到默认代码: {default_code_for_model}")
                            else:
                                logger.debug(
                                    f"Model '{model_name_default}' 未找到标记为默认 (is_default='1') 的代码。所有代码: {all_codes_for_model}")

                else:
                    missing_cols = {'model', 'code', 'is_default'} - \
                        set(df_default.columns if df_default is not None else [])
                    logger.warning(
                        f"CSV文件 '{csv_path_default}' 缺少必需列: {missing_cols}，无法为 {product_type_default} 生成模型详情。")

            except FileNotFoundError:
                logger.error(
                    f"找不到CSV文件 '{csv_path_default}'，无法为 {product_type_default} 生成模型详情。")
            except pd.errors.EmptyDataError:
                logger.warning(
                    f"CSV文件 '{csv_path_default}' 为空，无法为 {product_type_default} 生成模型详情。")
            except Exception as e:
                logger.error(f"处理CSV文件 '{csv_path_default}' 以生成模型详情时出错: {e}")

    logger.info(f"预加载的模型详情映射完成: {len(model_details_map)} 个条目")  # 日志信息更新
    logger.debug(f"模型详情映射内容: {model_details_map}")  # 日志信息更新

    # 3. 构建一个从 model 名称到 code 的快速查找字典 (来自 selected_codes_data)
    model_to_code_map = {}
    found_models_in_selection = set()
    for param_data in selected_codes_data.values():
        if isinstance(param_data, dict) and 'model' in param_data and 'code' in param_data:
            model_name = str(param_data['model'])  # 确保是字符串
            code_value = str(param_data['code']) if pd.notna(
                param_data['code']) else ''  # 处理可能的 NaN 或 None
            model_to_code_map[model_name] = code_value
            found_models_in_selection.add(model_name)
        else:
            logger.warning(
                f"selected_codes_data 中的条目格式不符合预期或缺少 model/code: {param_data}")

    logger.debug(
        f"从 selected_codes_data 构建的 model->code 映射: {model_to_code_map}")
    logger.debug(
        f"在 selected_codes_data 中找到的 models: {found_models_in_selection}")

    # --- 新增：1. 预先计算条件 ---
    has_tg_product = 'tg' in csv_list_map and bool(csv_list_map.get('tg'))
    has_sensor_product = 'sensor' in csv_list_map and bool(csv_list_map.get('sensor'))
    tg_csv_path = csv_list_map.get('tg', [None])[0] # 获取第一个tg csv路径，如果不存在则为None
    specific_tg_csvs = {'libs/standard/tg/TG_PT-1.csv', 'libs/standard/tg/TG_PT-2.csv', 'libs/standard/tg/TG_PT-3.csv'}
    is_specific_tg_csv = tg_csv_path in specific_tg_csvs
    logger.info(f"规则条件检查: has_tg={has_tg_product}, has_sensor={has_sensor_product}, is_specific_tg_csv={is_specific_tg_csv} (path: {tg_csv_path})")

    # SKIPPABLE_MODELS 已在文件顶部定义，无需在此重复

    # 4. 按产品顺序处理并拼接代码块
    product_code_strings = []
    product_order = ["transmitter", "sensor", "tg"]  # 预定义的产品处理顺序
    missing_models_log = {}  # 记录每个产品类型在 selected_codes_data 中缺失的 model

    for product_type in product_order:
        models_for_product = model_order_by_product.get(product_type, [])
        codes_for_this_product = []
        missing_models_for_product = []

        if not models_for_product:
            logger.info(f"产品类型 '{product_type}' 没有需要处理的 models。")
            continue  # 跳过这个产品类型

        logger.debug(f"处理产品类型 '{product_type}' 的 models: {models_for_product}")

        for target_model in models_for_product:
            target_model_str = str(target_model)  # 确保比较时类型一致
            code_to_use = None
            source = None
            handled_by_rule = False # 标记是否被新规则处理
            product_type_origin = product_type # 用于日志和 %int% 提示

            # --- 2. 新增：复杂规则处理块 ---
            if target_model_str == '插入长度（L）' and has_tg_product:
                logger.info(f"规则 2 触发：因存在 'tg' 产品，跳过模型 '{target_model_str}'。")
                handled_by_rule = True
                # code_to_use 保持 None, 不会添加代码

            elif target_model_str == '传感器连接螺纹（S）' and has_sensor_product:
                logger.info(f"规则 4 触发：因存在 'sensor' 产品，跳过模型 '{target_model_str}'。")
                handled_by_rule = True
                # code_to_use 保持 None, 不会添加代码

            elif target_model_str == '接头结构' and is_specific_tg_csv:
                code_to_use = '2'
                source = "rule_3_override"
                logger.info(f"规则 3 触发：因 'tg' 产品使用特定 CSV ({tg_csv_path})，模型 '{target_model_str}' 代码强制为 '2'。")
                handled_by_rule = True
                # 注意：强制代码 '2' 不包含 %int%，下面统一处理添加

            # --- 3. 标准代码查找 (仅当未被新规则处理时) ---
            if not handled_by_rule:
                if target_model_str in model_to_code_map:
                    # 在 selected_codes_data 中找到
                    code_to_use = model_to_code_map[target_model_str]
                    source = "selected"
                    logger.debug(f"找到产品 '{product_type}' 的 model '{target_model_str}' 对应的代码 (来自选择): {code_to_use}")
                else:
                    # 在 selected_codes_data 中未找到
                    missing_models_for_product.append(target_model_str)

                    # --- 规则 1 (旧可跳过逻辑) ---
                    if target_model_str in SKIPPABLE_MODELS:
                        logger.info(f"规则 1 (旧) 触发：产品 '{product_type}' 的 model '{target_model_str}' 在已选代码中缺失，且为可跳过项，将跳过。")
                        source = "rule_1_skip"
                        # code_to_use 保持 None
                    else:
                        # 非可跳过项，尝试默认值
                        model_details = model_details_map.get(target_model_str)
                        if model_details:
                            default_code = model_details.get("default_code")
                            product_type_origin = model_details.get("product_type", product_type) # 获取原始产品类型

                            if default_code is not None:
                                logger.info(f"在已选代码中未找到产品 '{product_type_origin}' 的 model '{target_model_str}' (非可跳过项)，将使用其默认代码: '{default_code}'")
                                code_to_use = str(default_code)
                                source = "default"
                            else:
                                logger.warning(f"在已选代码中未找到产品 '{product_type_origin}' 的 model '{target_model_str}' (非可跳过项)，且该 model 在 CSV 中没有标记为默认 (is_default='1') 的代码，将使用 '?'。")
                                code_to_use = "?"
                                source = "missing_default"
                        else:
                            logger.error(f"严重警告：在 model_details_map 中未找到 model '{target_model_str}' 的详情，无法确定默认代码，使用 '?'。")
                            code_to_use = "?"
                            source = "missing_details"

            # --- 统一处理代码添加和 %int% (无论代码来源) ---
            final_code_part = code_to_use # 初始值

            if code_to_use is not None: # 只有当 code_to_use 不是 None (即没有被跳过) 时才处理
                if "%int%" in code_to_use:
                    # 确定提示信息中的产品类型
                    prompt_product_type = product_type_origin if source == "default" else product_type
                    prompt_model_name = target_model_str
                    # ... (现有 %int% 输入和替换逻辑，确保使用 final_code_part = ... 来更新) ...
                    while True:
                        try:
                            prompt_message = (
                                f"请为 {prompt_product_type} - '{prompt_model_name}' 输入一个整数值 "
                                f"(代码模板: {code_to_use}, 直接回车跳过使用 '?'): "
                            )
                            user_input_str = input(prompt_message)

                            if not user_input_str:  # 用户直接按回车
                                final_code_part = "?"
                                logger.info(
                                    f"用户跳过了为 {prompt_product_type} - '{prompt_model_name}' 输入整数，使用 '?' 占位。")
                                break  # 跳出循环

                            # 用户有输入，尝试转换为整数
                            user_int = int(user_input_str)
                            # 替换占位符
                            final_code_part = code_to_use.replace(
                                "%int%", str(user_int))
                            logger.info(
                                f"用户为 {prompt_product_type} - '{prompt_model_name}' 输入了整数 {user_int}，替换占位符得到代码: '{final_code_part}'")
                            break  # 输入有效，跳出循环
                        except ValueError:
                            print("输入无效，请输入一个整数或直接回车跳过。")
                            logger.warning(
                                f"用户为 {prompt_product_type} - '{prompt_model_name}' 输入了非整数值，要求重新输入。")

                # 将最终处理后的代码部分添加到列表
                codes_for_this_product.append(final_code_part)

        if missing_models_for_product:
            missing_models_log[product_type] = missing_models_for_product

        if codes_for_this_product:  # 如果这个产品类型找到了任何代码
            # 将该产品类型的所有代码连接起来，中间不加空格
            product_string = "".join(codes_for_this_product)
            product_code_strings.append(product_string)
            logger.debug(f"生成产品 '{product_type}' 的代码块: {product_string}")
        else:
            logger.info(
                f"产品类型 '{product_type}' 没有生成任何代码（可能所有 model 都被跳过或无对应 code）。")


    if missing_models_log:
        logger.warning(
            f"以下 models 根据排序规则需要，但在 selected_codes_data 中缺失（已使用默认值或 '?' 替代，或被规则跳过）: {missing_models_log}") # 更新日志信息

    # 5. 将不同产品的代码字符串用空格连接
    final_code = " ".join(product_code_strings)

    # 6. 格式化输出
    output_string = f"产品型号生成：{final_code}"
    logger.info(f"最终生成的产品代码字符串: {output_string}")

    return output_string


# --- 主程序入口与模拟数据 ---
if __name__ == "__main__":
    # 模拟 fetch_csvlist.py 的输出
    mock_csv_list_map = {
        "transmitter": [],
        "sensor": [
            "libs/standard/sensor/HZ.csv"
        ],
        "tg": [
            "libs/standard/tg/TG_3.csv"
        ]
    }

    # 模拟 code_selector.py 的输出
    mock_standardized_params = {
        "'元件类型 ': '热电阻 '": {
            "model": "元件类型",
            "code": "HZ",
            "description": "热电阻",
            "param": "",
            "is_default": "1"
        },
        "'过程连接（法兰标准）': 'HG/T20615-2009'": {
            "model": "过程连接（法兰标准）",
            "code": "-H",
            "description": "HG20592、HG20615",
            "param": "HG□□；化工法兰",
            "is_default": "0"
        },
        "'插入深度 （U）': '250 '": {
            "model": "插入深度 (U)",
            "code": "-250",
            "description": "单位mm",
            "param": "",
            "is_default": "1"
        },
        "'管嘴长度 Length mm': '150'": {
            "model": "插入长度（L）",
            "code": "-150",
            "description": "单位mm",
            "param": "铠套热电阻/热点偶；带外保护套管时，此项可省略。",
            "is_default": "1"
        },
        "'元件数量 ': '单支式'": {
            "model": "元件数量",
            "code": "-S",
            "description": "单支式",
            "param": "1；单支；Simplex；单支单点；单元件；单只；Single，单支铠装；Single，，1支；one",
            "is_default": "1"
        },
        "'铠套外径（d）': 'Ø6'": {
            "model": "铠套外径(d)",
            "code": "6",
            "description": "Ø6mm",
            "param": "6；φ6；6mm；Φ6mm",
            "is_default": "1"
        },
        "'铠套材质 ': '316SS '": {
            "model": "铠套材质",
            "code": "RN",
            "description": "316SS",
            "param": "316SS；316；SS316；316S.S；316SST；S.S316；S31608；06Cr17Ni12Mo2",
            "is_default": "0"
        },
        "'分度号 ': 'PT100 三线 '": {
            "model": "分度号",
            "code": "3",
            "description": "PT100 三线",
            "param": "RTD Pt100 三线制；Pt100（三线制）；三线制；Pt100；Pt100（A级 三线制)；CLASS A/三线制；IEC60751 CLASS A / 三线制；RTD Pt100 三线制；三线制 Three wire；热电阻Pt100三线制；Pt100（A级 三线制）；3线制；3线制RTD；Pt100(3-wire) IEC60751 Class A；3 wires；3线RTD；PT100 3线； PT100 AT 0℃ 3WIRE (DIN TYPE)；3-wire；3线 IEC 60751；PT100-3WIRE；Pt100 ohm，3W，Class B；3 WIRE",
            "is_default": "0"
        },
        "'接线盒形式': '分体式 '": {
            "model": "接线盒形式",
            "code": "-2",
            "description": "接线盒、1/2NPT电气接口",
            "param": "分体式；精小型分体式温度变送器；分体式安装；分体式温度变送器",
            "is_default": "0"
        },
        "'TG套管形式 ': '整体钻孔锥形保护管 '": {
            "model": "套管形式",
            "code": "TG-K",
            "description": "K型法兰安装 锥形保护套管",
            "param": "整体钻孔直形套管；直型；直形；整体钻孔直形保护管；直形Straight；法兰式整体钻孔式直形；固定直形整体钻孔法兰套管；法兰直形套管；整体钻孔保护管Tapered Type；Solid hole， tapered；Tapered from drilled barstock；整体钻孔锥型；Tapered；单端钻孔锥型套管；法兰连接整体锥型钻孔；法兰式锥形整体钻孔外套管；钢棒整体钻孔锥形套管；固定法兰式整体钻孔锥形保护套管；固定法兰锥形整体钻孔式；加强型锥型整体钻孔；锥形；一体化整体钻孔锥形法兰套管；整体锥形套管；整体锥形钻孔；整体钻孔式的锥形套管；整体钻孔锥形；整体钻孔锥形保护管；整体钻孔锥形管；整体钻孔锥形套管；整钻锥形；锥形整体钻孔；锥形整体钻孔式套管；整体钻孔保护管",
            "is_default": "0"
        },
        "'套管材质 ': '316SS '": {
            "model": "套管材质",
            "code": "RN",
            "description": "316不锈钢",
            "param": "316SS；316；SS316；316S.S；316SST；S.S316；S31608；06Cr17Ni12Mo2；0Cr17Ni12Mo2",
            "is_default": "0"
        },
        "'法兰等级': 'Class150'": {
            "model": "过程连接（法兰等级）",
            "code": "1",
            "description": "PN2.0（150#）RF",
            "param": "PN2.0 RF；150# RF；PN20 RF；Class150 RF；150LB RF；□□-20 RF；CL150 RF；",
            "is_default": "0"
        },
        "'过程连接（法兰尺寸（Fs））': 'DN40'": {
            "model": "过程连接（法兰尺寸（Fs））",
            "code": "2",
            "description": "DN40（1-1/2\"）",
            "param": "DN40；□□DN40□□；1-1/2\"；□□1-1/2\"□□；1 1/2\"；□□1 1/2\"□□；40-□□",
            "is_default": "0"
        },
        "'根部直径（Q）': '根部不大于28,套管厚度由供货商根据振动频率和温度计算确定'": {
            "model": "根部直径 (Q)",
            "code": "-27",
            "description": "27mm",
            "param": "（不适用于DN25（1\"））",
            "is_default": "0"
        },
        "'法兰材质 ': '316SS '": {
            "model": "法兰材质",
            "code": "RN",
            "description": "316不锈钢",
            "param": "316SS；316；SS316；316S.S；316SST；S.S316；S31608；06Cr17Ni12Mo2；0Cr17Ni12Mo2",
            "is_default": "0"
        },
        "'接线口': '1/2\" NPT (F) '": {
            "model": "连接螺纹",
            "code": "6",
            "description": "1/2NPT",
            "param": "1/2NPT；1/2\"NPT；NPT1/2；NPT1/2\"；1/2NPT(M)；1/2\"NPT(M)；NPT1/2(M)；NPT1/2\"(M)；1/2\"NPT(外螺纹)；1/2\"NPT螺纹；热电阻(弹簧式1/2”NPT外螺纹连接)；固定外螺纹 1/2\"NPT；MFR STD",
            "is_default": "1"
        }
    }

    # --- 运行代码生成 ---
    print("-" * 60)
    print("开始模拟运行 code_generator (使用文件中的 mock 数据)...")
    print("-" * 60)

    # logger.setLevel(logging.DEBUG)

    final_product_code = generate_final_code(
        mock_csv_list_map, mock_standardized_params)

    print("-" * 60)
    print("模拟运行结束。")
    print("-" * 60)
    print(f"\n最终结果:\n{final_product_code}\n")
