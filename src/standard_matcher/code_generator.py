import pandas as pd
import json
import logging
from typing import Dict, List, Any, Set

# 配置日志记录 (与项目其他部分保持一致)
# 建议在项目入口或配置文件中统一配置，这里为了独立运行添加
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义可跳过的 model 名称集合 (将来可以考虑移到配置)
SKIPPABLE_MODELS = {
    "传感器连接螺纹 (S)注：带温度元件，此项可省"
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
    ordered_models_by_product: Dict[str, List[str]] = {"transmitter": [], "sensor": [], "tg": []}
    processed_models_globally: Set[str] = set() # 用于跟踪全局已添加的model，确保跨产品类型的唯一性（如果需要）

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
                df = pd.read_csv(csv_path) # 尝试默认逗号分隔符

                if 'model' not in df.columns:
                    # 如果没有 model 列，尝试分号分隔符 (常见于某些中文环境导出的CSV)
                    logger.warning(f"文件 {csv_path} 未找到 'model' 列 (逗号分隔)，尝试使用分号分隔符...")
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
                    ordered_models_by_product[product_type].extend(models_in_this_csv_ordered)
                    logger.debug(f"为产品 '{product_type}' 添加 models: {models_in_this_csv_ordered}")

                else:
                    logger.warning(f"尝试多种方式后，CSV文件 '{csv_path}' 中仍未找到 'model' 列。")

            except FileNotFoundError:
                logger.error(f"错误：找不到CSV文件 '{csv_path}'。")
            except pd.errors.EmptyDataError:
                logger.warning(f"警告：CSV文件 '{csv_path}' 为空。")
            except Exception as e:
                logger.error(f"读取或处理CSV文件 '{csv_path}' 时出错: {e}")
        else:
            logger.warning(f"在CSV列表映射中未找到产品类型 '{product_type}' 或其列表为空，无法确定其 model 顺序。")

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

    if not any(model_order_by_product.values()): # 检查是否所有产品类型的 model 列表都为空
        logger.error("未能确定任何产品类型的 model 排序顺序，无法生成代码。")
        return "产品型号生成失败：无法确定任何排序。"

    # 2. 预加载各产品主CSV中的默认代码或占位符 '?'
    default_code_map: Dict[str, str] = {}
    product_order_for_defaults = ["transmitter", "sensor", "tg"] # 与 get_model_order 一致

    for product_type_default in product_order_for_defaults:
        if product_type_default in csv_list_map and csv_list_map[product_type_default]:
            csv_path_default = csv_list_map[product_type_default][0]
            logger.info(f"正在为产品 '{product_type_default}' 从 {csv_path_default} 预加载默认/占位符代码...")
            try:
                # 尝试读取CSV，处理分隔符问题
                df_default = None
                try:
                    df_default = pd.read_csv(csv_path_default)
                except Exception:
                    pass # 稍后检查列是否存在

                if df_default is None or 'model' not in df_default.columns:
                    try:
                        df_default = pd.read_csv(csv_path_default, sep=';')
                    except Exception as sep_err_default:
                        logger.warning(f"无法读取 {csv_path_default} 的 'model' 列 (尝试逗号和分号后)，跳过为 {product_type_default} 生成默认代码。错误: {sep_err_default}")
                        continue # 跳过此产品类型的默认代码生成

                if df_default is not None and 'model' in df_default.columns and 'code' in df_default.columns:
                    # 清理数据：去除 model 或 code 为空的行，并将 model 转为字符串
                    df_cleaned_default = df_default[['model', 'code']].dropna(subset=['model', 'code'])
                    df_cleaned_default['model'] = df_cleaned_default['model'].astype(str)

                    # 计算每个 model 的出现次数
                    model_counts_default = df_cleaned_default['model'].value_counts()

                    # 遍历此CSV中的唯一 model
                    for model_name_default in df_cleaned_default['model'].unique():
                        if model_name_default not in default_code_map: # 避免被后续产品类型覆盖（虽然可能性低）
                            if model_counts_default[model_name_default] == 1:
                                # 唯一出现，获取其 code 作为默认值
                                default_code_val = df_cleaned_default[df_cleaned_default['model'] == model_name_default]['code'].iloc[0]
                                default_code_map[model_name_default] = str(default_code_val)
                                logger.debug(f"为缺失 model '{model_name_default}' 设置默认代码: {default_code_val}")
                            else:
                                # 多次出现，使用 '?' 作为占位符
                                default_code_map[model_name_default] = "?"
                                logger.debug(f"为缺失 model '{model_name_default}' 设置占位符 '?' (CSV中出现次数: {model_counts_default.get(model_name_default, 0)})")
                else:
                    logger.warning(f"CSV文件 '{csv_path_default}' 缺少 'model' 或 'code' 列，无法为 {product_type_default} 生成默认代码。")

            except FileNotFoundError:
                logger.error(f"找不到CSV文件 '{csv_path_default}'，无法为 {product_type_default} 生成默认代码。")
            except pd.errors.EmptyDataError:
                logger.warning(f"CSV文件 '{csv_path_default}' 为空，无法为 {product_type_default} 生成默认代码。")
            except Exception as e:
                logger.error(f"处理CSV文件 '{csv_path_default}' 以生成默认代码时出错: {e}")

    logger.info(f"预加载的默认/占位符代码映射完成: {len(default_code_map)} 个条目")
    logger.debug(f"默认/占位符映射内容: {default_code_map}")


    # 3. 构建一个从 model 名称到 code 的快速查找字典 (来自 selected_codes_data)
    model_to_code_map = {}
    found_models_in_selection = set()
    for param_data in selected_codes_data.values():
        if isinstance(param_data, dict) and 'model' in param_data and 'code' in param_data:
            model_name = str(param_data['model']) # 确保是字符串
            code_value = str(param_data['code']) if pd.notna(param_data['code']) else '' # 处理可能的 NaN 或 None
            model_to_code_map[model_name] = code_value
            found_models_in_selection.add(model_name)
        else:
             logger.warning(f"selected_codes_data 中的条目格式不符合预期或缺少 model/code: {param_data}")

    logger.debug(f"从 selected_codes_data 构建的 model->code 映射: {model_to_code_map}")
    logger.debug(f"在 selected_codes_data 中找到的 models: {found_models_in_selection}")

    # 4. 按产品顺序处理并拼接代码块
    product_code_strings = []
    product_order = ["transmitter", "sensor", "tg"] # 预定义的产品处理顺序
    missing_models_log = {} # 记录每个产品类型在 selected_codes_data 中缺失的 model

    for product_type in product_order:
        models_for_product = model_order_by_product.get(product_type, [])
        codes_for_this_product = []
        missing_models_for_product = []

        if not models_for_product:
            logger.info(f"产品类型 '{product_type}' 没有需要处理的 models。")
            continue # 跳过这个产品类型

        logger.debug(f"处理产品类型 '{product_type}' 的 models: {models_for_product}")

        for target_model in models_for_product:
            target_model_str = str(target_model) # 确保比较时类型一致
            if target_model_str in model_to_code_map:
                code = model_to_code_map[target_model_str]
                codes_for_this_product.append(code)
                logger.debug(f"找到产品 '{product_type}' 的 model '{target_model_str}' 对应的代码: {code}")
            else:
                # 在 selected_codes_data 中未找到
                missing_models_for_product.append(target_model_str) # 记录为在输入选择中缺失

                # 检查是否是可跳过项
                if target_model_str in SKIPPABLE_MODELS:
                    logger.info(f"产品 '{product_type}' 的 model '{target_model_str}' 在已选代码中缺失，且为可跳过项，将跳过。")
                    # 不添加任何代码或占位符
                else:
                    # 非可跳过项，尝试从 default_code_map 获取默认值或 '?'
                    default_or_placeholder = default_code_map.get(target_model_str, "?") # 如果连默认映射里都没有，则用 '?'
                    codes_for_this_product.append(default_or_placeholder)
                    logger.warning(f"在已选代码中未找到产品 '{product_type}' 的 model '{target_model_str}' (非可跳过项)，使用默认/占位符: '{default_or_placeholder}'")

        if missing_models_for_product:
            missing_models_log[product_type] = missing_models_for_product

        if codes_for_this_product: # 如果这个产品类型找到了任何代码
            # 将该产品类型的所有代码连接起来，中间不加空格
            product_string = "".join(codes_for_this_product)
            product_code_strings.append(product_string)
            logger.debug(f"生成产品 '{product_type}' 的代码块: {product_string}")
        else:
             logger.info(f"产品类型 '{product_type}' 没有生成任何代码（可能所有 model 都缺失或无对应 code）。")


    if missing_models_log:
         logger.warning(f"以下 models 根据排序规则需要，但在 selected_codes_data 中缺失（已使用默认值或 '?' 替代）: {missing_models_log}")

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
    "'元件类型 (仪表名称 Inst. Name)': '热电阻'": {
        "model": "元件类型",
        "code": "HZ",
        "description": "热电阻",
        "remark": "*1：适用于PT1、PT2、PT3金属保护管时，选择000。；*2：插入长度≤50mm，材质与外保护管一致，如需其他，请注明。；*3：带外保护套管时，此项可省略。；*4：仅适用于YTA50、YTA70。；*5：仅适用于HR。；*6：仅适用于NEPSI防爆/N2。；*7：产品如带外保护套管，仍需防尘防爆认证。；*9：仅适用于NEPSI防爆/NS2。"
    },
    "'TG套管形式 (套管材质 Well Mat'l)': '316'": {
        "model": "TG",
        "code": "TG",
        "description": "保护套管",
        "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数(温度，压力，介质密度，介质黏度，介质流速，管道尺寸)。 ；(2)粗糙度Ra=0.8，长度不超过500mm"
    },
    "'过程连接（法兰等级） (管嘴长度 Length mm)': '150'": {
        "model": "钎套长度",
        "code": "150",
        "description": "单位mm",
        "remark": "*1 采用抱箍将感温片固定在测温点，需提供现场管道直径。；*2 将感温片焊接固定在测温点，需提供现场曲面半径。；*3 铠套端预留螺纹接口，便于在隔爆安装环境下安装。"
    },
    "'过程连接（法兰等级） (插入深度 Well Length (mm))': '250'": {
        "model": "铠套长度",
        "code": "250",
        "description": "单位mm *注3，仅适用于SR)",
        "remark": "*1 采用抱箍将感温片固定在测温点，需提供现场管道直径。；*2 将感温片焊接固定在测温点，需提供现场曲面半径。；*3 铠套端预留螺纹接口，便于在隔爆安装环境下安装。"
    },
    "'连接螺纹 (测量范围 Meas. Range (°C))': '缺失（文档未提供）'": {
        "model": "补偿导线长度",
        "code": "%int%",
        "description": "单位mm *3",
        "remark": "*1 采用抱箍将感温片固定在测温点，需提供现场管道直径。；*2 将感温片焊接固定在测温点，需提供现场曲面半径。；*3 铠套端预留螺纹接口，便于在隔爆安装环境下安装。"
    },
    "'元件数量 (类型 Type)': '单支'": {
        "model": "元件数量",
        "code": "-S",
        "description": "单支式",
        "remark": "*1：适用于PT1、PT2、PT3金属保护管时，选择000。；*2：插入长度≤50mm，材质与外保护管一致，如需其他，请注明。；*3：带外保护套管时，此项可省略。；*4：仅适用于YTA50、YTA70。；*5：仅适用于HR。；*6：仅适用于NEPSI防爆/N2。；*7：产品如带外保护套管，仍需防尘防爆认证。；*9：仅适用于NEPSI防爆/NS2。"
    },
    "'连接螺纹 (温度元件型号 Therm. Element Model)': '缺失（文档未提供）'": {
        "model": "传感器连接螺纹 (S)注：带温度元件，此项可省",
        "code": "1",
        "description": "M12×1.5",
        "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数(温度，压力，介质密度，介质黏度，介质流速，管道尺寸)。 ；(2)粗糙度Ra=0.8，长度不超过500mm"
    },
    "'过程连接（法兰等级） (允差等级 Tolerance Error Rating)': 'A级'": {
        "model": "附加规格代码",
        "code": "/A3",
        "description": "外保护套管频率强度计算",
        "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数...；(2)粗糙度Ra=0.8，长度不超过500mm"
    },
    "'铠套材质 (铠装材质 Armo. Mat'l)': '316'": {
        "model": "铠装材质",
        "code": "RN",
        "description": "316SS",
        "remark": "*1：适用于PT1、PT2、PT3金属保护管时，选择000。；*2：插入长度≤50mm，材质与外保护管一致，如需其他，请注明。；*3：带外保护套管时，此项可省略。；*4：仅适用于YTA50、YTA70。；*5：仅适用于HR。；*6：仅适用于NEPSI防爆/N2。；*7：产品如带外保护套管，仍需防尘防爆认证。；*9：仅适用于NEPSI防爆/NS2。"
    },
    "'铠套外径(d) (铠装直径 Armo. Dia. (mm))': 'Ф6'": {
        "model": "铠装外径（d）",
        "code": "6",
        "description": "Ø6mm",
        "remark": "*1：适用于PT1、PT2、PT3金属保护管时，选择000。；*2：插入长度≤50mm，材质与外保护管一致，如需其他，请注明。；*3：带外保护套管时，此项可省略。；*4：仅适用于YTA50、YTA70。；*5：仅适用于HR。；*6：仅适用于NEPSI防爆/N2。；*7：产品如带外保护套管，仍需防尘防爆认证。；*9：仅适用于NEPSI防爆/NS2。"
    },
    "'TG套管形式 (套管形式 Well Type)': '整体钻孔锥形保护管'": {
        "model": "选型",
        "code": "-K",
        "description": "K型法兰安装锥形保护套管",
        "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数(温度，压力，介质密度，介质黏度，介质流速，管道尺寸)。 ；(2)粗糙度Ra=0.8，长度不超过500mm"
    },
    "'过程连接（法兰等级） (压力等级 Pressure Rating)': 'Class150'": {
        "model": "过程连接",
        "code": "-S",
        "description": "抱箍式 *1",
        "remark": "*1 采用抱箍将感温片固定在测温点，需提供现场管道直径。；*2 将感温片焊接固定在测温点，需提供现场曲面半径。；*3 铠套端预留螺纹接口，便于在隔爆安装环境下安装。"
    },
    "'铠套外径(d) (套管外径 Well Outside Dia. (mm))': '根部不大于28,套管厚度由供货商根据振动频率和强度计算确定'": {
        "model": "根部直径 (Q)",
        "code": "-%int%",
        "description": "单位mm",
        "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数...；(2)粗糙度Ra=0.8，长度不超过500mm"
    },
    "'过程连接 (法兰标准 Flange STD.)': 'HG/T20615-2009'": {
        "model": "法兰标准",
        "code": "-H",
        "description": "HG20592、HG20615",
        "remark": "例：TG-K2-H11PN-150GH-22/18 ；(1)需提供现场工况过程参数(温度，压力，介质密度，介质黏度，介质流速，管道尺寸)。 ；(2)粗糙度Ra=0.8，长度不超过500mm"
    }
}

    # --- 运行代码生成 ---
    print("-" * 60)
    print("开始模拟运行 code_generator (使用文件中的 mock 数据)...")
    print("-" * 60)

    # logger.setLevel(logging.DEBUG)

    final_product_code = generate_final_code(mock_csv_list_map, mock_standardized_params)

    print("-" * 60)
    print("模拟运行结束。")
    print("-" * 60)
    print(f"\n最终结果:\n{final_product_code}\n")
