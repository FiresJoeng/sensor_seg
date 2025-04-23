import os
import re
import json
import pandas as pd
from openai import OpenAI


# ------------------- 配置加载 -------------------
# 直接定义路径，不再使用环境变量
SEMANTIC_TABLE_PATH = "C:\\Users\\41041\\Desktop\\一体化温度变送器语义库 - 副本.xlsx"
INPUT_MD_DIR = "3-一体化温度变送器_20200218_.md"  # 或者直接指定单个文件路径
OUTPUT_EXCEL_PATH = "变送器推荐结果.xlsx"

# DeepSeek API配置
DEEPSEEK_API_KEY = "sk-28e66466f44148b4b6135f6e92d18651"  # 直接替换为你的API密钥
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1"

# ------------------- 全局常量 -------------------
# 乙方字段别名映射（甲方字段名→乙方标准参数）
FIELD_ALIAS_MAPPING = {
    "输出信号": [
    'OUTPUT SIGNAL', '变送器信号输出 Output Signal', '输出信号 / 通讯协议 Output Signal',
    '通讯形式 Communication Type', '电源/输出信号', '输出信号 Output Sig.', '输出信号类型',
    'OUTPUT SIGNAL mADC', 'Analog Signal 模拟信号', '信号输出 Output Signal', 'OUTPUT SIGAL',
    '输出信号 (两线制)', 'OUT PUT SIGNAL / 输出信号', '1st Output Signal 输出信号1',
    'Output Signall 电源 Power', 'output signal', '输出信号', 'mA.DC', '输出信号 OUTPUT',
    'Output / 输出', '输出信号 Output Signal 电源 Power', 'Output Signal 输出信号',
    '通讯形式 Communication TYPE', 'Output signal / HART 输出信号 / 带HART功能', 'Head Mounted',
    'Communication', 'Digital Signal 数字信号', '信号输出 Signal output', 'Communication Protocol',
    'OUTPUT SIGN.', 'Output signal', 'Output', '通讯协议 Communication Protocol', '信号 Signal',
    'HART COMMUNICATION PROTOCOL (INTEGRAL TYPE)', 'OUTPUT SIGNAL (TWO-WIRE SYSTEM)',
    '通讯方式 Communication Mode', '输出信号（两线制）'],

    "NEPSI": [
    '防爆标志 explosion proof', '防爆等级EXPLOSION-PROOF CLASS', '防爆等级Ex Class', 
    '防爆等级 EXPLOSION-PROOF CERTIF.', 'Ex Approval / Certification', '防爆等级Ex. Class', 
    'Ex Protection', '防护等级 / 防爆等级', 'EX-PROOF PROTECTION', '防爆等级 Explosion-proof Class', 
    'Explosion Proof 防爆等级', 'EXPLOSION PROOF CLASS/防爆等级', '防爆等级 Anti hazard Classification', 
    '防爆等级 Cenelec', '防爆等级 EXPLOSION PROOF', '外壳防爆等级 Explosion Proof', 
    '防爆标志Explosion Proof', '防爆等级Explosion Proof', '防爆等级Ex. CLASS', 
    '防护等级/防爆等级 Encl.portection/Exp.Pro', '防爆等级 Ex. Proof', '防爆等级Explosion-proof', 
    'EXPLOSION PROOF', '防爆等级', 'Explosion classification/防爆等级', '防爆等级 Ex. CERTIFICATION', 
    '防爆认证 Explosion-proof Certification', 'HAZARDOUS AREA CLASS', '防爆证书 Certificates For Material', 
    '防爆等级Explosion Protection', '防爆等级 EXPLOSION-PROOF CLASS', '防爆等级 EXPLOSION PROOF CLASS', 
    '防爆等级 Explosion Protection', '防爆认证 ExplosionProof Class', 'Explosion Approval 防爆认证', 
    '防爆等级 Explosion Proof', '防爆', '变送器防爆等级 Explosion Proof', 
    '防爆防护等级 Explosion & Explosion Proof'
],

    "壳体代码": [
    "接线盒材质 Terminal Box Mat'l", '接线盒材质 HEAD MATERIAL', '接线盒材质 Terminal Box Material',
    'Enclosure Material 壳体材质', 'Material 材质', '保护外壳 Housing', '外壳材质 HEAD MATERIAL',
    '外壳材料 Terminal Box Material', 'Housing 外壳', '接线盒形式 Terminal Box Style', '材质 Material',
    '壳体材质 Housing Material', "壳体材质 Housing Mat'l", 'Transmitter Housing Material 变送器外壳材质',
    "表壳材质 Housing Mat'l", 'Housing Matl. 外壳材质', 'Housing Material 外壳材质', '外壳材料 SHELL MATERIAL',
    '*外壳材质Housing Material', '接线盒盖和链 Screw Cap & Chain', 'CASE MATERIAL', '接线盒材质case material',
    '接线盒材料 Teminal Box Material', 'Housing/外壳材质', 'Body Material 本体材质', '表壳材质',
    '接线盒材料TERMINAL BOX MATERIAL', '外壳材质 BODY MAT,L', '外壳材质Case Material', '壳体材质',
    "表壳材质HOUSING MAT'L", '变送器外壳材质', '本体材质 BODY MATERIAL', '接线盒外壳材质 CASE MATERIAL',
    '外壳材质 Housing Material', '外壳材质Housing Material', '外壳 Housing Material', '接线盒形式 Terminal Box Type',
    '壳体材质Housing Mtl.', "外壳材质 Case Mat'l", 'HOUSING MATERIAL', '格兰/材质 Gland / Material',
    '外壳Housing', '外壳材质 Housing Mtl.', '变送器外壳材质 Housing Material', '表壳材质 Case Material',
    "壳体材质 Case Mat'l", '表体材质 Case Material', '外壳材料', '接线盒材质 Joint Box M.', '通用接线盒材质',
    '壳体材质 Case Material', '外壳材质', '接线盒材料 TERMINAL BOX MATERIAL', '外壳材质 BODY MATERIAL',
    '外壳材质 Case Material', '表头外壳材质HEAD SHELL MATERIAL', '壳体材质Housing Mat’l.',
    '壳体材质MATERIAL OF SHELL', '接线盒材料', '接线盒材质Terminal Box Mtl.', '接线盒材质',
    "外壳材质 Housing Mat'l", "变送器外壳材质 Mat'l"],

    "接线口": [
    '电气接口尺寸 ELEC. CONN. SIZE', '电气连接 Elec.Conn.', '电气接口 Elec. Conn.',
    'ELECTRIC CONNECT SIZE/电气连接尺寸', '电气接口尺寸 ELEC.CONN.SIZE', '电气接口Cable Entry',
    '电气连接 Electrical Conn.', '电气连接接口 Electrical Connection Size', '电气接口尺寸ELEC. CONN. SIZE',
    'Elec. Conn. 电气接口', '安装螺纹规格 THREAD SIZE/电气接口尺寸 ELEC.CONN.SIZE', '电气接口尺寸 CONDUIT SIZE',
    '电气接口Electrical Conn.', '电气接口 Electrical Interface', '与温度元件连接尺寸 Conn. With Element',
    '电气接口', '电气连接 Elec.Conn', 'Electrical connection/电气接口', 'Electrical Connection 电气接口',
    'Electrical connection', 'CABLE CONDUIT', '电气接口 Elec.Conn.Size', '电器连接Electrical Connection',
    '过程接口', '电⽓接⼝尺⼨ ELEC.CONN.SIZE', '电气接口 Cable Conn.', '电气接口尺寸ELECTRICAL CONN. SIZE',
    '电气接口尺寸 Electric Connection Size', '电气接口尺寸ELEC.CONN.SIZE', 'CONDUIT CONN. ( WITH PLUG )',
    '电气接口Elc.Conn.', '电气接口规格 Electrical Connection', '电气接口尺寸 ELEC.CINN.SIZE',
    '电气接口 Electrical Connection Size', '电气接口尺寸 ELEC. CONNECTION SIZE', '电气接口尺寸 Cable Entry',
    '电气接口 Eelectric Connection', '电气连接 electric connection', '电缆接口', '电气接口 Electrical Conn.',
    'Cable Entry', '电气接口规格 ELEC.CONN.SIZE', '电气接口Conduit Connection Size', '电气连接尺寸',
    '电气接口 Elec.Conn.', '电气连接 Elec. Conn.', '电气连接 Electrical Connection', '电气接口Conduit Connection',
    '电缆密封接头 Cable Gland', '电气接口尺寸 Elec.connect size', '电气接口 Conduit Conn.', '电气接口 Cable Entry',
    'Conduit Connection Size 电气连接尺寸', '电气接口Elec. Conn.', '电气接口尺寸 ELEC.CONN. SIZE', '电气接口尺寸',
    '变送器电气接口 electrical connection', 'Conduit Connection (Cable Entry)'],

    "传感器输入": ['传感器输入','元器件数量'],

    "说明书语言": ["使用说明书语言", "说明书语言", "操作手册语言",'使用说明书','Operating&Maintenance Manual'],

    "内置指示器": [
    '输出指示表OUTPUT INDICTOR', '就地指示 Local Indicator', 'LOCAL INDICATION', 
    '输出信号指示表\xa0Out.\xa0Sign.\xa0Indic.(0-100\xa0Linear)', '就地LCD显示', 
    '表头显示 Integral indicator', '输出信号 Output Sign.', '输出信号指示表 Out. Sign. Indic.(0-100 Linear)', 
    '一体化指示表 Integral Indicator', '现场指示 Local lndictor', '类型 Mount Type', '输出指示表', 
    '输出信号指示表LOCAL INDICATOR', '就地显示', '现场指示表', 'Local Indicator', 
    '输出指示表 Integral Indicator', 'LCD Indicator on Transmitter 液晶显示', '输出指示表 Indicator', 
    '一体化显示Integral Meter', '显示', '就地LCD显示Local LCD Display', '显示与界面 DISPLAY AND INTERFACE', 
    'Local Indicator 就地显示', '一体化指示表 Intgral INdic.', '现场指示表 LOCATION INDICATOR', 
    '就地显示Local Indicator', '输出信号指示表Out.Sign. Indic.', 'DISPLAY/显示', 'LCD', '其它other', 
    '输出指示表 Output Indicator', 'INDICATOR', '型号 Model', '现场指示表 Local Indication', 
    '现场指示Local Indicator', '形式', '一体化指示表Intgral Indic', '就地显示 Local Display', 
    '一体化表头 INTEGRAL INDICATOR', '一体式显示 Integral Meter', '输出指示 OUTPUT INDICATOR', 
    '输出指示表OUTPUT INDICATOR', '显示表头Indicator', '一体式显示 Integral Indicator', '显示表头 Indicator', 
    '数字显示表 Indicator', '现场指示表 Local Indicator', 'Local indicator 一体化显示'],

    "安装支架":  [
    '安装方式 INSTALLATION TYPE', '安装支架', '安装形式 INSTALLATION TYPE', '安装支架 Mounting Bracket', 
    '安装方式Mounting Type', '安装方式', '安装方式 Mounting Style', 'Mounting Brackets 安装支架', 
    'Mounting Bracket 安装支架', '安装支架Mounting Brackets', '安装支架 Mounting']
}

# DeepSeek参数提取Schema（根据乙方需求定义）
EXTRACTION_SCHEMA = {
    "类型": "object",
    "属性": {
        "位号": {"类型": "string", "描述": "设备唯一标识，如TT101"},
        "输出信号": {"类型": "string", "描述": "信号输出类型，如4~20 mA"},
        "防爆等级": {"类型": "string", "描述": "防爆认证代码，如ExiaⅡCT4"},
        "壳体代码": {"类型": "string", "描述": "接线盒材质，如不锈钢/铝"},
        "接线口": {"类型": "string", "描述": "电气接口尺寸，如M20×1.5"},
        "传感器输入": {"类型": "number", "描述": "测量元件数量，1或2"},
        "说明书语言": {"类型": "string", "描述": "中文/English"}
    },
    "必要项": ["位号", "输出信号", "防爆等级", "壳体代码", "接线口", "传感器输入", "说明书语言"]
}


# ------------------- 核心函数定义 -------------------
def load_semantic_table(excel_path):
    """加载乙方语义表，转换为结构化字典"""
    try:
        df = pd.read_excel(excel_path, sheet_name="变送器部分")
        semantic = {}
        for _, row in df.iterrows():
            standard_param = str(row["标准参数"]) if pd.notna(row["标准参数"]) else ""  # 确保标准参数是字符串
            actual_values = str(row["实际参数值"]) if pd.notna(row["实际参数值"]) else ""  # 确保实际参数值是字符串
            
            semantic[standard_param] = {
                "实际参数值": [v.strip() for v in actual_values.split("；")] if actual_values else [],
                "对应代码": str(row["对应代码"]) if pd.notna(row["对应代码"]) else "",
                "缺省默认值": str(row["缺省默认值"]) if pd.notna(row["缺省默认值"]) else "",
                "备注": str(row["备注"]) if pd.notna(row["备注"]) else "",
                "正则匹配": str(row.get("正则匹配", "")) if pd.notna(row.get("正则匹配", "")) else None
            }
        return semantic
    except Exception as e:
        print(f"语义表加载失败：{e}")
        return {}


def read_md_file(md_path):
    """读取MD文件内容"""
    try:
        with open(md_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"读取MD文件 {md_path} 失败: {e}")
        return None


# 在文件顶部添加客户端初始化
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_URL
)

def call_deepseek_api(md_content):
    try:
        # 更精确的token估算
        estimated_tokens = len(md_content) * 1.33 + 100  # 基础token + 系统消息
        
        if estimated_tokens > 30000:  # 设置更保守的阈值
            chunks = split_content(md_content, overlap=1000)  # 使用重叠分块
            combined_result = {}
            
            for i, chunk in enumerate(chunks):
                # 添加上下文继承（前一块的最后overlap内容）
                context = chunks[i-1][-1000:] if i > 0 else ""
                
                print(f"正在处理分块 {chunks.index(chunk)+1}/{len(chunks)}")
                
                response = client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content":'''你是一个工业设备参数提取专家，请从文本中提取参数并严格按JSON Schema返回,
                                    注意不要改变原文档中的参数名称，严格按照文档里的原本内容提取'''
                        },
                        {
                            "role": "assistant",
                            "content": f"上文上下文：{context}" if context else ""
                        },
                        {
                            "role": "user", 
                            "content": f"这是文档的第{chunks.index(chunk)+1}部分，请从中提取参数：\n{chunk}"
                        }
                    ],
                    response_format={
                        "type": "json_object",
                        "schema": EXTRACTION_SCHEMA
                    },
                    temperature=0.1,
                    max_tokens=8096
                )
                
                if response.choices:
                    try:
                        chunk_result = json.loads(response.choices[0].message.content)
                        combined_result.update(chunk_result)
                    except json.JSONDecodeError:
                        print("API返回的JSON解析失败，原始内容:", response.choices[0].message.content)
            return combined_result
        
        # 原始的单次调用逻辑(内容不超过阈值时使用)
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content":'''你是一个工业设备参数提取专家，请从文本中提取参数并严格按JSON Schema返回,
                            注意不要改变原文档中的参数名称，严格按照文档里的原本内容提取'''
                },
                {
                    "role": "user",
                    "content": f"请从以下markdown文档中提取参数：\n{md_content}"
                }
            ],
            response_format={
                "type": "json_object",
                "schema": EXTRACTION_SCHEMA
            },
            temperature=0.1,
            max_tokens=8096
        )
        
        # 处理响应
        if response.choices:
            try:
                # 添加更健壮的JSON预处理
                json_str = response.choices[0].message.content
                
                # 修复常见JSON格式问题
                json_str = json_str.replace('\n', ' ').replace('\r', '')  # 去除换行符
                json_str = re.sub(r',\s*}', '}', json_str)  # 修复多余的逗号
                json_str = re.sub(r',\s*]', ']', json_str)  # 修复多余的逗号
                
                # 检查并修复未闭合的JSON
                if not json_str.strip().endswith('}'):
                    json_str = json_str[:json_str.rfind('}')+1] if '}' in json_str else json_str + '}'
                if not json_str.strip().startswith('{'):
                    json_str = json_str[json_str.find('{'):] if '{' in json_str else '{' + json_str
                
                # 验证并解析JSON
                result = json.loads(json_str)
                
                # 删除必要字段检查代码
                return result
                
            except json.JSONDecodeError as e:
                # 保存错误响应到文件
                error_file = "api_error_response.json"
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(response.choices[0].message.content)
                print(f"JSON解析失败，已保存原始响应到: {error_file}")
                print(f"错误详情: {e}\n原始内容片段: {response.choices[0].message.content[max(0, e.pos-100):e.pos+100]}")
                
                # 尝试更激进的修复方式
                try:
                    fixed_json = re.sub(r'(?<!\\)(?:\\\\)*"(?=[^"]*$)', '', response.choices[0].message.content)
                    return json.loads(fixed_json)
                except Exception as e2:
                    print(f"JSON修复失败: {e2}")
                    return {}
        return {}
        
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        return {}

# 新增分块函数
def split_content(content, chunk_size=20000, overlap=1000):
    """改进的分块函数，保留上下文重叠"""
    paragraphs = content.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for i, para in enumerate(paragraphs):
        para_size = len(para)
        
        # 如果当前段落会使分块超过大小限制
        if current_size + para_size > chunk_size and current_chunk:
            # 添加当前分块（包含重叠部分）
            chunks.append('\n\n'.join(current_chunk))
            
            # 保留最后overlap个字符作为下一个分块的开头
            overlap_text = '\n\n'.join(current_chunk)[-overlap:] if overlap else ""
            current_chunk = [overlap_text] if overlap_text else []
            current_size = len(overlap_text)
        
        current_chunk.append(para)
        current_size += para_size
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


def align_field(甲方字段):
    """将甲方字段名映射到乙方标准参数"""
    for 标准参数, 别名列表 in FIELD_ALIAS_MAPPING.items():
        if 甲方字段.strip() in 别名列表:
            return 标准参数
    # 模糊匹配（处理大小写、空格）
    甲方字段_clean = 甲方字段.strip().lower().replace(" ", "").replace("-", "")
    for 标准参数, 别名列表 in FIELD_ALIAS_MAPPING.items():
        for 别名 in 别名列表:
            if 别名.strip().lower().replace(" ", "").replace("-", "") == 甲方字段_clean:
                return 标准参数
    return None  # 未匹配字段


def match_parameter_value(甲方值, 标准参数, 语义表):
    """根据甲方值和标准参数，匹配乙方代码"""
    if not 语义表.get(标准参数):
        return None  # 标准参数未定义
    
    # 确保输入值为字符串
    甲方值 = str(甲方值) if 甲方值 is not None else ""
    
    预处理值 = 甲方值.strip().lower()
    预处理值 = re.sub(r"[Ⅱⅱ]", "ii", 预处理值)  # 统一罗马数字
    预处理值 = re.sub(r"～", "-", 预处理值)  # 统一符号
    
    # 正则匹配（如果语义表中定义了正则）
    if 语义表[标准参数]["正则匹配"]:
        if re.match(语义表[标准参数]["正则匹配"], 预处理值):
            return 语义表[标准参数]["对应代码"]
    
    # 精确匹配（实际参数值列表，支持|分隔的多个值）
    for 有效值 in 语义表[标准参数]["实际参数值"]:
        有效值_clean = 有效值.strip().lower()
        if "|" in 有效值_clean:
            for 子值 in 有效值_clean.split("|"):
                if 子值 in 预处理值:
                    return 语义表[标准参数]["对应代码"]
        elif 有效值_clean in 预处理值:
            return 语义表[标准参数]["对应代码"]
    
    # 使用缺省值
    return 语义表[标准参数]["缺省默认值"]


def generate_spec_code(参数字典, 语义表):
    """生成乙方规格代码"""
    基础型号 = "YTA610"  # 固定基础型号（可从语义表中配置）
    参数顺序 = ["输出信号", "传感器输入", "壳体代码", "接线口", "NEPSI", "说明书语言"]
    代码段 = []
    
    for 标准参数 in 参数顺序:
        甲方值 = 参数字典.get(标准参数, 语义表[标准参数]["缺省默认值"])
        if 甲方值 is None:  # 修改判断条件
            continue  # 跳过空值或未匹配的参数
        
        # 处理特殊参数类型（如传感器输入需转为数字）
        if 标准参数 == "传感器输入":
            甲方值 = str(int(float(甲方值))) if 甲方值 else "1"  # 添加默认值处理
        else:
            甲方值 = str(甲方值)  # 确保其他参数也是字符串
            
        代码 = match_parameter_value(甲方值, 标准参数, 语义表)
        if not 代码:
            continue  # 代码未生成，跳过（记录日志）
        
        # 添加参数符号前缀（根据乙方规则）
        符号映射 = {
            "输出信号": "-",
            "防爆等级": "/",
            "传感器输入": "",  # 示例：单支输入代码为1，直接拼接
            "壳体代码": "",
            "接线口": "",
            "说明书语言": ""
        }
        前缀 = 符号映射.get(标准参数, "")
        代码段.append(f"{前缀}{代码}")
    
    return f"{基础型号}{''.join(代码段)}"


# ------------------- 主流程函数 -------------------
def main():
    # 1. 加载语义表
    semantic_table = load_semantic_table(SEMANTIC_TABLE_PATH)

    all_results = []
    # 遍历MD文件目录
    if os.path.isdir(INPUT_MD_DIR):
        for filename in os.listdir(INPUT_MD_DIR):
            if filename.endswith('.md'):
                md_path = os.path.join(INPUT_MD_DIR, filename)
                md_content = read_md_file(md_path)
                if md_content:
                    extracted_params = call_deepseek_api(md_content)
                    if extracted_params:
                        spec_code = generate_spec_code(extracted_params, semantic_table)
                        all_results.append({
                            "文件名": filename,
                            "提取参数": extracted_params,
                            "规格代码": spec_code
                        })
    elif os.path.isfile(INPUT_MD_DIR) and INPUT_MD_DIR.endswith('.md'):
        md_content = read_md_file(INPUT_MD_DIR)
        if md_content:
            extracted_params = call_deepseek_api(md_content)
            if extracted_params:
                spec_code = generate_spec_code(extracted_params, semantic_table)
                all_results.append({
                    "文件名": os.path.basename(INPUT_MD_DIR),
                    "提取参数": extracted_params,
                    "规格代码": spec_code
                })

    # 保存结果到Excel
    result_df = pd.DataFrame(all_results)
    result_df.to_excel(OUTPUT_EXCEL_PATH, index=False)


if __name__ == "__main__":
    main()