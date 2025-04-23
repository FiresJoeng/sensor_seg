from openai import OpenAI
import json
import re

DEEPSEEK_API_KEY = "sk-28e66466f44148b4b6135f6e92d18651"  # 直接替换为你的API密钥
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1"

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_URL
)


def split_content(content, chunk_size=30000):
    """改进的语义分块函数，保留关键上下文"""
    # 首先按章节分割（假设文档使用Markdown标题结构）
    sections = re.split(r'(?m)^(#+ .+)$', content)
    chunks = []
    current_chunk = []
    current_length = 0

    # 重建章节结构（re.split会保留分隔符）
    for i in range(0, len(sections), 2):
        if i+1 < len(sections):
            section = sections[i] + sections[i+1]
        else:
            section = sections[i]

        section_length = len(section) * 1.33

        # 关键章节（如参数部分）不分割
        if re.search(r'^#{1,2}\s*(参数|规格|技术指标)', section, re.IGNORECASE):
            if current_chunk:  # 先保存当前块
                chunks.append(''.join(current_chunk))
                current_chunk = []
            chunks.append(section)  # 关键章节单独成块
            continue

        # 普通章节处理
        if current_length + section_length > chunk_size:
            chunks.append(''.join(current_chunk))
            current_chunk = [section]
            current_length = section_length
        else:
            current_chunk.append(section)
            current_length += section_length

    if current_chunk:
        chunks.append(''.join(current_chunk))
    return chunks


def call_deepseek_api(md_content):
    """改进的API调用函数，添加上下文继承和去重功能"""
    try:
        # 更精确的token估算
        estimated_tokens = len(md_content) * 1.33 + 100  # 基础token + 系统消息

        if estimated_tokens > 30000:  # 设置更保守的阈值
            chunks = split_content(md_content)
            combined_result = {"设备列表": []}
            seen_tags = set()  # 用于记录已处理的位号

            for chunk in chunks:
                # 添加上下文继承（前一块的最后一段作为本块的上下文）
                context = chunks[chunks.index(
                    chunk)-1].split('\n\n')[-1] if chunks.index(chunk) > 0 else ""

                print(f"正在处理分块 {chunks.index(chunk)+1}/{len(chunks)}")

                response = client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": '''你是一个工业设备参数提取专家，负责从用户提供的PDF规格书中提取温度变送器参数，需遵循以下规则：

### 提取规则
1. **数据结构分层**  
   - **全局参数**：所有位号共用的参数（如爆炸区域、环境温度、传感器类型等），统一提取到每个位号的`参数`中。  
   - **位号专属参数**：用途、管道设备号、介质等仅关联特定位号的参数，逐行提取。

2. **严格字段映射**  
   - 参数名称必须与原文档表头完全一致（如`热电阻引线数量`而非`RTD Lead Wire Connection`）。  
   - 保留原文档中的单位和符号（如`操作温度 (°C)`，不可省略括号或单位）。

3. **缺失值处理**  
   - 若某参数在文档中未提及，标注为`缺失（文档未提供）`，禁止使用`缺失位号X`。  
   - 若整行位号缺失，用`缺失位号X`占位，但需在备注中说明。

4. **数值格式强制统一**  
   - 数值需完整保留原文档格式（如`Φ8mm`不可简化为`8mm`）。  
   - 对疑似异常值（如`操作压力: '200'`）添加注释，但不修改原值。

5. **完整性校验**  
   - 每个位号的`参数`必须包含所有字段（全局+专属），缺失字段需明确标注。

### 输出格式示例：（严格遵循JSON Schema，但可根据实际内容进行调整）
```json
{
    "设备列表": [
        {
            "位号": "TT-XXXXX"（如'TT101'、'TE/TIT-361201'、'P3036M21100101'等格式）, 
            "参数": {
                // 全局参数（所有位号必须包含）
                "爆炸危险区域划分": "2区",
                "当地大气压 (kPa)": "86.16",
                "环境温度 (°C)": "-32.3~38.1",
                "传感器类型": "Pt100双支",
                "热电阻引线数量": "三线制",
                "精度": "IEC60751 A级",
                // ...其他全局参数,
                
                // 位号专属参数（按文档逐行提取）
                "用途": "一级脱氢反应器床层温度",
                "管道设备号": "R-50101",
                "介质": "BOG",
                "操作温度 (°C)": "100~330",
                "操作压力 (MPa(G))": "0.29",
                "插入深度 (mm)": "300",
                "测量范围 (°C)": "-40~400",
                "法兰标准": "HG/T 20592-2009",
                // ...其他专属参数
            }
        },
        // 其他位号...
    ],
    "备注": {
        "全局规则": "插入深度 = 140mm + 管道插入深度，<DN100管道需扩径",
        "异常值": {
            "TT-50301.操作压力": "原文档数值为'200'，疑似单位错误（未标注MPa）"
        },
        "缺失位号说明": "缺失位号1: 原文档中未找到对应行数据"
    }
}'''
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

                    },
                    temperature=0.1,
                    max_tokens=8096
                )

                if response.choices:
                    try:
                        chunk_result = json.loads(
                            response.choices[0].message.content)
                        if "设备列表" in chunk_result:
                            for device in chunk_result["设备列表"]:
                                tag = device.get("位号")
                                if tag not in seen_tags:  # 只添加未处理过的设备
                                    combined_result["设备列表"].append(device)
                                    seen_tags.add(tag)
                    except json.JSONDecodeError:
                        print("API返回的JSON解析失败，原始内容:",
                              response.choices[0].message.content)
            return combined_result

        # 原始的单次调用逻辑（内容不超过阈值时使用）
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": '''你是一个工业设备参数提取专家，负责从用户提供的PDF规格书中提取温度变送器参数，需遵循以下规则：

### 提取规则
1. **数据结构分层**  
   - **全局参数**：所有位号共用的参数（如爆炸区域、环境温度、传感器类型等），统一提取到每个位号的`参数`中。  
   - **位号专属参数**：用途、管道设备号、介质等仅关联特定位号的参数，逐行提取。

2. **严格字段映射**  
   - 参数名称必须与原文档表头完全一致（如`热电阻引线数量`而非`RTD Lead Wire Connection`）。  
   - 保留原文档中的单位和符号（如`操作温度 (°C)`，不可省略括号或单位）。

3. **缺失值处理**  
   - 若某参数在文档中未提及，标注为`缺失（文档未提供）`，禁止使用`缺失位号X`。  
   - 若整行位号缺失，用`缺失位号X`占位，但需在备注中说明。

4. **数值格式强制统一**  
   - 数值需完整保留原文档格式（如`Φ8mm`不可简化为`8mm`）。  
   - 对疑似异常值（如`操作压力: '200'`）添加注释，但不修改原值。

5. **完整性校验**  
   - 每个位号的`参数`必须包含所有字段（全局+专属），缺失字段需明确标注。

6. **上下文继承**
   - 若分块内容中某参数缺失，尝试从上一块的最后一段内容中继承。
   - 若无法继承，使用`分块导致缺失`在备注里标注。
   - 位号顺序严格按照原文档顺序，不允许跳过或重复。

### 输出格式示例：（严格遵循JSON Schema，但可根据实际内容进行调整）
```json
{
    "设备列表": [
        {
            "位号": "TT-XXXXX"（如'TT101'、'TE/TIT-361201'、'P3036M21100101'等格式）, 
            "参数": {
                // 全局参数（所有位号必须包含）
                "爆炸危险区域划分": "2区",
                "当地大气压 (kPa)": "86.16",
                "环境温度 (°C)": "-32.3~38.1",
                "传感器类型": "Pt100双支",
                "热电阻引线数量": "三线制",
                "精度": "IEC60751 A级",
                // ...其他全局参数,
                
                // 位号专属参数（按文档逐行提取）
                "用途": "一级脱氢反应器床层温度",
                "管道设备号": "R-50101",
                "介质": "BOG",
                "操作温度 (°C)": "100~330",
                "操作压力 (MPa(G))": "0.29",
                "插入深度 (mm)": "300",
                "测量范围 (°C)": "-40~400",
                "法兰标准": "HG/T 20592-2009",
                // ...其他专属参数
            }
        },
        // 其他位号...
    ],
    "备注": {
        "全局规则": "插入深度 = 140mm + 管道插入深度，<DN100管道需扩径",
        "异常值": {
            "TT-50301.操作压力": "原文档数值为'200'，疑似单位错误（未标注MPa）"
        },
        "缺失位号说明": "缺失位号1: 原文档中未找到对应行数据"
    }
}'''
                },
                {
                    "role": "user",
                    "content": f"请严格按照要求从以下markdown文档中提取参数：\n{md_content}"
                }
            ],
            response_format={
                "type": "json_object",

            },
            temperature=0.1,
            max_tokens=8096
        )

        # 处理响应
        if response.choices:
            message = response.choices[0].message
            try:
                return json.loads(message.content)
            except json.JSONDecodeError:
                print("API返回的JSON解析失败，原始内容:", message.content)
        return {}

    except Exception as e:
        print(f"API调用失败: {str(e)}")
        return {}


if __name__ == "__main__":
    md_path = None
    json_path = f"./output/{md_path}.json"

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    result = call_deepseek_api(md_content)
    print(result)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
