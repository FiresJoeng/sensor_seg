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

EXTRACTION_SCHEMA = {
    "类型": "object",
    "属性": {
        "设备列表": {
            "类型": "array",
            "items": {
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
                "必要项": ["位号", "输出信号", "防爆等级", "壳体代码", "接线口", "传感器输入"]
            }
        }
    }
}

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
    """改进的API调用函数，添加上下文继承"""
    try:
        # 更精确的token估算
        estimated_tokens = len(md_content) * 1.33 + 100  # 基础token + 系统消息
        
        if estimated_tokens > 30000:  # 设置更保守的阈值
            chunks = split_content(md_content)
            combined_result = {}
            
            for chunk in chunks:
                # 添加上下文继承（前一块的最后一段作为本块的上下文）
                context = chunks[chunks.index(chunk)-1].split('\n\n')[-1] if chunks.index(chunk) > 0 else ""
                
                print(f"正在处理分块 {chunks.index(chunk)+1}/{len(chunks)}")
                
                response = client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": '''你是一个工业设备参数提取专家，请从文本中提取多个设备参数，
                            每个设备以位号（如TT101、TT102）为标识，按以下JSON Schema返回，确保每个位号的参数独立成对象，
                            嵌套在“设备列表”数组中：\n{EXTRACTION_SCHEMA}.若出现位号缺失的情况请用 缺失位号1、缺失位号2..进行填充'''
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
        
        # 原始的单次调用逻辑（内容不超过阈值时使用）
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": '''你是一个工业设备参数提取专家，请从文本中分别提取多个设备参数，
                            每个设备以位号或编号（如TT101、TT102）为标识，按以下JSON Schema返回，确保每个位号的参数独立成对象，
                            嵌套在“设备列表”数组中：\n{EXTRACTION_SCHEMA}.若出现位号缺失的情况请用 缺失位号1、缺失位号2..进行填充'''
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
            message = response.choices[0].message
            try:
                return json.loads(message.content)
            except json.JSONDecodeError:
                print("API返回的JSON解析失败，原始内容:", message.content)
        return {}
        
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        return {}
    
with open(r"S363B-INS-DTS-1112_A_Data_Sheet_for_Temperature_transmitter.md", "r", encoding="utf-8") as f:
        md_content = f.read()

result = call_deepseek_api(md_content)
print(result)