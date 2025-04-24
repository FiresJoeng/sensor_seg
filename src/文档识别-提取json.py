import os
import json
import re
from openai import OpenAI
from docling.document_converter import DocumentConverter
from typing import Dict, List, Any, Optional, Union, Tuple


class InfoExtractor:
    """
    工业设备参数提取系统，用于提取工业设备参数。
    提供文档转换、基于LLM的信息提取和验证功能。
    """
    
    def __init__(self, api_key: str = None, model: str = "deepseek-chat", 
                 api_url: str = "https://api.deepseek.com/v1",
                 output_dir: str = None):
        """
        使用API配置初始化InfoExtractor。
        
        参数:
            api_key: LLM服务的API密钥
            model: 用于提取的模型名称
            api_url: API服务的基础URL
            output_dir: 保存输出的目录（默认为当前目录）
        """
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.output_dir = output_dir or os.getcwd()
        self.md_conv = self.MDConv(self)
        self.json_proc = self.JSONProc(self)
        
        # 如果输出目录不存在则创建
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    class MDConv:
        """
        Markdown转换工具，用于处理不同的文档格式。后续有更好的OCR方法把这个类替换掉
        """
        
        def __init__(self, parent):
            """使用父类引用进行初始化。"""
            self.parent = parent
            self.converter = DocumentConverter()
        
        def pdf_to_md(self, pdf_path: str, output_path: str = None) -> str:
            """
            将PDF转换为Markdown格式。
            
            参数:
                pdf_path: PDF文件的路径
                output_path: 保存Markdown输出的可选路径
                
            返回:
                str: 生成的Markdown文件的路径
            """
            try:
                # 验证输入文件
                if not os.path.exists(pdf_path):
                    raise FileNotFoundError(f"未找到PDF文件: {pdf_path}")
                
                # 如果未提供输出路径则生成
                if output_path is None:
                    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                    output_path = os.path.join(self.parent.output_dir, f"{base_name}.md")
                
                # 将PDF转换为Markdown
                print(f"正在转换PDF: {pdf_path} 为Markdown...")
                result = self.converter.convert(pdf_path)
                markdown_content = result.document.export_to_markdown()
                
                # 保存到文件
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
                print(f"转换完成，已保存至: {output_path}")
                return output_path
                
            except Exception as e:
                print(f"将PDF转换为Markdown时出错: {e}")
                raise
    
    class JSONProc:
        """
        JSON处理工具，用于参数提取和验证。
        """
        
        def __init__(self, parent):
            """使用父类引用进行初始化。"""
            self.parent = parent
            self._client = None
        
        @property
        def client(self):
            """API客户端的懒加载初始化。"""
            if self._client is None:
                if not self.parent.api_key:
                    raise ValueError("未提供API密钥。请在InfoExtractor构造函数中设置api_key。")
                
                self._client = OpenAI(
                    api_key=self.parent.api_key,
                    base_url=self.parent.api_url
                )
            return self._client
        
        def md_to_json(self, md_path: str, output_path: str = None, 
                      chunk_size: int = 30000, chunk_overlap: int = 500) -> dict:
            """
            使用LLM从Markdown中提取结构化数据并转换为JSON。
            
            参数:
                md_path: Markdown文件的路径
                output_path: 保存JSON输出的可选路径
                chunk_size: 每个块的最大字符长度
                chunk_overlap: 块之间重叠的字符数
                
            返回:
                dict: JSON格式的提取数据
            """
            try:
                # 验证输入文件
                if not os.path.exists(md_path):
                    raise FileNotFoundError(f"未找到Markdown文件: {md_path}")
                
                # 如果未提供输出路径则生成
                if output_path is None:
                    base_name = os.path.splitext(os.path.basename(md_path))[0]
                    output_path = os.path.join(self.parent.output_dir, f"{base_name}.json")
                
                # 读取markdown内容
                with open(md_path, "r", encoding="utf-8") as f:
                    md_content = f.read()
                
                # 使用LLM处理
                print(f"正在使用{self.parent.model}处理Markdown...")
                result = self._call_llm_api(md_content, chunk_size, chunk_overlap)
                
                # 保存到文件
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
                
                print(f"提取完成，已保存至: {output_path}")
                return result
                
            except Exception as e:
                print(f"提取参数时出错: {e}")
                raise
        
        def _call_llm_api(self, md_content: str, chunk_size: int = 30000, 
                         chunk_overlap: int = 500) -> dict:
            """
            使用优化的大文档分块方法调用LLM API。
            
            参数:
                md_content: 要处理的Markdown内容
                chunk_size: 每个块的最大字符长度
                chunk_overlap: 块之间重叠的字符数
                
            返回:
                dict: 所有块的合并结果
            """
            try:
                # 估计总token数
                estimated_tokens = len(md_content) * 1.33 + 100
                
                # 如有必要，分块处理
                if estimated_tokens > chunk_size:
                    chunks = self._split_content(md_content, chunk_size, chunk_overlap)
                    combined_result = {"设备列表": []}
                    seen_tags = set()
                    
                    for idx, chunk in enumerate(chunks):
                        print(f"正在处理块 {idx+1}/{len(chunks)}")
                        
                        # 获取前一个块的上下文
                        context = chunks[idx-1][-chunk_overlap:] if idx > 0 else ""
                        
                        # 准备消息
                        messages = [
                            {"role": "system", "content": self._get_system_prompt()},
                        ]
                        
                        if context:
                            messages.append({"role": "assistant", "content": f"上文上下文：{context}"})
                            
                        messages.append({
                            "role": "user",
                            "content": f"这是文档的第{idx+1}部分，请从中提取参数：\n{chunk}"
                        })
                        
                        # 处理块
                        response = self.client.chat.completions.create(
                            model=self.parent.model,
                            messages=messages,
                            response_format={"type": "json_object"},
                            temperature=0.1,
                            max_tokens=8096
                        )
                        
                        # 合并结果并去重
                        if response.choices:
                            try:
                                chunk_result = json.loads(response.choices[0].message.content)
                                for device in chunk_result.get("设备列表", []):
                                    tag = device.get("位号")
                                    if tag and tag not in seen_tags:
                                        combined_result["设备列表"].append(device)
                                        seen_tags.add(tag)
                                
                                # 如果存在备注则合并
                                if "备注" in chunk_result and "备注" not in combined_result:
                                    combined_result["备注"] = chunk_result["备注"]
                                elif "备注" in chunk_result and "备注" in combined_result:
                                    # 合并不同类型的备注
                                    for key, value in chunk_result["备注"].items():
                                        if key not in combined_result["备注"]:
                                            combined_result["备注"][key] = value
                                        elif isinstance(value, dict):
                                            combined_result["备注"][key].update(value)
                                
                            except json.JSONDecodeError:
                                print("JSON解析失败，原始内容：", 
                                      response.choices[0].message.content)
                    
                    return combined_result
                
                # 单块处理
                messages = [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": f"请严格按照要求从以下markdown文档中提取参数：\n{md_content}"}
                ]
                
                response = self.client.chat.completions.create(
                    model=self.parent.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=8096
                )
                
                if response.choices:
                    try:
                        return json.loads(response.choices[0].message.content)
                    except json.JSONDecodeError:
                        print("JSON解析失败，内容：", 
                              response.choices[0].message.content)
                
                return {}
                
            except Exception as e:
                print(f"API调用失败: {e}")
                return {}
        
        def _split_content(self, content: str, chunk_size: int = 30000, 
                          chunk_overlap: int = 500) -> List[str]:
            """
            将内容分割成具有语义意义的块，并保持重叠。
            
            参数:
                content: 要分割的内容
                chunk_size: 每个块的最大大小
                chunk_overlap: 块之间的重叠部分
                
            返回:
                List[str]: 内容块列表
            """
            # 按Markdown标题分割
            sections = re.split(r'(?m)^(#+ .+)$', content)
            raw_chunks = []
            current_chunk = []
            current_length = 0
            
            # 构建初始块（不包含重叠部分）
            for i in range(0, len(sections), 2):
                section = sections[i] + (sections[i+1] if i+1 < len(sections) else "")
                sec_len = len(section) * 1.33  # 粗略的token估计
                
                # 单独处理关键部分
                if re.search(r'^#{1,2}\s*(参数|规格|技术指标)', section, re.IGNORECASE):
                    if current_chunk:
                        raw_chunks.append(''.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    raw_chunks.append(section)
                    continue
                
                # 如果超过大小则开始新块
                if current_length + sec_len > chunk_size:
                    raw_chunks.append(''.join(current_chunk))
                    current_chunk = [section]
                    current_length = sec_len
                else:
                    current_chunk.append(section)
                    current_length += sec_len
            
            # 添加最后一个块
            if current_chunk:
                raw_chunks.append(''.join(current_chunk))
            
            # 在块之间添加重叠部分
            overlapped_chunks = []
            for idx, chunk in enumerate(raw_chunks):
                if idx > 0:
                    # 添加前一个块的后缀作为上下文
                    prefix = overlapped_chunks[-1][-chunk_overlap:]
                    chunk = prefix + chunk
                overlapped_chunks.append(chunk)
            
            return overlapped_chunks
        
        def _get_system_prompt(self) -> str:
            """
            获取参数提取的系统提示。
            
            返回:
                str: 用于LLM的系统提示
            """
            return '''你是一个工业设备参数提取专家，负责从用户提供的复杂md文档中提取温度变送器参数，你具有专业工程师复杂的判断
                      能力和理解能力，能够站在全局的角度去思考用户提供的参数需求，需遵循以下规则：

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
   - 注意识别有些位号存在大量缺失参数可能是因为该部分所有位号的参数都是一样的情况。但需在备注中说明。

4. **数值格式强制统一**  
   - 数值需完整保留原文档格式（如`Φ8mm`不可简化为`8mm`）。  
   - 对疑似异常值（如`操作压力: '200'`）添加注释，但不修改原值。

5. **完整性校验**  
   - 每个位号的`参数`必须包含所有字段（全局+专属），缺失字段需明确标注。

### 输出格式示例：（严格遵循JSON Schema）
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
        
        def json_check(self, json_path: str, output_path: str = None) -> Dict:
            """
            验证和检查提取的JSON的完整性和一致性。
            
            参数:
                json_path: 要检查的JSON文件路径
                output_path: 保存验证后JSON的可选路径
                
            返回:
                dict: 验证过的、可能已修正的JSON数据


            验证规则：
            结构验证：检查 JSON 是否包含必需的顶层结构（如"设备列表"字段）
            完整性检查：检查每个设备条目是否包含所有必需的字段（位号、参数等）
            空值检测：发现并标记参数中的空值或缺失值
            全局参数一致性检查：确保那些应该在所有设备中保持一致的全局参数没有矛盾

            输出内容:
            json_check 会输出一个包含以下信息的字典，供人工检查：
            数据 (data)：验证后的 JSON 数据，可能包含自动修正的内容
            问题 (issues)：所有发现的问题列表，包括缺失字段、空值、结构问题等
            是否修改 (modified)：一个布尔值，表示验证过程中是否对原始数据进行了修改
            """
            try:
                # 验证输入文件
                if not os.path.exists(json_path):
                    raise FileNotFoundError(f"未找到JSON文件: {json_path}")
                
                # 如果未提供输出路径则生成
                if output_path is None:
                    base_name = os.path.splitext(os.path.basename(json_path))[0]
                    output_path = os.path.join(self.parent.output_dir, f"{base_name}_validated.json")
                
                # 读取JSON文件
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # 执行验证
                print("正在验证JSON结构和内容...")
                validation_result = self._validate_json(data)
                
                # 如果修改了，保存验证后的JSON
                if validation_result["modified"]:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(validation_result["data"], f, ensure_ascii=False, indent=4)
                    print(f"验证后的JSON已保存至: {output_path}")
                else:
                    print("JSON验证完成，无需修改。")
                
                # 返回验证信息
                return {
                    "data": validation_result["data"],
                    "issues": validation_result["issues"],
                    "modified": validation_result["modified"]
                }
                
            except Exception as e:
                print(f"JSON验证过程中出错: {e}")
                raise
        
        def _validate_json(self, data: Dict) -> Dict:
            """
            内部验证JSON数据结构和内容。
            
            参数:
                data: 要验证的JSON数据
                
            返回:
                dict: 包含问题和修改标志的验证结果
            """
            issues = []
            modified = False
            
            # 检查顶层结构
            if "设备列表" not in data:
                issues.append("缺少必需的键: '设备列表'")
                data["设备列表"] = []
                modified = True
            
            # 验证每个设备条目
            device_list = data.get("设备列表", [])
            for i, device in enumerate(device_list):
                # 检查设备结构
                if "位号" not in device:
                    issues.append(f"索引{i}处的设备缺少必需的'位号'字段")
                
                if "参数" not in device:
                    issues.append(f"位号为'{device.get('位号', f'索引 {i}')}的设备缺少'参数'字段")
                    device["参数"] = {}
                    modified = True
                
                # 检查空值或缺失值
                params = device.get("参数", {})
                for key, value in params.items():
                    if not value or value == "":
                        issues.append(f"设备'{device.get('位号', f'索引 {i}')}'中参数'{key}'的值为空")
                        params[key] = "缺失（文档未提供）"
                        modified = True
            
            # 检查全局参数在设备间的一致性
            if len(device_list) > 1:
                global_params = self._identify_global_params(device_list)
                
                # 确保全局参数的一致性
                for param in global_params:
                    values = set()
                    for device in device_list:
                        if param in device.get("参数", {}):
                            values.add(device["参数"][param])
                    
                    # 如果存在不一致，报告它们
                    if len(values) > 1:
                        issues.append(f"全局参数'{param}'的值不一致: {values}")
            
            # 确保存在"备注"部分
            if "备注" not in data and issues:
                data["备注"] = {
                    "验证问题": "自动验证发现以下问题: " + "; ".join(issues[:3]) + (f"...等{len(issues)}个问题" if len(issues) > 3 else "")
                }
                modified = True
            
            return {
                "data": data,
                "issues": issues,
                "modified": modified
            }
        
        def _identify_global_params(self, device_list: List[Dict]) -> List[str]:
            """
            识别可能在所有设备中都是全局的参数。
            
            参数:
                device_list: 设备条目列表
                
            返回:
                List[str]: 看起来是全局的参数列表
            """
            if not device_list:
                return []
            
            # 计算参数出现次数
            param_count = {}
            device_count = len(device_list)
            
            for device in device_list:
                params = device.get("参数", {})
                for param in params:
                    param_count[param] = param_count.get(param, 0) + 1
            
            # 出现在至少80%设备中的参数被视为全局参数
            threshold = max(1, int(device_count * 0.8))
            global_params = [param for param, count in param_count.items() if count >= threshold]
            
            return global_params


# 使用示例
if __name__ == "__main__":
    # 初始化提取器
    extractor = InfoExtractor(
        api_key="sk-28e66466f44148b4b6135f6e92d18651",  # 替换为你的API密钥
        output_dir="文档识别和json测试结果"
    )
    
    # 将PDF转换为Markdown
    pdf_path = "C:\\Users\\41041\\Desktop\\项目文件\\系统仓库\\sensor_seg\\input\\P2021-180008温度变送器规格书.pdf"
    md_path = extractor.md_conv.pdf_to_md(pdf_path)
    
    # 从Markdown提取参数到JSON
    # 注意：md_to_json返回解析后的JSON数据和保存路径
    json_data = extractor.json_proc.md_to_json(md_path)
    
    # 构造正确的JSON文件路径
    base_name = os.path.splitext(os.path.basename(md_path))[0]
    json_path = os.path.join(extractor.output_dir, f"{base_name}.json")

    # 验证JSON（可选）
    validation = extractor.json_proc.json_check(json_path)
    
    print("处理成功完成！")
