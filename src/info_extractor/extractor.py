
import sys
from pathlib import Path

# Ensure project root is in sys.path for absolute imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
import json
import re
import logging
import pandas as pd
from openai import OpenAI
from base64 import b64encode
from typing import Dict, List, Any, Optional, Union, Tuple
import io  # Added for in-memory file operations
import pdfplumber # 导入 pdfplumber 库
import camelot # 导入 camelot 库用于表格提取

from config import settings, prompts

logger = logging.getLogger(__name__)

class InfoExtractor:
    """支持PDF/Excel/图片的工业参数提取系统"""
    def __init__(self):
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL_NAME
        self.api_url = settings.LLM_API_URL
        self.output_dir = settings.OUTPUT_DIR
        self.temperature = settings.LLM_TEMPERATURE
        self.request_timeout = settings.LLM_REQUEST_TIMEOUT

        if not self.api_key:
            logger.warning("LLM功能不可用：未配置API密钥")

        self.json_proc = self.JSONProc(self)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"初始化完成，输出目录：{self.output_dir}")

    def detect_file_type(self, file_path: Path) -> str:
        """精准文件类型检测"""
        ext = file_path.suffix.lower()
        if ext in ['.xls', '.xlsx', '.xlsm', '.xlsb', '.csv']:
            return "excel"
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.tif']:
            return "image"
        elif ext == '.pdf':
            return "pdf"
        else:
            raise ValueError(f"不支持的文件类型：{ext}")

    def extract_parameters(self, file_path: Union[str, Path], output_filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """增强型统一提取入口"""
        file_path = Path(file_path)
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在：{file_path}")

            file_type = self.detect_file_type(file_path)
            logger.info(f"开始处理 {file_type.upper()} 文件：{file_path.name}")

            # 处理文件内容
            processed_data = self._process_file_content(file_path, file_type)
            if not processed_data:
                return None

            # 动态构建API请求
            api_response = self._build_api_request(processed_data, file_type)
            if not api_response:
                return None

            # 处理结果并保存
            return self._handle_api_response(api_response, file_path, output_filename)

        except Exception as e:
            logger.error(f"处理失败：{str(e)}", exc_info=True)
            return None
            
    # 兼容性方法 - 为了支持现有代码
    def extract_parameters_from_pdf(self, file_path: Union[str, Path], output_filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """PDF文件参数提取（兼容性方法）"""
        logger.info(f"通过兼容性方法调用PDF处理: {file_path}")
        return self.extract_parameters(file_path, output_filename)
        
    def extract_parameters_from_excel(self, file_path: Union[str, Path], output_filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Excel文件参数提取（兼容性方法）"""
        logger.info(f"通过兼容性方法调用Excel处理: {file_path}")
        return self.extract_parameters(file_path, output_filename)
        
    def extract_parameters_from_image(self, file_path: Union[str, Path], output_filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """图片文件参数提取（兼容性方法）"""
        logger.info(f"通过兼容性方法调用图片处理: {file_path}")
        return self.extract_parameters(file_path, output_filename)

    def _process_file_content(self, file_path: Path, file_type: str) -> Optional[Dict]:
        """文件内容处理器"""
        try:
            if file_type == "pdf":
                # 提取 Base64 图像数据
                image_content = self._encode_to_base64(file_path)
                
                # 使用 pdfplumber 提取文本内容
                text_content = ""
                try:
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            text_content += page.extract_text() or "" # 提取页面文本，如果为空则使用空字符串
                    logger.info(f"成功从PDF提取文本内容: {file_path.name}")
                except Exception as e:
                    logger.warning(f"使用 pdfplumber 提取文本失败: {e}", exc_info=True)
                    text_content = "无法提取文本内容。" # 提取失败时提供默认文本

                # 使用 camelot 提取表格数据
                table_content_csv = ""
                try:
                    tables = camelot.read_pdf(str(file_path), flavor='lattice', pages='all') # 使用 lattice 模式提取所有页面的表格
                    logger.info(f"成功使用 camelot 提取到 {len(tables)} 个表格: {file_path.name}")
                    for i, table in enumerate(tables):
                        # 将每个表格转换为 CSV 字符串
                        table_csv = table.df.to_csv(index=False, encoding='utf-8')
                        table_content_csv += f"--- Table {i+1} ---\n" # 添加表格分隔符
                        table_content_csv += table_csv + "\n\n"
                    if not table_content_csv:
                         logger.warning(f"使用 camelot 未提取到任何表格数据: {file_path.name}")
                         table_content_csv = "未从PDF中提取到表格数据。" # 未提取到表格时提供默认文本

                except Exception as e:
                    logger.warning(f"使用 camelot 提取表格失败: {e}", exc_info=True)
                    table_content_csv = "使用 camelot 提取表格失败。" # 提取失败时提供默认文本


                return {
                    "data": image_content, # Base64 图像数据
                    "type": "application/pdf",
                    "mode": "image_url",
                    "text_content": text_content, # pdfplumber 提取的文本内容
                    "table_content_csv": table_content_csv # camelot 提取的表格内容 (CSV 格式)
                }
            
            elif file_type == "excel":
                # 在内存中处理Excel转CSV，不保存到磁盘
                try:
                    df = pd.read_excel(file_path)
                    # 使用StringIO将CSV内容保留在内存中
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False, encoding='utf-8')
                    csv_content = csv_buffer.getvalue()
                    logger.info(f"成功将Excel转换为CSV格式（内存中）: {file_path.name}")
                    
                    return {
                        "text_content": csv_content,
                        "type": "text/csv",
                        "mode": "text"
                    }
                except Exception as e:
                    logger.error(f"转换Excel为CSV时出错: {e}")
                    return None
            
            elif file_type == "image":
                content = self._encode_to_base64(file_path)
                mime_type = self._get_image_mime_type(file_path)
                return {"data": content, "type": mime_type, "mode": "image_url"}
            
            raise ValueError("未知文件类型")
        except pd.errors.EmptyDataError:
            logger.error("Excel文件无有效数据")
            return None
        except Exception as e:
            logger.error(f"内容处理异常：{str(e)}")
            return None

    def _build_api_request(self, processed_data: Dict, file_type: str) -> Optional[Any]:
        """动态构建API请求载荷"""
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_url,
                timeout=self.request_timeout
            )

            messages = [
                {"role": "system", "content": prompts.LLM_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": []}
            ]

            # 根据文件类型添加内容
            if processed_data["mode"] == "image_url":
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{processed_data['type']};base64,{processed_data['data']}"
                    }
                })
            elif processed_data["mode"] == "text":
                messages[1]["content"].append({
                    "type": "text",
                    "text": processed_data["text_content"]
                })

            # 添加用户提示词（重复系统提示，如原代码逻辑）
            messages[1]["content"].append({
                "type": "text",
                "text": prompts.LLM_EXTRACTION_SYSTEM_PROMPT
            })

            # 根据文件类型添加内容
            if processed_data["mode"] == "image_url":
                # 添加图像内容
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{processed_data['type']};base64,{processed_data['data']}"
                    }
                })
                # 如果提取了文本内容，也添加到消息中
                if "text_content" in processed_data and processed_data["text_content"]:
                     messages[1]["content"].append({
                        "type": "text",
                        "text": f"以下是使用文本解析工具从PDF中提取的文本内容，供您参考和校对：\n\n{processed_data['text_content']}"
                    })
                
                # 如果提取了表格内容，也添加到消息中
                if "table_content_csv" in processed_data and processed_data["table_content_csv"]:
                     messages[1]["content"].append({
                        "type": "text",
                        "text": f"以下是使用表格提取工具从PDF中提取的结构化表格数据（CSV格式）。请优先参考这些数据来准确识别参数和值，特别是处理表格中的信息时：\n\n{processed_data['table_content_csv']}"
                    })


            elif processed_data["mode"] == "text":
                # 添加文本内容
                messages[1]["content"].append({
                    "type": "text",
                    "text": processed_data["text_content"]
                })

            api_result = client.chat.completions.create( # 将 api_result 赋值给变量
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                reasoning_effort='high'
            )
            # 添加日志记录 API 调用结果的类型
            logger.debug(f"API call successful. Type of result: {type(api_result)}")
            return api_result
        except Exception as e:
            logger.error(f"API请求构建失败：{str(e)}", exc_info=True) # 记录详细错误信息
            return None

    def _handle_api_response(self, response: Any, file_path: Path, output_filename: Optional[str]) -> Optional[Dict]:
        """统一处理API响应"""
        # 添加日志记录接收到的 response 的类型和部分内容
        logger.debug(f"Handling API response. Type: {type(response)}, Value (partial): {str(response)[:200]}...")
        try:
            # 在访问属性前检查响应类型是否符合预期
            if not hasattr(response, 'choices'):
                 logger.error(f"API 响应格式不符合预期，缺少 'choices' 属性。响应类型: {type(response)}")
                 logger.error(f"响应内容 (部分): {str(response)[:500]}...")
                 return None

            # 现在可以安全地检查 choices
            if not response.choices:
                logger.error("API返回空响应 (choices is empty or None)")
                return None

            raw_content = response.choices[0].message.content
            cleaned_json = self._clean_json_response(raw_content)
            result_dict = json.loads(cleaned_json)

            # 直接使用原始响应数据
            output_data = result_dict  # 不再进行参数合并

            # 保存原始结果
            output_filename = output_filename or f"{file_path.stem}_analysis.json"
            output_path = self.output_dir / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"成功保存原始分析结果：{output_path}")
            return output_data  # 返回原始数据（接口兼容）

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败：{str(e)}")
            logger.debug(f"原始响应内容：{raw_content[:200]}...")
            return None
        except Exception as e:
            logger.error(f"结果处理异常：{str(e)}")
            return None

    # ---------- 工具方法 ---------- 
    def _get_image_mime_type(self, file_path: Path) -> str:
        """精确图片MIME类型映射"""
        type_map = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.gif': 'image/gif',
            '.webp': 'image/webp', '.bmp': 'image/bmp',
            '.tiff': 'image/tiff', '.tif': 'image/tiff'
        }
        return type_map.get(file_path.suffix.lower(), 'application/octet-stream')

    def _encode_to_base64(self, file_path: Path) -> str:
        """高效Base64编码"""
        with open(file_path, "rb") as f:
            return b64encode(f.read()).decode('utf-8')

    def _clean_json_response(self, response_str: str) -> str:
        """增强型JSON清洗"""
        # 处理代码块包裹
        code_block = re.search(r'```json(.*?)```', response_str, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()
        
        # 处理无包裹的JSON
        json_start = response_str.find('{')
        json_end = response_str.rfind('}')
        if json_start != -1 and json_end != -1:
            return response_str[json_start:json_end+1]
        
        # 兜底处理
        return response_str.strip(' \n\t`')

    # ---------- JSON处理器（保持核心逻辑） ---------- 
    class JSONProc:
        def __init__(self, parent: 'InfoExtractor'):
            self.parent = parent
            # 合并规则可以根据需要进行配置，但当前逻辑固定
            pass

        def merge_parameters(self, data: Dict[str, Any]) -> Dict[str, Any]:
            """
            根据新的JSON结构合并参数。
            将包含"共用参数"和"不同参数"的设备组列表，转换为每个位号包含所有参数的设备列表。
            
            Args:
                data (Dict[str, Any]): 从LLM提取并保存在JSON文件中的数据，
                                        结构类似 {'设备列表': [...], '备注': {...}}
            
            Returns:
                Dict[str, Any]: 合并后的数据，结构类似 {'设备列表': [...]}，
                                其中每个设备包含 "位号" 和 "参数"。
            """
            merged_device_list = [] # 用于存储最终合并结果的列表
            stats = {'total_groups': 0, 'total_devices': 0, 'processed_devices': 0} # 统计信息

            if '设备列表' not in data or not isinstance(data['设备列表'], list):
                logger.warning("输入数据中缺少'设备列表'或格式不正确，无法执行合并。")
                # 即使没有设备列表，也尝试保留备注信息
                remarks = data.get('备注')
                return {"设备列表": merged_device_list, "stats": stats, "备注": remarks}

            stats['total_groups'] = len(data['设备列表']) # 统计设备组数量
            remarks = data.get('备注') # 保留备注信息

            # 遍历每个设备组 (现在这些组可能已经过标准化)
            for device_group in data['设备列表']:
                tag_nos = device_group.get('位号', []) # 获取位号列表

                # 优先获取标准化后的共用参数，如果不存在，则获取原始共用参数
                standardized_common_params = device_group.get('标准化共用参数')
                common_params_to_merge = standardized_common_params if isinstance(standardized_common_params, dict) else device_group.get('共用参数', {})

                # 优先获取标准化后的不同参数，如果不存在，则获取原始不同参数
                standardized_diff_params = device_group.get('标准化不同参数')
                diff_params_to_merge = standardized_diff_params if isinstance(standardized_diff_params, dict) else device_group.get('不同参数', {})


                if not tag_nos:
                    logger.warning(f"设备组缺少'位号'信息，跳过处理: {device_group}")
                    continue

                stats['total_devices'] += len(tag_nos) # 累加设备总数

                # 遍历该组中的每个位号
                for tag_no in tag_nos:
                    # 为每个位号创建一个新的设备字典
                    individual_device_params = {} # 初始化参数字典

                    # 合并共用参数
                    if common_params_to_merge:
                        individual_device_params.update(common_params_to_merge)
                    elif standardized_common_params is None and '共用参数' in device_group:
                         # 标准化失败且原始共用参数存在
                         logger.debug(f"位号 '{tag_no}' 所在组共用参数标准化失败，使用原始共用参数。")
                    else:
                         logger.debug(f"位号 '{tag_no}' 所在组无有效共用参数或标准化共用参数。")


                    # 遍历不同参数，并将对应位号的值合并进来 (覆盖共用参数中的同名项)
                    if diff_params_to_merge:
                        for param_name, tag_value_map in diff_params_to_merge.items():
                            if isinstance(tag_value_map, dict) and tag_no in tag_value_map:
                                individual_device_params[param_name] = tag_value_map[tag_no] # 添加或覆盖特定参数值
                            else:
                                # 如果不同参数的值不是字典或者当前位号不在其中，记录一个调试信息（可选）
                                logger.debug(f"位号 '{tag_no}' 在不同参数 '{param_name}' 中未找到特定值或格式错误。")
                    elif standardized_diff_params is None and '不同参数' in device_group:
                         # 标准化失败且原始不同参数存在
                         logger.debug(f"位号 '{tag_no}' 所在组不同参数标准化失败，使用原始不同参数。")
                    else:
                        logger.debug(f"设备组 '{', '.join(tag_nos)}' 无有效不同参数或标准化不同参数。")


                    # 将合并后的设备信息添加到结果列表
                    merged_device_list.append({
                        "位号": tag_no, # 位号是字符串
                        "参数": individual_device_params # 合并后的参数字典
                    })
                    stats['processed_devices'] += 1 # 增加已处理设备计数

            logger.info(f"参数合并完成。处理了 {stats['total_groups']} 个设备组，"
                        f"共 {stats['total_devices']} 个位号，成功处理 {stats['processed_devices']} 个。")

            # 返回包含合并后设备列表和备注的字典
            return {"设备列表": merged_device_list, "stats": stats, "备注": remarks}

        def extract_remarks(self, data: Dict[str, Any]) -> Optional[Dict[str, str]]:
            """
            从输入的JSON数据中提取 "备注" 部分。

            Args:
                data (Dict[str, Any]): 从LLM提取的原始JSON数据。

            Returns:
                Optional[Dict[str, str]]: 包含备注信息的字典，如果不存在则返回 None。
            """
            remarks = data.get("备注") # 尝试获取 "备注" 键对应的值
            if isinstance(remarks, dict):
                logger.info("成功提取到备注信息。")
                return remarks # 如果是字典，则返回它
            elif remarks:
                # 如果存在但不是字典，记录警告
                logger.warning(f"找到'备注'键，但其值不是预期的字典格式: {type(remarks)}")
                return None # 返回 None 表示未找到有效的备注字典
            else:
                # 如果 "备注" 键不存在
                logger.info("未在数据中找到'备注'信息。")
