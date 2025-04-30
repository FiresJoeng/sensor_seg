# new_sensor_project/src/info_extractor/extractor.py
import os
import json
import re
import logging
import pandas as pd
from pathlib import Path
from openai import OpenAI
from base64 import b64encode
from typing import Dict, List, Any, Optional, Union, Tuple
import io  # Added for in-memory file operations

try:
    from config import settings, prompts
except ImportError:
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
                content = self._encode_to_base64(file_path)
                return {"data": content, "type": "application/pdf", "mode": "image_url"}
            
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

            # 添加提示词
            messages[1]["content"].insert(0, {
                "type": "text",
                "text": prompts.LLM_EXTRACTION_SYSTEM_PROMPT
            })

            return client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
        except Exception as e:
            logger.error(f"API请求构建失败：{str(e)}")
            return None

    def _handle_api_response(self, response: Any, file_path: Path, output_filename: Optional[str]) -> Optional[Dict]:
        """统一处理API响应"""
        try:
            if not response.choices:
                logger.error("API返回空响应")
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
                return {"设备列表": merged_device_list, "stats": stats}

            stats['total_groups'] = len(data['设备列表']) # 统计设备组数量

            # 遍历每个设备组
            for device_group in data['设备列表']:
                tag_nos = device_group.get('位号', []) # 获取位号列表
                common_params = device_group.get('共用参数', {}) # 获取共用参数
                diff_params = device_group.get('不同参数', {}) # 获取不同参数

                if not tag_nos:
                    logger.warning(f"设备组缺少'位号'信息，跳过处理: {device_group}")
                    continue

                stats['total_devices'] += len(tag_nos) # 累加设备总数

                # 遍历该组中的每个位号
                for tag_no in tag_nos:
                    # 为每个位号创建一个新的设备字典
                    individual_device_params = common_params.copy() # 复制共用参数作为基础

                    # 遍历不同参数，并将对应位号的值合并进来
                    for param_name, tag_value_map in diff_params.items():
                        if isinstance(tag_value_map, dict) and tag_no in tag_value_map:
                            individual_device_params[param_name] = tag_value_map[tag_no] # 添加或覆盖特定参数值
                        else:
                            # 如果不同参数的值不是字典或者当前位号不在其中，记录一个警告（可选）
                            logger.debug(f"位号 '{tag_no}' 在不同参数 '{param_name}' 中未找到特定值或格式错误。")

                    # 将合并后的设备信息添加到结果列表
                    merged_device_list.append({
                        "位号": tag_no, # 位号是字符串
                        "参数": individual_device_params # 合并后的参数字典
                    })
                    stats['processed_devices'] += 1 # 增加已处理设备计数

            logger.info(f"参数合并完成。处理了 {stats['total_groups']} 个设备组，"
                        f"共 {stats['total_devices']} 个位号，成功处理 {stats['processed_devices']} 个。")

            return {"设备列表": merged_device_list, "stats": stats} # 返回包含合并后设备列表的字典

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
                return None # 返回 None
