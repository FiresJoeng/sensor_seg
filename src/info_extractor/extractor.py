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

            # 参数合并处理
            merge_result = self.json_proc.merge_parameters(result_dict)
            simplified_result = {
                "merged_devices": merge_result["merged_devices"],
                "备注": result_dict.get("备注", {})
            }

            # 清理元数据
            for device in simplified_result["merged_devices"]:
                device.pop("元数据", None)

            # 保存结果
            output_filename = output_filename or f"{file_path.stem}_analysis.json"
            output_path = self.output_dir / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(simplified_result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"成功保存分析结果：{output_path}")
            return simplified_result

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
            self.merge_rules = {
                '覆盖策略': '独有参数优先',
                '缺失处理': '保留原始标记',
                '深度合并': True
            }

        def merge_parameters(self, data: Dict[str, Any]) -> Dict[str, Any]:
            """增强型参数合并逻辑"""
            merged_data = []
            merge_issues = []
            stats = {'total': 0, 'merged': 0, 'errors': 0}

            # 构建共用参数索引
            common_map = self._build_common_index(data.get('共用参数设备列表', []))

            # 处理每个设备
            for device in data.get('设备列表', []):
                stats['total'] += 1
                tag = device.get('位号')
                if not tag:
                    merge_issues.append(f"发现无名设备: {device}")
                    stats['errors'] += 1
                    continue

                # 合并参数 - 支持多种参数结构
                common_params = common_map.get(tag, {})
                
                # 获取设备特有参数，支持不同的字段名称
                unique_params = {}
                if '不同参数' in device:
                    unique_params = device.get('不同参数', {})
                elif '参数' in device:
                    unique_params = device.get('参数', {})
                
                if self.merge_rules['覆盖策略'] == '独有参数优先':
                    merged_params = {**common_params, **unique_params}
                else:
                    merged_params = {**unique_params, **common_params}

                # 记录合并结果
                merged_data.append({
                    "位号": tag,
                    "参数": merged_params
                })
                stats['merged'] += 1

            # 检查未使用的共用组
            unused = [k for k, v in common_map.items() if not v.get('_used')]
            if unused:
                merge_issues.append(f"发现 {len(unused)} 个未使用的共用参数位号")

            logger.info(f"参数合并完成。成功: {stats['merged']}/{stats['total']}, 错误: {stats['errors']}")
            return {
                "merged_devices": merged_data,
                "merge_issues": merge_issues,
                "stats": stats
            }

        def _build_common_index(self, common_groups: List) -> Dict:
            """构建位号->共用参数的哈希索引"""
            index = {}
            for group in common_groups:
                params = group.get('共用参数', {})
                for tag in group.get('共用参数位号', []):
                    if tag in index:
                        logger.warning(f"位号 {tag} 存在于多个共用组，将覆盖")
                    index[tag] = params.copy()
                    index[tag]['_used'] = False  # 标记使用状态
            return index