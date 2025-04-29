# new_sensor_project/src/info_extractor/extractor.py
import os
import json
import re
import logging
from pathlib import Path
from openai import OpenAI
from base64 import b64encode
from typing import Dict, List, Any, Optional, Union, Tuple

# 导入项目配置和提示
try:
    from config import settings, prompts
except ImportError:
    # 适应不同的导入上下文
    from config import settings, prompts

# 获取日志记录器实例
logger = logging.getLogger(__name__)

class InfoExtractor:
    """
    工业设备参数提取系统。
    提供直接从PDF提取JSON的功能，并包含参数合并功能。
    """
    def __init__(self):
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL_NAME
        self.api_url = settings.LLM_API_URL
        self.output_dir = settings.OUTPUT_DIR
        self.temperature = settings.LLM_TEMPERATURE
        self.request_timeout = settings.LLM_REQUEST_TIMEOUT

        if not self.api_key:
            logger.warning("未配置 LLM_API_KEY。LLM 功能将不可用。")

        self.json_proc = self.JSONProc(self)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PDFInfoExtractor 初始化完成。输出目录: {self.output_dir}")

    def extract_parameters_from_pdf(self, pdf_path: Union[str, Path], output_filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        self.output_dir = Path(r"data/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"{Path(pdf_path).stem}.json"
        pdf_path = Path(pdf_path)
        try:
            if not pdf_path.exists():
                logger.error(f"输入PDF文件未找到: {pdf_path}")
                raise FileNotFoundError(f"输入PDF文件未找到: {pdf_path}")

            pdf_base64 = self._encode_pdf_to_base64(pdf_path)
            logger.info(f"正在使用模型 '{self.model}' 从PDF提取参数...")
            result_dict = self._call_llm_api_with_pdf(pdf_base64)

            if result_dict:
                logger.info(f"参数提取成功 (来自 {pdf_path.name})。")
                
                # 执行参数合并
                merge_result = self.json_proc.merge_parameters(result_dict)
                
                # 只保留合并后的内容和备注，删除其他不需要的数据
                simplified_result = {
                    "merged_devices": merge_result["merged_devices"],
                    "备注": result_dict.get("备注", {})
                }
                
                # 确保从合并设备中移除元数据字段
                for device in simplified_result["merged_devices"]:
                    if "元数据" in device:
                        del device["元数据"]

                output_path = self.output_dir / output_filename
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(simplified_result, f, ensure_ascii=False, indent=2)
                logger.info(f"简化后的提取结果已保存至: {output_path}")
                return simplified_result
            else:
                logger.error(f"参数提取失败 (来自 {pdf_path.name})。")
                return None

        except Exception as e:
            logger.error(f"从PDF提取参数时出错: {e}", exc_info=True)
            return None

    def _encode_pdf_to_base64(self, file_path: Path) -> str:
        """
        读取PDF文件并编码为base64。

        参数:
            file_path: PDF文件的路径

        返回:
            str: Base64编码的PDF内容
        """
        try:
            with open(file_path, "rb") as pdf_file:
                return b64encode(pdf_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"将PDF编码为base64时出错: {e}", exc_info=True)
            raise

    def _call_llm_api_with_pdf(self, pdf_base64: str) -> Optional[Dict[str, Any]]:
        """
        调用LLM API并传入PDF内容以提取参数。

        参数:
            pdf_base64: Base64编码的PDF内容

        返回:
            Optional[Dict[str, Any]]: 提取的数据字典，如果失败则返回None
        """
        try:
            # 获取客户端
            client = self._get_llm_client()
            if not client:
                return None

            # 从配置获取系统提示
            system_prompt = prompts.LLM_EXTRACTION_SYSTEM_PROMPT

            # 创建带有PDF文件的消息载荷
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that can analyze PDF documents."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": system_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:application/pdf;base64,{pdf_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=self.temperature
            )

            # 提取并解析响应
            if response.choices:
                try:
                    result_str = response.choices[0].message.content
                    logger.debug(f"LLM原始响应: {result_str[:200]}...")  # 记录部分响应
                    
                    # RESTORED: Call to _clean_json_response restored.
                    result_str = self._clean_json_response(result_str) 
                    
                    result_dict = json.loads(result_str)
                    logger.info("LLM调用成功并解析JSON。")
                    return result_dict
                except json.JSONDecodeError as json_err:
                    logger.error(f"解析LLM JSON响应失败: {json_err}", exc_info=True)
                    logger.error(f"失败的JSON字符串: {result_str}")
                    return None
            else:
                logger.warning("LLM API调用未返回有效响应。")
                return None

        except Exception as e:
            logger.error(f"调用LLM API时出错: {e}", exc_info=True)
            return None

    # RESTORED: _clean_json_response method definition restored.
    def _clean_json_response(self, response_str: str) -> str:
        """
        清理LLM响应以提取有效的JSON。
        这处理模型将JSON包装在markdown代码块中或添加解释的情况。

        参数:
            response_str: 来自LLM的原始响应字符串

        返回:
            str: 清理后的JSON字符串
        """
        # 检查响应是否包装在markdown代码块中
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(json_pattern, response_str)
        
        if matches:
            # 使用找到的第一个JSON代码块
            return matches[0]
        
        # 如果没有代码块，则返回原始字符串
        return response_str

    def _get_llm_client(self) -> Optional[OpenAI]:
        """
        获取初始化的OpenAI客户端。

        返回:
            Optional[OpenAI]: 初始化的客户端，如果初始化失败则返回None
        """
        if not self.api_key:
            logger.error("无法初始化OpenAI客户端: 未提供API密钥。")
            raise ValueError("未提供API密钥。请在.env文件中设置LLM_API_KEY。")

        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_url,
                timeout=300
            )
            logger.debug(f"OpenAI客户端初始化成功 (超时: {self.request_timeout}秒)。")
            return client
        except Exception as e:
            logger.error(f"初始化OpenAI客户端失败: {e}", exc_info=True)
            raise

    class JSONProc:
        """JSON处理器"""
        def __init__(self, parent: 'InfoExtractor'):
            self.parent = parent
            self.merge_rules = {
                '覆盖策略': '独有参数优先',
                '缺失处理': '保留原始标记',
                '深度合并': True
            }

        def merge_parameters(self, data: Dict[str, Any]) -> Dict[str, Any]:
            """执行参数合并的核心方法"""
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