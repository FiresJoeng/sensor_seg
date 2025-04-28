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
    提供直接从PDF提取JSON的功能，并包含验证功能。
    使用集中的配置和日志记录。
    """

    def __init__(self):
        """
        使用 config/settings.py 中的配置初始化 PDFInfoExtractor。
        """
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL_NAME
        self.api_url = settings.LLM_API_URL
        self.output_dir = settings.OUTPUT_DIR
        self.temperature = settings.LLM_TEMPERATURE
        self.request_timeout = settings.LLM_REQUEST_TIMEOUT

        # 检查 API 密钥是否存在
        if not self.api_key:
            logger.warning("未配置 LLM_API_KEY。LLM 功能将不可用。")
            # 可以选择抛出异常或允许继续，但 LLM 调用会失败

        self.json_proc = self.JSONProc(self)

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PDFInfoExtractor 初始化完成。输出目录: {self.output_dir}")

    def extract_parameters_from_pdf(self, pdf_path: Union[str, Path], output_filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        直接从PDF文件中提取参数。

        参数:
            pdf_path: PDF文件的路径
            output_filename: 保留参数但不再使用

        返回:
            Optional[Dict[str, Any]]: JSON格式的提取数据字典，如果失败则返回None
        """
        # 固定输出目录为项目data/output目录
        self.output_dir = Path("C:/Users/41041/Desktop/项目文件/new_sensor_project/data/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 自动生成输出文件名: 输入文件名(不含扩展名) + .json
        output_filename = f"{Path(pdf_path).stem}.json"
        pdf_path = Path(pdf_path)  # 确保是 Path 对象
        try:
            # 验证输入文件
            if not pdf_path.exists():
                logger.error(f"输入PDF文件未找到: {pdf_path}")
                raise FileNotFoundError(f"输入PDF文件未找到: {pdf_path}")

            # 读取PDF文件并编码为base64
            logger.info(f"正在读取PDF文件: {pdf_path}")
            pdf_base64 = self._encode_pdf_to_base64(pdf_path)

            # 使用LLM提取参数
            logger.info(f"正在使用模型 '{self.model}' 从PDF提取参数...")
            result_dict = self._call_llm_api_with_pdf(pdf_base64)

            if result_dict:
                logger.info(f"参数提取成功 (来自 {pdf_path.name})。")
                
                # 验证提取的数据
                validation_result = self.json_proc.json_check(result_dict)
                result_dict = validation_result["data"]
                
                # 自动保存结果到指定目录
                output_path = self.output_dir / output_filename
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result_dict, f, ensure_ascii=False, indent=2)
                logger.info(f"提取结果已自动保存至: {output_path}")
                
                return result_dict
            else:
                logger.error(f"参数提取失败或返回空结果 (来自 {pdf_path.name})。")
                return None

        except Exception as e:
            logger.error(f"从PDF ({pdf_path.name}) 提取参数时出错: {e}", exc_info=True)
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
                    
                    # 清理响应，如果它包含非JSON内容
                    # 一些模型可能包括markdown代码块或其他格式
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
                api_key="sk-PBxP00lzPV0L2zlTFiNy6O1qW4DX5Ujdf3fEVYtl2PSjXlNZ",  # Replace with your actual Gemini API key
                base_url="https://api.skyi.cc/v1",
                timeout=self.request_timeout
            )
            logger.debug(f"OpenAI客户端初始化成功 (超时: {self.request_timeout}秒)。")
            return client
        except Exception as e:
            logger.error(f"初始化OpenAI客户端失败: {e}", exc_info=True)
            raise

    class JSONProc:
        """
        JSON处理工具，用于参数验证。
        """
        def __init__(self, parent: 'InfoExtractor'):
            """使用父类引用进行初始化。"""
            self.parent = parent

        def json_check(self, data: Dict[str, Any]) -> Dict[str, Any]:
            """
            验证和检查提取的JSON数据的完整性和一致性。
            (此实现基于原代码，可能需要根据实际需求调整验证规则)

            参数:
                data: 要检查的JSON数据字典。

            返回:
                dict: 包含验证信息（问题列表、是否修改）和可能已修正的数据的字典。
                      格式: {'data': dict, 'issues': list, 'modified': bool}
            """
            issues = []
            modified = False
            validated_data = data.copy()  # 创建副本以进行修改

            logger.info("开始验证提取的JSON数据...")

            # 检查顶层结构
            if "设备列表" not in validated_data:
                issues.append("顶层缺少必需的键: '设备列表'")
                validated_data["设备列表"] = []
                modified = True
                logger.warning("JSON缺少'设备列表'键，已自动添加空列表。")

            # 验证每个设备条目
            device_list = validated_data.get("设备列表", [])
            all_param_keys = set()  # 用于收集所有参数键以检查全局一致性

            for i, device in enumerate(device_list):
                device_tag = device.get('位号', f'索引{i}')  # 用于日志记录

                # 检查设备结构
                if "位号" not in device or not device["位号"]:
                    issues.append(f"索引{i}处的设备缺少或位号为空")
                    logger.warning(f"设备索引{i}缺少或位号为空。")
                    # 可以考虑是否需要修改或移除此设备

                if "参数" not in device:
                    issues.append(f"设备'{device_tag}'缺少必需的'参数'字段")
                    device["参数"] = {}
                    modified = True
                    logger.warning(f"设备'{device_tag}'缺少'参数'字段，已自动添加空字典。")

                # 检查参数中的空值或缺失值，并收集参数键
                params = device.get("参数", {})
                current_device_keys = set()
                for key, value in params.items():
                    current_device_keys.add(key)
                    if value is None or value == "":
                        issues.append(f"设备'{device_tag}'的参数'{key}'值为空")
                        # 根据规则，空值应该被标记，这里假设LLM已经处理了，只记录问题
                        # params[key] = "缺失（文档未提供）" # 如果需要自动修复，取消注释此行
                        # modified = True
                        logger.debug(f"设备'{device_tag}'参数'{key}'值为空。")
                all_param_keys.update(current_device_keys)

            # 检查全局参数在设备间的一致性 (基于原逻辑)
            if len(device_list) > 1:
                # 识别潜在的全局参数（出现在 >= 80% 设备中的参数）
                param_counts = {}
                for device in device_list:
                    for key in device.get("参数", {}):
                        param_counts[key] = param_counts.get(key, 0) + 1

                threshold = max(1, int(len(device_list) * 0.8))
                potential_global_params = {k for k, v in param_counts.items() if v >= threshold}
                logger.debug(f"潜在的全局参数 (出现率 >= 80%): {potential_global_params}")

                # 检查这些潜在全局参数的值是否一致
                for param_key in potential_global_params:
                    values = set()
                    for device in device_list:
                        if param_key in device.get("参数", {}):
                            # 将值转换为字符串以便放入集合（处理列表或字典等不可哈希类型）
                            param_value = device["参数"][param_key]
                            try:
                                values.add(str(param_value))
                            except TypeError:
                                logger.warning(f"无法将参数'{param_key}'的值'{param_value}'添加到集合进行一致性检查。")

                    if len(values) > 1:
                        issues.append(f"潜在全局参数'{param_key}'在不同设备间的值不一致: {values}")
                        logger.warning(f"潜在全局参数'{param_key}'值不一致: {values}")

            # 可以在这里添加更多验证规则，例如：
            # - 检查数值范围
            # - 检查特定参数格式 (如日期、单位)
            # - 检查参数完整性（是否所有设备都有所有 '全局' 参数）

            # 如果发现问题，在备注中添加说明
            if issues and "备注" not in validated_data:
                validated_data["备注"] = {}
                modified = True
            if issues:
                 # 避免覆盖现有备注
                if "验证问题" not in validated_data.get("备注", {}):
                    validated_data["备注"]["验证问题"] = "自动验证发现以下问题: " + "; ".join(issues[:5]) + (f"...等共{len(issues)}个问题" if len(issues) > 5 else "")
                    modified = True
                    logger.warning(f"JSON验证发现{len(issues)}个问题。详情已添加到备注。")

            logger.info(f"JSON验证完成。发现{len(issues)}个问题。数据是否被修改: {modified}")

            return {
                "data": validated_data,
                "issues": issues,
                "modified": modified
            }
