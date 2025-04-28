# new_sensor_project/src/info_extractor/extractor.py
import os
import json
import re
import logging
from pathlib import Path
from openai import OpenAI
from docling.document_converter import DocumentConverter
from typing import Dict, List, Any, Optional, Union, Tuple

# 导入项目配置和提示
try:
    from ..config import settings, prompts
except ImportError:
    # 适应不同的导入上下文
    from config import settings, prompts

# 获取日志记录器实例
logger = logging.getLogger(__name__)

class InfoExtractor:
    """
    工业设备参数提取系统。
    提供文档转换、基于LLM的信息提取和验证功能。
    使用集中的配置和日志记录。
    """

    def __init__(self):
        """
        使用 config/settings.py 中的配置初始化 InfoExtractor。
        """
        self.api_key = settings.LLM_API_KEY # Use the generic LLM key name from settings
        self.model = settings.LLM_MODEL_NAME
        self.api_url = settings.LLM_API_URL
        self.output_dir = settings.OUTPUT_DIR # 输出目录现在由配置决定
        self.chunk_size = settings.LLM_CHUNK_SIZE
        self.chunk_overlap = settings.LLM_CHUNK_OVERLAP
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.temperature = settings.LLM_TEMPERATURE

        # 检查 API 密钥是否存在
        if not self.api_key:
            logger.warning("未配置 LLM_API_KEY。LLM 功能将不可用。") # Updated warning message
            # 可以选择抛出异常或允许继续，但 LLM 调用会失败

        self.md_conv = self.MDConv(self)
        self.json_proc = self.JSONProc(self)

        # 确保输出目录存在 (settings.py 中也可能创建了)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"InfoExtractor 初始化完成。输出目录: {self.output_dir}")

    class MDConv:
        """
        Markdown 转换工具，用于处理不同的文档格式。
        """
        def __init__(self, parent: 'InfoExtractor'):
            """使用父类引用进行初始化。"""
            self.parent = parent
            try:
                self.converter = DocumentConverter()
                logger.debug("DocumentConverter 初始化成功。")
            except Exception as e:
                logger.error(f"初始化 DocumentConverter 失败: {e}", exc_info=True)
                raise

        def convert_to_md(self, input_path: Union[str, Path], output_filename: Optional[str] = None) -> Optional[Path]:
            """
            将支持的文档格式 (如 PDF) 转换为 Markdown。

            参数:
                input_path: 输入文件的路径 (PDF, Word, Excel 等)。
                output_filename: 保存 Markdown 输出的可选文件名 (不含路径)。如果为 None，则基于输入文件名生成。

            返回:
                Optional[Path]: 生成的 Markdown 文件的 Path 对象，如果转换失败则返回 None。
            """
            input_path = Path(input_path) # 确保是 Path 对象
            try:
                # 验证输入文件
                if not input_path.exists():
                    logger.error(f"输入文件未找到: {input_path}")
                    raise FileNotFoundError(f"输入文件未找到: {input_path}")

                # 如果未提供输出文件名则生成
                if output_filename is None:
                    output_filename = f"{input_path.stem}.md"

                output_path = self.parent.output_dir / output_filename

                # 将文档转换为 Markdown
                logger.info(f"正在转换文档: {input_path} -> {output_path}...")
                result = self.converter.convert(str(input_path)) # convert 需要字符串路径
                markdown_content = result.document.export_to_markdown()

                # 保存到文件
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)

                logger.info(f"文档转换完成，已保存至: {output_path}")
                return output_path

            except Exception as e:
                logger.error(f"将文档 {input_path} 转换为 Markdown 时出错: {e}", exc_info=True)
                return None # 返回 None 表示失败

    class JSONProc:
        """
        JSON 处理工具，用于参数提取和验证。
        """
        def __init__(self, parent: 'InfoExtractor'):
            """使用父类引用进行初始化。"""
            self.parent = parent
            self._client: Optional[OpenAI] = None # 添加类型提示

        @property
        def client(self) -> OpenAI:
            """API 客户端的懒加载初始化。"""
            if self._client is None:
                if not self.parent.api_key:
                    logger.error("无法初始化 OpenAI 客户端：未提供 API 密钥。")
                    raise ValueError("未提供 API 密钥。请在 .env 文件中设置 LLM_API_KEY。") # Updated error message

                try:
                    # Add a default timeout (e.g., 120 seconds) to prevent hangs
                    request_timeout = settings.LLM_REQUEST_TIMEOUT # Get from settings if defined, else use default
                    self._client = OpenAI(
                        api_key=self.parent.api_key,
                        base_url=self.parent.api_url,
                        timeout=request_timeout # Set the timeout for API requests
                    )
                    logger.debug(f"OpenAI 客户端初始化成功 (Timeout: {request_timeout}s)。")
                except Exception as e:
                    logger.error(f"初始化 OpenAI 客户端失败: {e}", exc_info=True)
                    raise
            return self._client

        def md_to_json(self, md_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
            """
            使用 LLM 从 Markdown 中提取结构化数据。

            参数:
                md_path: Markdown 文件的路径。

            返回:
                Optional[Dict[str, Any]]: JSON 格式的提取数据字典，如果失败则返回 None。
            """
            md_path = Path(md_path) # 确保是 Path 对象
            try:
                # 验证输入文件
                if not md_path.exists():
                    logger.error(f"Markdown 文件未找到: {md_path}")
                    raise FileNotFoundError(f"Markdown 文件未找到: {md_path}")

                # 读取 markdown 内容
                logger.debug(f"正在读取 Markdown 文件: {md_path}")
                with open(md_path, "r", encoding="utf-8") as f:
                    md_content = f.read()

                # 使用 LLM 处理
                logger.info(f"正在使用模型 '{self.parent.model}' 从 {md_path.name} 提取参数...")
                result_dict = self._call_llm_api(md_content)

                if result_dict:
                    logger.info(f"参数提取成功 (来自 {md_path.name})。")
                    # 注意：这里只返回字典，保存文件的操作由调用者（例如 pipeline）决定
                    return result_dict
                else:
                    logger.error(f"参数提取失败或返回空结果 (来自 {md_path.name})。")
                    return None

            except Exception as e:
                logger.error(f"从 Markdown ({md_path.name}) 提取参数时出错: {e}", exc_info=True)
                return None # 返回 None 表示失败

        def _call_llm_api(self, md_content: str) -> Optional[Dict[str, Any]]:
            """
            使用优化的大文档分块方法调用 LLM API。

            参数:
                md_content: 要处理的 Markdown 内容。

            返回:
                Optional[Dict[str, Any]]: 所有块的合并结果字典，如果失败则返回 None。
            """
            try:
                # 估计总 token 数 (这是一个非常粗略的估计)
                # 注意：更准确的方法是使用 tokenizer，但这里为了简单起见保留原逻辑
                estimated_tokens = len(md_content) * 1.33 + 100
                logger.debug(f"估计的 Token 数量: {estimated_tokens:.0f}")

                system_prompt = prompts.LLM_EXTRACTION_SYSTEM_PROMPT

                # 如有必要，分块处理
                # 注意：这里的 chunk_size 是字符数，不是 token 数
                if len(md_content) > self.parent.chunk_size:
                    logger.info(f"内容长度超过 {self.parent.chunk_size} 字符，将进行分块处理。")
                    chunks = self._split_content(md_content)
                    combined_result = {"设备列表": []}
                    seen_tags = set() # 用于设备去重

                    for idx, chunk in enumerate(chunks):
                        logger.info(f"正在处理块 {idx+1}/{len(chunks)}")

                        # 获取前一个块的上下文 (如果不是第一个块)
                        context = chunks[idx-1][-self.parent.chunk_overlap:] if idx > 0 else ""

                        # 准备消息列表
                        messages = [{"role": "system", "content": system_prompt}]
                        if context:
                            # 简单地将上下文添加到用户消息前，或者可以尝试用 assistant 角色
                            # messages.append({"role": "assistant", "content": f"上文部分内容摘要：... {context}"})
                            logger.debug(f"为块 {idx+1} 添加了 {len(context)} 字符的上下文。")

                        messages.append({
                            "role": "user",
                            "content": f"这是文档的第{idx+1}部分（共{len(chunks)}部分），请从中提取参数：\n{context}\n{chunk}" # 将上下文放在前面
                        })

                        # 调用 API 处理块
                        response = self.client.chat.completions.create(
                            model=self.parent.model,
                            messages=messages,
                            response_format={"type": "json_object"},
                            temperature=self.parent.temperature,
                            max_tokens=self.parent.max_tokens
                        )

                        # 合并结果并去重
                        if response.choices:
                            try:
                                chunk_result_str = response.choices[0].message.content
                                logger.debug(f"块 {idx+1} LLM 原始响应: {chunk_result_str[:200]}...") # 记录部分响应
                                chunk_result = json.loads(chunk_result_str)

                                # 合并设备列表并去重
                                for device in chunk_result.get("设备列表", []):
                                    tag = device.get("位号")
                                    if tag and tag not in seen_tags:
                                        combined_result["设备列表"].append(device)
                                        seen_tags.add(tag)
                                        logger.debug(f"添加新设备: {tag}")
                                    elif tag:
                                        logger.debug(f"跳过重复设备: {tag}")

                                # 合并备注信息
                                if "备注" in chunk_result:
                                    if "备注" not in combined_result:
                                        combined_result["备注"] = chunk_result["备注"]
                                    else:
                                        # 简单合并，如果需要更复杂的逻辑（如更新字典）则需要修改
                                        for key, value in chunk_result["备注"].items():
                                            if key not in combined_result["备注"]:
                                                combined_result["备注"][key] = value
                                            elif isinstance(value, dict) and isinstance(combined_result["备注"].get(key), dict):
                                                combined_result["备注"][key].update(value)
                                            # 可以添加其他合并逻辑，例如列表合并等

                            except json.JSONDecodeError as json_err:
                                logger.error(f"解析块 {idx+1} 的 LLM JSON 响应失败: {json_err}", exc_info=True)
                                logger.error(f"失败的 JSON 字符串: {chunk_result_str}")
                                # 可以选择跳过此块或中止处理
                                continue # 继续处理下一个块
                            except Exception as e:
                                logger.error(f"处理块 {idx+1} 结果时发生未知错误: {e}", exc_info=True)
                                continue # 继续处理下一个块
                        else:
                            logger.warning(f"块 {idx+1} 的 LLM API 调用未返回有效响应。")

                    logger.info(f"所有块处理完成。共合并 {len(combined_result.get('设备列表',[]))} 个唯一设备。")
                    return combined_result

                # --- 单块处理 ---
                logger.info("内容长度未超过阈值，进行单次 LLM 调用。")
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请严格按照要求从以下markdown文档中提取参数：\n{md_content}"}
                ]

                response = self.client.chat.completions.create(
                    model=self.parent.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=self.parent.temperature,
                    max_tokens=self.parent.max_tokens
                )

                if response.choices:
                    try:
                        result_str = response.choices[0].message.content
                        logger.debug(f"单块 LLM 原始响应: {result_str[:200]}...")
                        result_dict = json.loads(result_str)
                        logger.info("单块 LLM 调用成功并解析 JSON。")
                        return result_dict
                    except json.JSONDecodeError as json_err:
                        logger.error(f"解析单块 LLM JSON 响应失败: {json_err}", exc_info=True)
                        logger.error(f"失败的 JSON 字符串: {result_str}")
                        return None
                else:
                    logger.warning("单块 LLM API 调用未返回有效响应。")
                    return None

            except Exception as e:
                logger.error(f"调用 LLM API 时发生错误: {e}", exc_info=True)
                return None

        def _split_content(self, content: str) -> List[str]:
            """
            将内容分割成具有语义意义的块，并保持重叠。
            (此实现基于原代码，可能需要根据实际效果调整)

            参数:
                content: 要分割的内容。

            返回:
                List[str]: 内容块列表。
            """
            chunk_size = self.parent.chunk_size
            chunk_overlap = self.parent.chunk_overlap
            logger.debug(f"开始分割内容。块大小: {chunk_size}, 重叠: {chunk_overlap}")

            # 按 Markdown 标题分割 (保留分隔符)
            # 使用 lookbehind 和 lookahead 来保留标题行
            sections = re.split(r'(?m)(?=^#+ .+)', content)
            raw_chunks = []
            current_chunk = ""
            current_length = 0 # 使用字符长度

            # 构建初始块（不包含重叠部分）
            for section in sections:
                if not section: continue # 跳过空字符串

                sec_len = len(section)

                # 如果当前块为空，或者添加新部分后不超过大小，则添加到当前块
                if not current_chunk or (current_length + sec_len <= chunk_size):
                    current_chunk += section
                    current_length += sec_len
                else:
                    # 当前块已满或添加后会超长，保存当前块，开始新块
                    raw_chunks.append(current_chunk)
                    logger.debug(f"创建原始块，长度: {len(current_chunk)}")
                    current_chunk = section
                    current_length = sec_len

            # 添加最后一个块
            if current_chunk:
                raw_chunks.append(current_chunk)
                logger.debug(f"创建最后一个原始块，长度: {len(current_chunk)}")

            # --- 在块之间添加重叠部分 ---
            # 注意：这种简单的后缀重叠可能不是最优的，取决于内容结构
            overlapped_chunks = []
            for idx, chunk in enumerate(raw_chunks):
                if idx > 0 and chunk_overlap > 0:
                    # 获取前一个块的后缀作为重叠前缀
                    # 确保不超出前一个块的长度
                    overlap_len = min(chunk_overlap, len(overlapped_chunks[-1]))
                    prefix = overlapped_chunks[-1][-overlap_len:]
                    # 将重叠部分加到当前块前面
                    overlapped_chunks.append(prefix + chunk)
                    logger.debug(f"为块 {idx+1} 添加了 {len(prefix)} 字符的重叠前缀。")
                else:
                    # 第一个块或无重叠
                    overlapped_chunks.append(chunk)

            logger.info(f"内容被分割成 {len(overlapped_chunks)} 个重叠块。")
            return overlapped_chunks

        def json_check(self, data: Dict[str, Any]) -> Dict[str, Any]:
            """
            验证和检查提取的 JSON 数据的完整性和一致性。
            (此实现基于原代码，可能需要根据实际需求调整验证规则)

            参数:
                data: 要检查的 JSON 数据字典。

            返回:
                dict: 包含验证信息（问题列表、是否修改）和可能已修正的数据的字典。
                      格式: {'data': dict, 'issues': list, 'modified': bool}
            """
            issues = []
            modified = False
            validated_data = data.copy() # 创建副本以进行修改

            logger.info("开始验证提取的 JSON 数据...")

            # 检查顶层结构
            if "设备列表" not in validated_data:
                issues.append("顶层缺少必需的键: '设备列表'")
                validated_data["设备列表"] = []
                modified = True
                logger.warning("JSON 缺少 '设备列表' 键，已自动添加空列表。")

            # 验证每个设备条目
            device_list = validated_data.get("设备列表", [])
            all_param_keys = set() # 用于收集所有参数键以检查全局一致性

            for i, device in enumerate(device_list):
                device_tag = device.get('位号', f'索引 {i}') # 用于日志记录

                # 检查设备结构
                if "位号" not in device or not device["位号"]:
                    issues.append(f"索引 {i} 处的设备缺少或位号为空")
                    logger.warning(f"设备索引 {i} 缺少或位号为空。")
                    # 可以考虑是否需要修改或移除此设备

                if "参数" not in device:
                    issues.append(f"设备 '{device_tag}' 缺少必需的 '参数' 字段")
                    device["参数"] = {}
                    modified = True
                    logger.warning(f"设备 '{device_tag}' 缺少 '参数' 字段，已自动添加空字典。")

                # 检查参数中的空值或缺失值，并收集参数键
                params = device.get("参数", {})
                current_device_keys = set()
                for key, value in params.items():
                    current_device_keys.add(key)
                    if value is None or value == "":
                        issues.append(f"设备 '{device_tag}' 的参数 '{key}' 值为空")
                        # 根据规则，空值应该被标记，这里假设 LLM 已经处理了，只记录问题
                        # params[key] = "缺失（文档未提供）" # 如果需要自动修复，取消注释此行
                        # modified = True
                        logger.debug(f"设备 '{device_tag}' 参数 '{key}' 值为空。")
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
                                logger.warning(f"无法将参数 '{param_key}' 的值 '{param_value}' 添加到集合进行一致性检查。")

                    if len(values) > 1:
                        issues.append(f"潜在全局参数 '{param_key}' 在不同设备间的值不一致: {values}")
                        logger.warning(f"潜在全局参数 '{param_key}' 值不一致: {values}")

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
                    logger.warning(f"JSON 验证发现 {len(issues)} 个问题。详情已添加到备注。")

            logger.info(f"JSON 验证完成。发现 {len(issues)} 个问题。数据是否被修改: {modified}")

            return {
                "data": validated_data,
                "issues": issues,
                "modified": modified
            }

# 注意：原 __main__ 部分已移除，因为此类应作为模块被 pipeline 调用。
