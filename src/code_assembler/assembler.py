# src/code_assembler/assembler.py
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
import pandas as pd

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from openai import OpenAI
    from config import settings
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules in assembler.py: {e}. Ensure all dependencies are installed and PYTHONPATH is correct.", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = Path(__file__).parent / "assembly_prompt.txt"
# 定义语义库的路径
SEMANTIC_LIB_PATH = project_root / "libs" / "一体化温度变送器语义库.xlsx"

def load_prompt_template(file_path: Path) -> Optional[str]:
    """从文件中加载提示模板。"""
    if not file_path.exists():
        logger.error(f"提示模板文件未找到: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"读取提示模板文件 {file_path} 时出错: {e}")
        return None

def extract_response_content(response_content: str) -> str:
    """从 LLM 响应中提取型号代码字符串。"""
    return response_content.strip()

class CodeAssembler:
    def __init__(self, client: OpenAI):
        self.client = client
        self.prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)
        if not self.prompt_template:
            raise ValueError("无法加载组装 Prompt 模板，CodeAssembler 无法初始化。")
        
        self.transmitter_order = [
            "温度变送器", "输出信号", "说明书语言", "传感器输入", "壳体代码", 
            "接线口", "内置指示器", "安装支架", "NEPSI", "变送器附加规格"
        ]
        self.sensor_order = [
            "元件类型", "元件数量", "铠套外径(d)", "铠套材质", "加强管长度", 
            "分度号", "接线盒形式", "接头结构", "连接螺纹", "插入长度（L）", "传感器证书", "温度传感器（附加选项）"
        ]
        
        self.tg_orders_map = self._load_tg_orders_from_semantic_lib()

    def _load_tg_orders_from_semantic_lib(self) -> Dict[str, List[str]]:
        """从语义库加载保护套管的参数顺序，键是代码，值是参数列表。"""
        if not SEMANTIC_LIB_PATH.exists():
            logger.error(f"语义库文件未找到: {SEMANTIC_LIB_PATH}")
            return {}
        try:
            xls = pd.ExcelFile(SEMANTIC_LIB_PATH)
            sheet_names = xls.sheet_names
            tg_orders = {}
            for sheet_name in sheet_names:
                if "保护管" in sheet_name.strip():
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                    if df.empty or df.shape[1] < 7:
                        continue
                    
                    # 修正：从第二列（索引1）“标准参数”列获取参数顺序
                    full_param_order = df.iloc[:, 1].dropna()
                    full_param_order = full_param_order[~full_param_order.apply(lambda x: isinstance(x, (int, float)))]
                    full_param_order = full_param_order.astype(str).tolist()
                    # 移除可能读取到的表头
                    if '标准参数' in full_param_order:
                        full_param_order.remove('标准参数')

                    for _, row in df.iterrows():
                        # 键在第七列（索引6）
                        key_code = str(row.iloc[6]).strip()
                        if key_code and key_code.lower() != 'nan':
                            tg_orders[key_code] = full_param_order
            
            logger.info(f"成功从语义库加载了 {len(tg_orders)} 个保护套管代码到参数顺序的映射。")
            return tg_orders
        except Exception as e:
            logger.error(f"从语义库 {SEMANTIC_LIB_PATH} 加载保护套管顺序时出错: {e}", exc_info=True)
            return {}

    def _construct_llm_prompt(self, standardized_params: Dict[str, Any], assembly_structure: Dict[str, List[str]]) -> str:
        """为单个设备构建三段式 LLM Prompt。"""
        prompt = self.prompt_template
        
        processed_params = standardized_params.copy()
        for key, value in processed_params.items():
            if isinstance(value, list):
                processed_params[key] = "-".join(str(v) for v in value)

        prompt += f"\n\n## 组装结构与顺序：\n{json.dumps(assembly_structure, ensure_ascii=False, indent=2)}"
        prompt += f"\n\n## 标准化参数：\n{json.dumps(processed_params, ensure_ascii=False, indent=2)}"
        return prompt

    def _call_llm_api(self, prompt: str) -> Optional[str]:
        """调用 LLM API 并返回型号代码字符串。"""
        try:
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.LLM_TEMPERATURE,
                timeout=settings.LLM_REQUEST_TIMEOUT,
            )
            if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
                logger.error("LLM API response is missing expected content.")
                return None
            return extract_response_content(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"处理 LLM API 响应时发生意外错误: {e}", exc_info=True)
            return None

    def assemble_code(self, standardized_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据标准化参数组装型号代码。"""
        assembled_results = []
        if '设备列表' not in standardized_data or not isinstance(standardized_data['设备列表'], list):
            logger.error("输入数据中缺少'设备列表'或格式不正确，无法执行代码组装。")
            return assembled_results

        for device in standardized_data['设备列表']:
            tag_no = device.get('位号')
            standardized_params = device.get('参数', {})
            logger.info(f"开始为位号 {tag_no} 组装型号代码...")

            tg_type_code = standardized_params.get("TG套管形式")
            tg_order = self.tg_orders_map.get(tg_type_code, [])
            if not tg_order:
                logger.warning(f"位号 {tag_no} 的套管类型代码 '{tg_type_code}' 未在语义库中找到对应的参数顺序，将无法组装保护套管部分。")

            assembly_structure = {
                "变送器 (Transmitter)": self.transmitter_order,
                "传感器 (Sensor)": self.sensor_order,
                "热保护套管 (TG)": tg_order
            }
            
            prompt = self._construct_llm_prompt(standardized_params, assembly_structure)
            assembled_code = self._call_llm_api(prompt)
            
            current_assembly_order_list = self.transmitter_order + self.sensor_order + tg_order

            if assembled_code:
                logger.info(f"成功为位号 {tag_no} 组装型号代码: {assembled_code}")
                missing_params = [param for param in current_assembly_order_list if param not in standardized_params]
                assembled_results.append({
                    "位号": tag_no, "型号代码": assembled_code, "缺失参数": missing_params
                })
            else:
                logger.error(f"未能为位号 {tag_no} 组装型号代码。")
                assembled_results.append({
                    "位号": tag_no, "型号代码": "组装失败", "缺失参数": current_assembly_order_list
                })
        return assembled_results

import argparse

def main():
    """主函数，用于处理命令行参数、读取输入文件、调用组装器并输出结果。"""
    parser = argparse.ArgumentParser(description="根据标准化参数的 JSON 文件组装型号代码。")
    parser.add_argument("input_file", type=str, help="包含标准化参数的输入 JSON 文件的路径。")
    parser.add_argument("-o", "--output_file", type=str, help="用于保存组装结果的输出 JSON 文件的路径（可选）。")
    parser.add_argument("--use_real_llm", action="store_true", help="使用真实的 OpenAI LLM API 而不是模拟客户端（需要配置 API 密钥）。")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
    logger.info("========== 开始运行 assembler.py 主程序 ==========")
    logger.info(f"输入文件: {args.input_file}")
    if args.output_file:
        logger.info(f"输出文件: {args.output_file}")

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            standardized_data = json.load(f)
        logger.info(f"成功从 {args.input_file} 加载数据。")
    except Exception as e:
        logger.error(f"读取输入文件时发生错误: {e}", exc_info=True)
        sys.exit(1)

    if args.use_real_llm:
        if not settings.LLM_API_KEY or settings.LLM_API_KEY == "YOUR_API_KEY":
            logger.error("错误: 需要使用真实的 LLM，但 OpenAI API 密钥未在配置中设置。")
            sys.exit(1)
        try:
            client = OpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_API_URL)
            logger.info("已初始化真实的 OpenAI 客户端。")
        except Exception as e:
            logger.error(f"初始化 OpenAI 客户端失败: {e}", exc_info=True)
            sys.exit(1)
    else:
        client = MockOpenAI()
        logger.info("已初始化模拟的 OpenAI 客户端。")

    try:
        assembler = CodeAssembler(client=client)
    except ValueError as e:
        logger.error(f"初始化 CodeAssembler 失败: {e}")
        sys.exit(1)

    logger.info("开始调用代码组装方法...")
    assembled_codes = assembler.assemble_code(standardized_data)
    logger.info("代码组装完成。")

    logger.info("========== 组装结果 ==========")
    results_str = json.dumps(assembled_codes, ensure_ascii=False, indent=4)
    print(results_str)

    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(assembled_codes, f, ensure_ascii=False, indent=4)
            logger.info(f"组装结果已成功保存到: {args.output_file}")
        except Exception as e:
            logger.error(f"将结果写入输出文件时出错: {e}")

    logger.info("========== assembler.py 主程序运行结束 ==========")

class MockOpenAI:
    """一个模拟的 OpenAI 客户端，用于在没有真实 API 密钥的情况下进行测试。"""
    def __init__(self):
        self.chat = self.MockChat()

    class MockChat:
        def __init__(self):
            self.completions = self.MockCompletions()

        class MockCompletions:
            def create(self, model, messages, temperature, timeout, **kwargs):
                prompt_content = messages[0]['content']
                params_match = re.search(r'## 标准化参数：\s*(\{.*?\})\s*$', prompt_content, re.DOTALL)
                structure_match = re.search(r'## 组装结构与顺序：\s*(\{.*?\})\s*## 标准化参数', prompt_content, re.DOTALL)
                
                params = {}
                if params_match:
                    try:
                        params = json.loads(params_match.group(1))
                    except json.JSONDecodeError:
                        pass
                
                structure = {}
                if structure_match:
                    try:
                        structure = json.loads(structure_match.group(1))
                    except json.JSONDecodeError:
                        pass

                assembled_parts = []
                part_keys = ["变送器 (Transmitter)", "传感器 (Sensor)", "热保护套管 (TG)"]
                for key in part_keys:
                    order_list = structure.get(key, [])
                    
                    part_codes = []
                    for p in order_list:
                        if p in params:
                            param_value = params[p]
                            if isinstance(param_value, list):
                                part_codes.append("-".join(str(v) for v in param_value))
                            else:
                                part_codes.append(str(param_value))
                    
                    if part_codes:
                        part_code = "-".join(part_codes)
                        assembled_parts.append(part_code)
                
                final_code = " ".join(assembled_parts)

                class MockMessage:
                    def __init__(self, content):
                        self.content = content
                class MockChoice:
                    def __init__(self, message_content):
                        self.message = MockMessage(message_content)
                class MockResponse:
                    def __init__(self, model_name, content):
                        self.choices = [MockChoice(content)]
                        self.id = "mock-asm-xxxxxxxxxxxxxx"
                        self.model = model_name
                        self.object = "chat.completion"
                        self.created = int(Path(__file__).stat().st_mtime)
                
                return MockResponse(model_name=model, content=final_code)

if __name__ == "__main__":
    main()
