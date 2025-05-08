# ==============================================================================
# 5. Analysis JSON Processor (Moved Class Definition)
# ==============================================================================
import tempfile # 用于创建临时文件
import os       # 用于删除临时文件
from pathlib import Path # Added
import json             # Added
from typing import List, Dict, Any # Added
import logging          # Added

#模块级 logger
logger = logging.getLogger(__name__)

class AnalysisJsonProcessor:
    """
    负责解析 _analysis.json 格式的输入文件或类似格式的 JSON 文件，
    提取出位号列表和对应的共用参数。
    能够兼容 "位号" 为单个字符串以及参数键为 "参数" 的情况。
    """
    def __init__(self, analysis_json_path: Path):
        """
        初始化 AnalysisJsonProcessor。

        Args:
            analysis_json_path (Path): _analysis.json 文件的路径。
        """
        self.analysis_json_path = Path(analysis_json_path) # Path is now imported
        if not self.analysis_json_path.is_file():
            logger.error(f"Analysis JSON 文件未找到: {self.analysis_json_path}")
            raise FileNotFoundError(f"Analysis JSON 文件未找到: {self.analysis_json_path}")
        logger.info(f"AnalysisJsonProcessor 初始化完成。分析文件: {self.analysis_json_path}")

    def extract_tag_and_common_params(self) -> List[Dict[str, Any]]: # List, Dict, Any are now imported
        """
        解析 _analysis.json 文件或类似格式的 JSON 文件，返回一个列表，
        每个元素包含一个位号列表和对应的共用参数字典。
        能够兼容 "位号" 为单个字符串以及参数键为 "参数" 的情况。

        Returns:
            List[Dict[str, Any]]: 例如:
            [
                {"位号": ["tag1", "tag2"], "共用参数": {"paramA": "valA"}},
                {"位号": ["tag3"], "共用参数": {"paramB": "valB"}}
            ]
            如果解析失败或文件内容不符合预期，则返回空列表。
        """
        results: List[Dict[str, Any]] = []
        try:
            with open(self.analysis_json_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f) # json is now imported
        except Exception as e:
            logger.error(f"加载或解析分析文件 {self.analysis_json_path} 时出错: {e}")
            return [] 

        device_list = analysis_data.get("设备列表", [])
        if not isinstance(device_list, list): 
            logger.error(f"分析文件 {self.analysis_json_path} 中的 '设备列表' 不是一个列表。")
            return []
        if not device_list:
            logger.warning(f"分析文件 {self.analysis_json_path} 中未找到 '设备列表' 或列表为空。")
            return []

        for i, device_group in enumerate(device_list):
            if not isinstance(device_group, dict): 
                logger.warning(f"设备列表中的第 {i+1} 项不是一个有效的设备组字典，跳过。")
                continue

            # 处理位号，兼容单个字符串和列表
            tag_numbers_val = device_group.get("位号")
            tag_numbers: List[str] # Type hint for clarity
            if isinstance(tag_numbers_val, str):
                tag_numbers = [tag_numbers_val]
            elif isinstance(tag_numbers_val, list) and all(isinstance(t, str) for t in tag_numbers_val):
                tag_numbers = tag_numbers_val
            else:
                # If not a string, not a list of strings, or None
                logger.warning(f"设备组 {i+1} 的 '位号' ({tag_numbers_val}) 格式不正确或不存在，跳过。")
                continue # Skip this device_group

            # 处理共用参数，兼容 "共用参数" 和 "参数" 键
            common_params_val = device_group.get("共用参数")
            if not isinstance(common_params_val, dict):
                common_params_val = device_group.get("参数") # Try '参数'

            if not isinstance(common_params_val, dict):
                logger.warning(f"设备组 {i+1} (位号: {tag_numbers[0] if tag_numbers else '未知'}) 的 '共用参数' 或 '参数' 都不是有效的字典或不存在，跳过。")
                continue # Skip this device_group
            
            common_params: Dict[str, Any] = common_params_val # Assign the validated dict

            results.append({
                "位号": tag_numbers,
                "共用参数": common_params
            })
            logger.debug(f"提取设备组 {i+1} (起始位号: {tag_numbers[0] if tag_numbers else 'N/A'}): "
                         f"{len(tag_numbers)} 个位号, {len(common_params)} 个共用参数。")
        
        logger.info(f"从 {self.analysis_json_path} 提取完成，共 {len(results)} 个设备组数据。")
        return results

    def save_params_to_files(self, extracted_data: List[Dict[str, Any]], output_dir_name: str = "data/output/temp") -> None:
        """
        将提取的数据按照位号和共用参数分别保存到指定目录下的 JSON 文件中。

        Args:
            extracted_data (List[Dict[str, Any]]): 从 extract_tag_and_common_params 返回的数据。
            output_dir_name (str): 保存输出文件的目录路径字符串 (相对于项目根目录)。
        """
        output_path = Path(output_dir_name)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"输出目录 '{output_path}' 已创建或已存在。")
        except OSError as e:
            logger.error(f"创建输出目录 '{output_path}' 失败: {e}")
            return # 如果目录创建失败，则不继续

        if not extracted_data:
            logger.info("没有提取到数据，无需保存文件。")
            return

        for i, device_group in enumerate(extracted_data):
            tag_numbers = device_group.get("位号", [])
            common_params = device_group.get("共用参数", {})

            if not tag_numbers:
                logger.warning(f"第 {i+1} 个设备组没有位号，跳过保存。")
                continue

            # 生成文件名，替换特殊字符
            # 将列表中的所有位号连接起来，并替换掉文件名中不安全的字符
            filename_base = "_".join(tag_numbers).replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")
            filename = f"{filename_base}.json"
            file_path = output_path / filename

            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(common_params, f, indent=4, ensure_ascii=False)
                logger.info(f"已保存参数到文件: {file_path}")
            except Exception as e:
                logger.error(f"保存参数到文件 {file_path} 时出错: {e}")

# ==============================================================================
# Main Processing Function (Callable)
# ==============================================================================
def process_analysis_file(input_file_path_str: str, output_directory_str: str) -> bool:
    """
    处理指定的分析JSON文件，提取数据并将其保存到单独的文件中。

    Args:
        input_file_path_str (str): 输入的JSON文件的路径字符串。
        output_directory_str (str): 保存输出文件的目录路径字符串。

    Returns:
        bool: 如果处理和保存成功完成则返回 True，否则返回 False。
    """
    func_logger = logging.getLogger(__name__ + ".process_analysis_file") # Function-specific logger
    input_path = Path(input_file_path_str)
    output_dir = Path(output_directory_str)

    if not input_path.is_file():
        func_logger.error(f"输入文件未找到: {input_path.resolve()}")
        return False
    
    func_logger.info(f"开始处理文件: {input_path.resolve()}")
    try:
        processor = AnalysisJsonProcessor(analysis_json_path=input_path)
        extracted_data = processor.extract_tag_and_common_params()
        
        if extracted_data:
            func_logger.info(f"成功从 {input_path.resolve()} 提取到 {len(extracted_data)} 组数据。")
            
            func_logger.info(f"开始将提取的数据保存到 '{output_dir.resolve()}' 目录...")
            processor.save_params_to_files(extracted_data, output_dir_name=str(output_dir))
            func_logger.info(f"数据保存过程完成。请检查 '{output_dir.resolve()}' 目录。")
            return True
        elif not extracted_data: # 明确检查空列表
             func_logger.info(f"从文件 {input_path.resolve()} 中提取到0组有效数据。")
             return True # Technically successful, just no data to process
        # No 'else' needed as extract_tag_and_common_params is typed to return List
    except FileNotFoundError as e: 
        func_logger.error(f"文件处理错误 (可能在构造函数中): {e}")
        return False
    except Exception as e:
        func_logger.error(f"处理文件 {input_path.resolve()} 时发生未知错误: {e}", exc_info=True)
        return False
    return False # Should not be reached if logic is correct

if __name__ == "__main__":
    # 配置日志输出格式，方便调试 (仅在直接运行此脚本时配置)
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # CWD is project root /home/kk/pythonProject/ai_seg/sensor_seg
    default_input_file = "data/output/test.json"
    default_output_dir = "data/output/temp"
    
    main_execution_logger = logging.getLogger(__name__ + ".__main__")
    main_execution_logger.info(f"脚本直接运行。输入: {default_input_file}, 输出目录: {default_output_dir}")
    
    success = process_analysis_file(input_file_path_str=default_input_file, 
                                    output_directory_str=default_output_dir)
    
    if success:
        main_execution_logger.info("脚本处理成功完成。")
    else:
        main_execution_logger.error("脚本处理失败。")
