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

if __name__ == "__main__":
    # 配置日志输出格式，方便调试
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    main_logger = logging.getLogger(__name__) # Logger for the __main__ block
    
    # CWD is project root /home/kk/pythonProject/ai_seg/sensor_seg
    input_json_path_str = "data/output/test.json"
    input_json_path = Path(input_json_path_str)

    if not input_json_path.is_file():
        main_logger.error(f"测试用的输入文件未找到: {input_json_path_str}")
    else:
        main_logger.info(f"开始处理测试文件: {input_json_path_str}")
        try:
            processor = AnalysisJsonProcessor(analysis_json_path=input_json_path)
            extracted_data = processor.extract_tag_and_common_params()
            
            if extracted_data: # Will be true for an empty list too, which is fine.
                main_logger.info(f"成功提取到 {len(extracted_data)} 组数据:")
                print(json.dumps(extracted_data, indent=4, ensure_ascii=False))
            elif extracted_data == []: 
                 main_logger.info("从测试文件中提取到0组有效数据。")
            # No 'else' needed as extract_tag_and_common_params is typed to return List
        except FileNotFoundError as e:
            main_logger.error(f"文件未找到错误: {e}")
        except Exception as e:
            main_logger.error(f"处理测试文件时发生未知错误: {e}", exc_info=True)
