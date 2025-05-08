# ==============================================================================
# 5. Analysis JSON Processor (Moved Class Definition)
# ==============================================================================
import tempfile # 用于创建临时文件
import os       # 用于删除临时文件

class AnalysisJsonProcessor:
    """
    负责解析 _analysis.json 格式的输入文件，
    提取出位号列表和对应的共用参数。
    """
    def __init__(self, analysis_json_path: Path):
        """
        初始化 AnalysisJsonProcessor。

        Args:
            analysis_json_path (Path): _analysis.json 文件的路径。
        """
        self.analysis_json_path = Path(analysis_json_path)
        if not self.analysis_json_path.is_file():
            logger.error(f"Analysis JSON 文件未找到: {self.analysis_json_path}")
            raise FileNotFoundError(f"Analysis JSON 文件未找到: {self.analysis_json_path}")
        logger.info(f"AnalysisJsonProcessor 初始化完成。分析文件: {self.analysis_json_path}")

    def extract_tag_and_common_params(self) -> List[Dict[str, Any]]:
        """
        解析 _analysis.json 文件，返回一个列表，
        每个元素包含一个位号列表和对应的共用参数字典。

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
                analysis_data = json.load(f)
        except Exception as e:
            logger.error(f"加载或解析分析文件 {self.analysis_json_path} 时出错: {e}")
            return [] 

        device_list = analysis_data.get("设备列表", [])
        if not isinstance(device_list, list): # 确保是列表
            logger.error(f"分析文件 {self.analysis_json_path} 中的 '设备列表' 不是一个列表。")
            return []
        if not device_list:
            logger.warning(f"分析文件 {self.analysis_json_path} 中未找到 '设备列表' 或列表为空。")
            return []

        for i, device_group in enumerate(device_list):
            if not isinstance(device_group, dict): # 确保设备组是字典
                logger.warning(f"设备列表中的第 {i+1} 项不是一个有效的设备组字典，跳过。")
                continue

            tag_numbers = device_group.get("位号", [])
            common_params = device_group.get("共用参数", {})

            if not isinstance(tag_numbers, list) or not all(isinstance(tag, str) for tag in tag_numbers):
                logger.warning(f"设备组 {i+1} 的 '位号' 不是一个字符串列表或不存在，跳过。")
                continue
            
            if not isinstance(common_params, dict):
                logger.warning(f"设备组 {i+1} (位号: {tag_numbers[0] if tag_numbers else '未知'}) 的 '共用参数' 不是一个字典或不存在，跳过。")
                continue
            
            # 即使位号列表为空，也应该处理，后续模块可以决定如何处理没有位号但有参数的情况
            # if not tag_numbers:
            #     logger.warning(f"设备组 {i+1} 缺少 '位号'，但仍提取其共用参数。")
            # 如果共用参数为空，后续模块在尝试生成型号时会处理

            results.append({
                "位号": tag_numbers,
                "共用参数": common_params
            })
            logger.debug(f"提取设备组 {i+1} (起始位号: {tag_numbers[0] if tag_numbers else 'N/A'}): "
                         f"{len(tag_numbers)} 个位号, {len(common_params)} 个共用参数。")
        
        logger.info(f"从 {self.analysis_json_path} 提取完成，共 {len(results)} 个设备组数据。")
        return results

