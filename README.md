# 传感器分类
传感器分类

### 描述 / Description
一个基于人工智能技术的传感器参数提取与匹配系统，能够从各种格式的文档中提取传感器参数，并与标准参数库进行匹配，生成合适的传感器型号推荐。 

### 功能特点 / Features
- 多格式文档处理 ：支持PDF、Excel、Word和图像等多种格式文档的参数提取
- 智能参数识别 ：利用大语言模型(LLM)和OCR技术精准识别文档中的参数信息
- 多种知识库支持 ：集成语义搜索、向量知识库和知识图谱三种技术路径
- 智能参数匹配 ：自动将提取的参数与标准参数库进行匹配
- 型号推荐生成 ：基于匹配结果自动生成最适合的传感器型号推荐

### 安装 / Installation
1. 将仓库克隆或下载到本地。
   
   ```bash
   git clone https://github.com/yourusername/sensor_seg.git
    ```
2. 安装所需依赖包。
   
   ```bash
   pip install -r requirements.txt
    ```
3. 配置环境变量（如需要）。

### 使用方法 / Usage
1. 准备待处理的文档（PDF、Excel、Word或图像）并放入input文件夹。
2. 运行主程序。
   
   ```bash
   python main.py
    ```
3. 查看output文件夹中的结果。

### 系统架构 / System Architecture
系统由三个主要模块组成：

1. 文档处理模块 (InfoExtractor)
   - 支持多种格式文档转换为Markdown格式
   - 使用LLM将Markdown文档转换为结构化JSON数据
   - 提供人工检查JSON数据的功能
2. 参数输出模块 (ParameterOutput)
   - 支持三种知识库技术：语义搜索、向量知识库和知识图谱
   - 比较不同知识库的结果并择优选择
   - 输出知识库之外的特殊参数
3. 标准匹配模块 (StandardMatch)
   - 从标准参数库提取匹配参数
   - 基于匹配结果生成推荐型号

### 文件结构 / File Structure
- main.py - 主程序文件
- src/ - 源代码目录
- libs/ - 知识库目录
- input/ - 输入文档目录
- output/ - 输出结果目录
- README.md - 项目说明文档

### 技术实现 / Technical Implementation
- 文档转换 ：使用docling库进行PDF、Excel、Word等格式转换
- 参数提取 ：利用大语言模型(如DeepSeek)进行智能参数识别和提取
- 知识库构建 ：实现语义搜索、向量知识库和知识图谱三种技术路径
- 参数匹配 ：采用规则匹配和大模型检索相结合的方式
- 型号生成 ：基于匹配结果智能生成传感器型号推荐

### 注意事项 / Notes
- 使用LLM功能需要配置DeepSeek API密钥。
- 首次使用时，建议先使用小型文档进行测试。

### 许可证 / License
本项目采用MIT许可证 - 详情请参阅LICENSE文件。
