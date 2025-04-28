# 一体式温度变送器参数处理与代码生成系统 (重构版)

**版本**: 1.0.0 (重构)

## 1. 项目目标

该系统旨在自动化处理一体式温度变送器订单文件（如 PDF），提取参数，通过知识库进行标准化，并最终生成符合规范的产品型号代码。

## 2. 系统架构

系统采用模块化设计，包含以下核心组件：

1.  **信息提取模块 (`src.info_extractor`)**:
    *   负责将输入文档（当前支持 PDF）转换为 Markdown 格式。
    *   利用大语言模型 (LLM, DeepSeek) 从 Markdown 中提取结构化的参数信息，输出为 JSON 格式。
    *   (可选) 提供 JSON 格式校验功能。

2.  **参数标准化模块 (`src.parameter_standardizer`)**:
    *   包含一个基于源 Excel 文件构建的参数知识库（使用向量嵌入和 ChromaDB）。
    *   提供 `SearchService`，接收提取出的实际参数名和值，查询知识库返回最匹配的标准参数名、标准参数值和标准代码。
    *   包含构建和管理知识库所需的 `DataProcessor`, `EmbeddingGenerator`, `VectorStoreManager`。

3.  **标准匹配与代码生成模块 (`src.standard_matcher`)**:
    *   接收来自参数标准化模块的标准化参数集（标准参数名 -> 标准参数值）。
    *   根据标准化参数确定产品主型号（查询 `libs/standard/**/index.json`）。
    *   加载主型号对应的标准库 CSV 文件 (`libs/standard/**/`)。
    *   将标准化参数值与标准库 CSV 中的条目进行模糊匹配，为每个标准参数确定唯一的标准代码。
    *   按标准库定义的顺序拼接代码，生成最终产品型号代码字符串。

4.  **流水线模块 (`src.pipeline`)**:
    *   `main_pipeline.py` 脚本负责协调以上三个模块，执行端到端的处理流程。

5.  **配置模块 (`config`)**:
    *   `settings.py`: 集中管理所有配置，如文件路径、API 密钥、模型名称、日志级别等。
    *   `prompts.py`: 存储用于 LLM 的提示信息。

6.  **工具模块 (`src.utils`)**:
    *   `logging_config.py`: 配置项目范围的日志记录。

7.  **脚本模块 (`scripts`)**:
    *   `build_kb.py`: 用于构建或更新参数知识库（向量数据库）。

## 3. 文件结构

```
new_sensor_project/
├── .env                    # 环境变量 (需手动创建, 包含 DEEPSEEK_API_KEY)
├── .gitignore
├── README.md
├── requirements.txt
├── config/                 # 配置模块
│   ├── settings.py
│   └── prompts.py
├── data/                   # 运行时输入/输出目录
│   ├── input/              # 放置待处理的 PDF 文件
│   └── output/             # 存放处理结果 (日志, 中间文件, 最终输出)
├── knowledge_base/         # 参数知识库文件
│   ├── source/             # 知识库源文件 (semantic_source.xlsx)
│   └── vector_store/       # ChromaDB 向量数据库文件
├── libs/                   # 标准库 CSV 文件
│   └── standard/
│       ├── transmitter/    # 示例：变送器标准库
│       │   ├── index.json  # 型号到 CSV 文件的索引
│       │   └── *.csv       # 具体型号的标准代码表
│       └── ...
├── src/                    # 项目源代码
│   ├── info_extractor/     # 信息提取模块
│   ├── parameter_standardizer/ # 参数标准化模块
│   ├── standard_matcher/   # 标准匹配与代码生成模块
│   ├── pipeline/           # 流水线协调模块
│   └── utils/              # 通用工具模块 (日志等)
└── scripts/                # 辅助脚本
    └── build_kb.py         # 构建知识库脚本
```

## 4. 安装

1.  **克隆仓库**:
    ```bash
    # git clone ... (如果项目在 Git 仓库中)
    # 或者直接将代码解压到 new_sensor_project 目录
    ```

2.  **创建虚拟环境** (推荐):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate  # Windows
    ```

3.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **配置环境变量**:
    *   在 `new_sensor_project` 根目录下创建一个名为 `.env` 的文件。
    *   在 `.env` 文件中添加 DeepSeek API 密钥：
        ```
        DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        ```
    *   (可选) 在 `.env` 文件中设置日志级别 (默认为 INFO):
        ```
        LOG_LEVEL=DEBUG
        ```

5.  **准备数据文件**:
    *   确保 `knowledge_base/source/semantic_source.xlsx` 文件存在且包含正确的参数知识。
    *   确保 `libs/standard/` 目录下包含所需的标准库 CSV 文件和 `index.json` 文件。

## 5. 使用方法

### 5.1 构建知识库 (首次运行或更新知识库时需要)

在项目根目录 (`new_sensor_project/`) 下运行构建脚本：

```bash
python scripts/build_kb.py
```

该脚本会执行以下操作：
*   读取 `knowledge_base/source/semantic_source.xlsx`。
*   处理数据，生成嵌入向量。
*   清空并重建 `knowledge_base/vector_store/` 中的 ChromaDB 向量数据库。

### 5.2 处理文档并生成代码

1.  将需要处理的 PDF 文件放入 `data/input/` 目录。
2.  在项目根目录 (`new_sensor_project/`) 下运行主流水线脚本，并指定输入文件：

    ```bash
    python -m src.pipeline.main_pipeline data/input/your_document.pdf
    ```
    (将 `your_document.pdf` 替换为实际的文件名)

3.  **查看结果**:
    *   处理过程的日志会输出到控制台，并根据配置写入 `data/output/pipeline.log` 文件。
    *   中间文件（如转换后的 Markdown、提取的 JSON）会保存在 `data/output/` 目录下。
    *   最终生成的设备位号到产品代码的映射会打印到控制台。

4.  **可选：将结果保存到文件**:
    使用 `--output-json` 参数指定输出文件路径：
    ```bash
    python -m src.pipeline.main_pipeline data/input/your_document.pdf --output-json data/output/final_codes.json
    ```

## 6. 代码规范

本项目遵循 PEP 8 编码规范，并强制使用中文注释和类型提示。日志记录使用 Python 内置的 `logging` 模块。

## 7. 注意事项

*   确保 `.env` 文件配置正确，特别是 `DEEPSEEK_API_KEY`。
*   知识库源文件 (`semantic_source.xlsx`) 和标准库文件 (`libs/standard/`) 的内容直接影响标准化和代码生成的准确性。
*   首次运行 `build_kb.py` 可能需要一些时间，因为它需要下载嵌入模型并处理数据。
*   如果遇到导入错误，请确保您是从项目根目录 (`new_sensor_project/`) 运行脚本，或者已正确设置 PYTHONPATH。
