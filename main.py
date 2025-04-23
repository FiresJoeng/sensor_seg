# 导入依赖
import os


# 文档处理
class InfoExtractor:
    # Markdown 转换器
    class MDConv:
        # pdf转md
        def pdf_to_md(pdf_path):
            pass

        # xlsx转md
        def xlsx_to_md(pdf_path):
            pass

        # docx转md
        def docx_to_md(docx_path):
            pass

        # 图片转md
        def img_to_md(img_path):
            pass

    # JSON 处理器
    class JSONProc:
        # 使用LLMs将md转json
        def md_to_json(md_path):
            pass

        # 检查json
        def json_check():
            pass


# 语义搜索
class SamanticSearch:
    def __init__(self, text):
        self.text = text

    # 语义搜索逻辑
    def search(self, query):
        pass

    def __call__(self, query):
        return self.search(query)


# 标准匹配
class StandardMatch:
    def func():
        pass


# 型号输出
class ModelOutput:
    def func():
        pass


# 底层入口
if __name__ == "__main__":
    pass
