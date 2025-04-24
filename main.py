# 导入依赖
import os


# 文档处理
class InfoExtractor:
    # Markdown 转换器
    class MDConv:
        # 文件转md
        def pdf_to_md(pdf_path):
            pass

    # JSON 处理器

    class JSONProc:
        # 使用LLMs将md转json
        def md_to_json(md_path):
            pass

        # 提供人工检查json可选项
        def json_check():
            pass


# 语义搜索/向量知识库库/知识图谱知识库，在这三种中选择一种，测试准度，择优
class ParameterOutput:
    class KnowledgeBase:
        # 语义搜索
        def SemanticSearch():
            pass

        # 向量知识库
        def VectorKB():
            pass

        # 知识图谱
        def KG():
            pass

    class ResultCompare:
        """
        比较上面的结果，择优
        知识库之外的参数输出出来，比如说一个参数没有出现在知识库中，那么就输出出来
        """
        def func():
            pass

    def Output_Info():
        pass


# 标准匹配
class StandardMatch:
    # 从标准参数库提取参数（注意技术实现路径：规则匹配/大模型检索）
    def load_standard_params():
        pass

    # 型号生成 + 人工审核
    def generate_model():
        pass


# 底层入口
if __name__ == "__main__":
    pass
