import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 解决OpenMP冲突
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用oneDNN优化

from docling.document_converter import DocumentConverter

# 指定PDF文件路
pdf_path = r"D:\\项目甲方文件\\2021\\2021\\1\\P2021-180008温变选型\\P2021-180008温度变送器规格书.pdf"

# 初始化转换器并执行转换
converter = DocumentConverter()
result = converter.convert(pdf_path)

# 导出为Markdown并保存
markdown_content = result.document.export_to_markdown()
output_path1 = r"P2021-180008温度变送器规格书.md"
with open(output_path1, 'w',encoding='utf-8') as f:
    f.write(markdown_content)

print(f"转换完成，结果已保存至: {output_path1}")
