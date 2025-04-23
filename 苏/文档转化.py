

from docling.document_converter import DocumentConverter

# 指定PDF文件路径
pdf_path = r"C:\Users\41041\Desktop\项目文件\系统仓库\sensor_seg\苏\提取后的json数据及原文件\1001-01一体化温度变送器3台.pdf"

# 初始化转换器并执行转换
converter = DocumentConverter()
result = converter.convert(pdf_path)

# 导出为Markdown并保存
markdown_content = result.document.export_to_markdown()
output_path1 = r"C:\\Users\\41041\\Desktop\\项目文件\\系统仓库\\sensor_seg\\苏\\提取后的json数据及原文件\\1001-01一体化温度变送器3台.md"
with open(output_path1, 'w',encoding='utf-8') as f:
    f.write(markdown_content)

print(f"转换完成，结果已保存至: {output_path1}")
