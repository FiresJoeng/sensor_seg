from docling.document_converter import DocumentConverter

def convert_pdf_to_markdown(input_path, output_path):
    # 初始化转换器并执行转换
    converter = DocumentConverter()
    result = converter.convert(input_path)

    # 导出为Markdown并保存
    markdown_content = result.document.export_to_markdown()
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"转换完成，结果已保存至: {output_path}")


if __name__ == "__main__":
    # 指定PDF文件路径
    pdf_path = "./input/example.pdf"
    md_path = "./output/example.md"

    convert_pdf_to_markdown(pdf_path, md_path)
