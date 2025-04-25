from docling.document_converter import DocumentConverter


def converter(xlsx_path):
    converter = DocumentConverter()
    md_content = converter.convert(xlsx_path).document.export_to_markdown()
    return md_content


if __name__ == "__main__":
    xlsx_path = "libs/standard.xlsx"
    md_content = converter(xlsx_path)
    print(md_content)
