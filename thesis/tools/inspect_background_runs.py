from docx import Document
from docx.oxml.ns import qn


path = r"D:\code\Postgraduate-dissertation\thesis\Yikai_Zhao_MSc_Dissertation.docx"
document = Document(path)
for index in [40, 44, 46, 48, 50, 52, 54, 55, 57, 59, 60, 61, 63, 64, 65, 67, 68, 69]:
    paragraph = document.paragraphs[index]
    print(f"\n=== P{index}: {paragraph.text}")
    for child_index, child in enumerate(paragraph._p):
        tag = child.tag.split("}")[-1]
        texts = [node.text or "" for node in child.iter(qn("w:t"))]
        instructions = [node.text or "" for node in child.iter(qn("w:instrText"))]
        field_chars = [node.get(qn("w:fldCharType")) for node in child.iter(qn("w:fldChar"))]
        print(child_index, tag, "TEXT=", repr("".join(texts)), "FIELD=", field_chars, "INSTR=", instructions)
