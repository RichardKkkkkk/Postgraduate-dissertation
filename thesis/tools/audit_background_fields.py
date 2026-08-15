from docx import Document


path = r"D:\code\Postgraduate-dissertation\thesis\Yikai_Zhao_MSc_Dissertation.docx"
document = Document(path)
for index in range(39, 71):
    paragraph = document.paragraphs[index]
    xml = paragraph._p.xml
    if paragraph.text.strip():
        print(
            index,
            "field=" + str("fldChar" in xml or "instrText" in xml),
            "runs=" + str(len(paragraph.runs)),
            paragraph.text[:90],
        )
