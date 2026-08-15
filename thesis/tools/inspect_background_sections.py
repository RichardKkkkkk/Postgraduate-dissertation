from docx import Document


path = r"D:\code\Postgraduate-dissertation\thesis\Yikai_Zhao_MSc_Dissertation.docx"
document = Document(path)
inside = False
for index, paragraph in enumerate(document.paragraphs):
    text = paragraph.text.strip()
    if text.startswith("2.2 "):
        inside = True
    if inside:
        print(f"P{index} [{paragraph.style.name}] {text}")
    if text.startswith("2.4 "):
        break
