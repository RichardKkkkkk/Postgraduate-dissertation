from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


DOCX = Path(r"D:\code\Postgraduate-dissertation\thesis\Yikai_Zhao_MSc_Dissertation.docx")
OUT = DOCX.parent / "full_thesis_audit" / "display_equations"
OUT.mkdir(parents=True, exist_ok=True)
doc = Document(DOCX)

for index, paragraph in enumerate(doc.paragraphs):
    math = paragraph._p.xpath(".//*[local-name()='oMathPara']")
    if not math:
        continue
    texts = [node.text or "" for node in paragraph._p.iter(qn("w:t"))]
    math_texts = [node.text or "" for node in paragraph._p.xpath(".//*[local-name()='t']")]
    ppr = paragraph._p.pPr.xml if paragraph._p.pPr is not None else ""
    print(
        f"P{index} style={paragraph.style.name} visible={paragraph.text!r} "
        f"w_text={''.join(texts)!r} math_text={''.join(math_texts)!r}"
    )
    print("  pPr:", ppr)
    (OUT / f"p{index}.xml").write_text(paragraph._p.xml, encoding="utf-8")
