from pathlib import Path
import sys

from docx import Document


TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
DOCX = ROOT / "thesis" / "Yikai_Zhao_MSc_Dissertation.docx"
sys.path.insert(0, str(TOOLS))

import standardize_math_notation as audit  # noqa: E402


document = Document(str(DOCX))
audit.validate(document)
print("MATH_AUDIT: PASS")

for number in range(1, 15):
    paragraph = audit.paragraph_with_bookmark(document, f"eq{number}")
    fractions = len(paragraph._p.xpath(".//m:f"))
    print(
        f"eq{number}: fractions={fractions}; "
        f"text={audit.math_text(paragraph)}"
    )

shift = audit.find_paragraph(document, "The shifted variants retain")
print(
    "inline_shift: "
    f"fractions={len(shift._p.xpath('.//m:f'))}; "
    f"refs={len(shift._p.xpath('.//w:instrText'))}; "
    f"text={audit.math_text(shift)}"
)

confidence_interval = audit.find_paragraph(document, "For each five-seed model")
print(
    "inline_CI: "
    f"fractions={len(confidence_interval._p.xpath('.//m:f'))}; "
    f"text={audit.math_text(confidence_interval)}"
)
