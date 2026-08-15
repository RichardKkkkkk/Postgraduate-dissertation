from pathlib import Path
import sys

from docx import Document


if len(sys.argv) != 4:
    raise SystemExit("usage: inspect_section.py DOCX START END")

document = Document(sys.argv[1])
start = int(sys.argv[2])
end = int(sys.argv[3])

for index, paragraph in enumerate(document.paragraphs):
    if start <= index <= end:
        print(f"P{index} [{paragraph.style.name}] {paragraph.text}")
