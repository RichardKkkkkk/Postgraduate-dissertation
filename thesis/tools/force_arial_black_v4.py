from __future__ import annotations

import hashlib
import os
import tempfile
import zipfile
from pathlib import Path

from lxml import etree


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "thesis" / "Yikai_Zhao_MSc_Dissertation_Core_Draft_v3.docx"
OUTPUT = ROOT / "thesis" / "Yikai_Zhao_MSc_Dissertation_Core_Draft_Arial_Black_v4.docx"
EXPECTED_SOURCE_SHA256 = "75CA10693DBA9FE1DFD6552C667FE1D46E0452C0B935B994BE4BC995CCDD3D18"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS = {"w": W, "a": A}


def qn(namespace: str, tag: str) -> str:
    return f"{{{namespace}}}{tag}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def ensure_child(parent, namespace: str, tag: str, first: bool = False):
    child = parent.find(qn(namespace, tag))
    if child is None:
        child = etree.Element(qn(namespace, tag))
        if first:
            parent.insert(0, child)
        else:
            parent.append(child)
    return child


def set_font_and_colour(r_pr) -> None:
    r_fonts = ensure_child(r_pr, W, "rFonts", first=True)
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        r_fonts.set(qn(W, attr), "Arial")
    for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        r_fonts.attrib.pop(qn(W, attr), None)

    colour = ensure_child(r_pr, W, "color")
    colour.set(qn(W, "val"), "000000")
    for attr in ("themeColor", "themeTint", "themeShade"):
        colour.attrib.pop(qn(W, attr), None)


def patch_word_xml(root, part_name: str) -> None:
    # Every existing run-property block, including paragraph marks and styles.
    for r_pr in root.xpath(".//w:rPr", namespaces=NS):
        set_font_and_colour(r_pr)

    # Runs without direct formatting must also be explicit so all current text
    # stays Arial/black even when opened under a different Word theme.
    for run in root.xpath(".//w:r", namespaces=NS):
        r_pr = run.find(qn(W, "rPr"))
        if r_pr is None:
            r_pr = etree.Element(qn(W, "rPr"))
            run.insert(0, r_pr)
        set_font_and_colour(r_pr)

    if part_name == "word/styles.xml":
        doc_defaults = root.find(qn(W, "docDefaults"))
        if doc_defaults is None:
            doc_defaults = etree.Element(qn(W, "docDefaults"))
            root.insert(0, doc_defaults)
        r_pr_default = ensure_child(doc_defaults, W, "rPrDefault")
        default_r_pr = ensure_child(r_pr_default, W, "rPr")
        set_font_and_colour(default_r_pr)

        for style in root.xpath(".//w:style", namespaces=NS):
            r_pr = style.find(qn(W, "rPr"))
            if r_pr is None:
                r_pr = etree.Element(qn(W, "rPr"))
                style.append(r_pr)
            set_font_and_colour(r_pr)

    if part_name == "word/numbering.xml":
        for level in root.xpath(".//w:lvl", namespaces=NS):
            r_pr = level.find(qn(W, "rPr"))
            if r_pr is None:
                r_pr = etree.Element(qn(W, "rPr"))
                level.append(r_pr)
            set_font_and_colour(r_pr)

    # The results table formerly used white text on a dark-blue header. Forced
    # black text needs a light fill for legibility.
    for shading in root.xpath(".//w:shd", namespaces=NS):
        if shading.get(qn(W, "fill"), "").upper() == "2F5597":
            shading.set(qn(W, "fill"), "E7E6E6")


def patch_theme(root) -> None:
    for group_name in ("majorFont", "minorFont"):
        groups = root.xpath(f".//a:fontScheme/a:{group_name}", namespaces=NS)
        for group in groups:
            for child_name in ("latin", "ea", "cs"):
                child = group.find(qn(A, child_name))
                if child is not None:
                    child.set("typeface", "Arial")


def should_patch_xml(part_name: str, data: bytes) -> bool:
    if not part_name.endswith(".xml"):
        return False
    if part_name == "word/theme/theme1.xml":
        return True
    return part_name.startswith("word/") and b"schemas.openxmlformats.org/wordprocessingml" in data


def build() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Core draft v3 changed after the Arial/black formatting contract was recorded.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix="arial_black_v4_", suffix=".docx", dir=OUTPUT.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(SOURCE, "r") as source_zip, zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as output_zip:
            for info in source_zip.infolist():
                data = source_zip.read(info.filename)
                if should_patch_xml(info.filename, data):
                    parser = etree.XMLParser(remove_blank_text=False, resolve_entities=False)
                    root = etree.fromstring(data, parser)
                    if info.filename == "word/theme/theme1.xml":
                        patch_theme(root)
                    else:
                        patch_word_xml(root, info.filename)
                    data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
                output_zip.writestr(info, data)
        os.replace(temporary, OUTPUT)
    finally:
        if temporary.exists():
            temporary.unlink()

    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("The retained v3 source changed during generation.")
    print(OUTPUT)


if __name__ == "__main__":
    build()
