#!/usr/bin/env python3
"""Build PDF report from explainability markdown with embedded images."""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

from fpdf import FPDF
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert explainability markdown report to PDF.")
    parser.add_argument(
        "--input-md",
        type=Path,
        default=Path("RelatorioExplicabilidadeKsat.md"),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("RelatorioExplicabilidadeKsat.pdf"),
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    # Keep compatibility with standard core PDF fonts.
    txt = unicodedata.normalize("NFKD", text)
    txt = txt.encode("latin-1", "ignore").decode("latin-1")
    return txt.replace("**", "").replace("`", "")


def add_heading(pdf: FPDF, level: int, text: str) -> None:
    size_map = {1: 17, 2: 14, 3: 12}
    size = size_map.get(level, 11)
    pdf.set_font("Helvetica", "B", size=size)
    pdf.ln(2)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 7, clean_text(text))
    pdf.ln(1)


def add_paragraph(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", size=11)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 6, clean_text(text))


def add_image(pdf: FPDF, image_path: Path, caption: str) -> None:
    if not image_path.exists():
        add_paragraph(pdf, f"[Imagem nao encontrada: {image_path}]")
        return

    max_w = pdf.w - pdf.l_margin - pdf.r_margin
    max_h = pdf.h - pdf.t_margin - pdf.b_margin - 12
    with Image.open(image_path) as im:
        w_px, h_px = im.size
    ratio = h_px / max(w_px, 1)
    draw_w = max_w
    draw_h = draw_w * ratio
    if draw_h > max_h:
        draw_h = max_h
        draw_w = draw_h / max(ratio, 1e-12)

    if pdf.get_y() + draw_h + 10 > pdf.h - pdf.b_margin:
        pdf.add_page()

    pdf.set_font("Helvetica", "I", size=10)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5, clean_text(caption))
    pdf.ln(1)
    pdf.set_x(pdf.l_margin)
    pdf.image(str(image_path), w=draw_w, h=draw_h)
    pdf.ln(3)
    pdf.set_x(pdf.l_margin)


def render_markdown_to_pdf(md_path: Path, pdf_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    image_re = re.compile(r"!\[(.*?)\]\((.*?)\)")

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    in_code = False
    for raw in lines:
        line = raw.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            if not in_code:
                pdf.ln(1)
            continue
        if in_code:
            pdf.set_font("Courier", size=9)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 4.5, clean_text(line))
            continue

        if not line.strip():
            pdf.ln(2)
            continue

        img_match = image_re.match(line.strip())
        if img_match:
            caption = img_match.group(1).strip()
            path_str = img_match.group(2).strip()
            img_path = (md_path.parent / path_str).resolve()
            add_image(pdf, img_path, caption)
            continue

        if line.startswith("### "):
            add_heading(pdf, 3, line[4:].strip())
            continue
        if line.startswith("## "):
            add_heading(pdf, 2, line[3:].strip())
            continue
        if line.startswith("# "):
            add_heading(pdf, 1, line[2:].strip())
            continue

        if line.startswith("- "):
            add_paragraph(pdf, "• " + line[2:].strip())
            continue

        m_num = re.match(r"^\d+\.\s+(.*)$", line)
        if m_num:
            add_paragraph(pdf, line)
            continue

        add_paragraph(pdf, line)

    pdf.output(str(pdf_path))


def main() -> None:
    args = parse_args()
    if not args.input_md.exists():
        raise FileNotFoundError(f"Input markdown not found: {args.input_md}")
    render_markdown_to_pdf(md_path=args.input_md, pdf_path=args.output_pdf)
    print(f"PDF generated: {args.output_pdf.resolve()}")


if __name__ == "__main__":
    main()
