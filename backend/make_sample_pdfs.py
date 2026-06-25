"""
make_sample_pdfs.py — (Optional) turn the sample .md policies into real PDFs.

The RAG pipeline reads .md files just fine, but for a portfolio demo it looks
more realistic to ingest actual PDFs. This uses PyMuPDF (already a dependency)
to render each .md in data/hr_docs/ into a matching .pdf next to it.

Usage:
    cd backend
    python make_sample_pdfs.py
"""
import fitz  # PyMuPDF

import config


def md_to_pdf(md_path, pdf_path):
    text = md_path.read_text(encoding="utf-8")
    # Strip the most common markdown markers for cleaner print output.
    for marker in ("### ", "## ", "# ", "**", "_"):
        text = text.replace(marker, "")

    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(56, 56, page.rect.width - 56, page.rect.height - 56)

    # insert_textbox returns leftover text that didn't fit; add pages as needed.
    remaining = text
    while remaining:
        leftover = page.insert_textbox(rect, remaining, fontsize=11, fontname="helv")
        if leftover <= 0:
            break
        # Rough split: drop the portion that fit and continue on a new page.
        used = len(remaining) + int(leftover)
        remaining = remaining[max(0, used):]
        if not remaining:
            break
        page = doc.new_page()

    doc.save(str(pdf_path))
    doc.close()


def main():
    md_files = sorted(config.DATA_DIR.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {config.DATA_DIR}")
        return
    for md in md_files:
        pdf = md.with_suffix(".pdf")
        md_to_pdf(md, pdf)
        print(f"  {md.name}  ->  {pdf.name}")
    print(
        "\nDone. You can now delete the .md files if you only want PDFs, "
        "then re-run `python ingest.py`."
    )


if __name__ == "__main__":
    main()
