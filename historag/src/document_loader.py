from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path: str) -> str:
    """Extract text from a PDF while preserving page numbers."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    reader = PdfReader(str(path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            pages.append(
                f"[Page {page_number}]\n{text.strip()}"
            )

    return "\n\n".join(pages)


if __name__ == "__main__":

    # Project root:
    # C:\Projects\rag-assistant
    project_root = Path(__file__).resolve().parents[2]

    documents_dir = project_root / "historag" / "documents"

    print(f"Documents directory: {documents_dir}")
    print()

    pdf_files = list(documents_dir.glob("*.pdf"))

    print(f"PDF files found: {len(pdf_files)}")

    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    if not pdf_files:
        print("\nNo PDF files found.")
        exit()

    file_path = pdf_files[0]

    print(f"\nLoading: {file_path.name}")

    reader = PdfReader(str(file_path))

    print(f"Pages: {len(reader.pages)}")

    text = load_pdf(str(file_path))

    print("\n=== DOCUMENT PREVIEW ===\n")
    print(text[:5000])

    print("\n=== DOCUMENT INFO ===")
    print(f"Characters extracted: {len(text)}")