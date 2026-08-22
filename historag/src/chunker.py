import re
from typing import List, Dict


# ============================================================
# PAGE MARKERS
# ============================================================

PAGE_MARKER_PATTERN = re.compile(
    r"^\[Page\s+(\d+)\]$",
    re.IGNORECASE
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    PDF extraction sırasında oluşan gereksiz boşlukları temizler.
    """

    text = text.replace("\xa0", " ")

    # Birden fazla space -> tek space
    text = re.sub(r"[ \t]+", " ", text)

    # 3+ newline -> 2 newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# PAGE SPLITTING
# ============================================================

def split_into_pages(text: str):
    """
    [Page N] markerlarını kullanarak metni sayfalara ayırır.
    """

    lines = text.splitlines()

    pages = []

    current_page = 1
    current_lines = []

    for line in lines:

        stripped = line.strip()

        marker = PAGE_MARKER_PATTERN.match(stripped)

        if marker:

            if current_lines:

                page_text = clean_text("\n".join(current_lines))

                if page_text:
                    pages.append({
                        "page": current_page,
                        "text": page_text
                    })

            current_page = int(marker.group(1))
            current_lines = []

            continue

        current_lines.append(line)

    if current_lines:

        page_text = clean_text("\n".join(current_lines))

        if page_text:
            pages.append({
                "page": current_page,
                "text": page_text
            })

    return pages


# ============================================================
# SENTENCE SPLITTING
# ============================================================

# FIX: the old pattern r"(?<=[.!?])\s+" also splits after ordinal /
# abbreviation numbers like "7." in "7. Ordu Komutanı" (7th Army
# Commander), tearing one sentence into two ("...Halep'e 7." /
# "Ordu Komutanı oldu.") and further diluting an already-crowded
# chunk. The extra lookbehind refuses to split when the punctuation
# is immediately preceded by a digit.
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])(?<!\d\.)\s+")


def split_sentences(text: str) -> List[str]:
    """
    Türkçe düz metni cümlelere ayırır.
    """

    text = clean_text(text)

    if not text:
        return []

    text = re.sub(r"\s*\n\s*", " ", text)

    sentences = SENTENCE_SPLIT_PATTERN.split(text)

    return [s.strip() for s in sentences if s.strip()]


# ============================================================
# GENERIC TEXT CHUNKING
# ============================================================

def split_generic_section(
    text: str,
    page_number: int,
    chunk_size: int = 280,
    overlap: int = 40,
    max_sentences: int = 2
) -> List[Dict]:
    """
    Düz metinleri semantik olarak daha anlamlı, TOPICALLY FOCUSED
    cümle gruplarına ayırır.

    FIX: the previous defaults (chunk_size=700) let 6-8 unrelated
    historical events get packed into a single chunk (e.g. "became
    major general" + "fought the Russians" + "went to Damascus" +
    "traveled to Germany" + "got sick" + "took command of the 7th
    Army" all in one chunk). A chunk's embedding is roughly the
    average of everything in it, so cramming in many unrelated facts
    dilutes it and makes it rank poorly even for a query that matches
    one specific fact inside it perfectly — which is exactly the bug
    reported (the right answer existed in the database but ranked
    11th out of 20). Capping BOTH the character budget AND the
    number of sentences per chunk (whichever is hit first) keeps each
    chunk anchored to one or two closely related facts, regardless of
    how long or short individual sentences happen to be.
    """

    text = clean_text(text)

    if not text:
        return []

    sentences = split_sentences(text)

    if not sentences:
        return []

    chunks = []

    current_sentences = []
    current_length = 0

    def emit():
        chunk_body = " ".join(current_sentences).strip()

        chunks.append({
            "text": chunk_body,
            "page": page_number,
            "section": "HISTORY",
            "ticker": None
        })

    for sentence in sentences:

        sentence_length = len(sentence)

        would_exceed_chars = (
            current_sentences
            and current_length + sentence_length + 1 > chunk_size
        )

        would_exceed_count = (
            current_sentences
            and len(current_sentences) >= max_sentences
        )

        if would_exceed_chars or would_exceed_count:

            emit()

            # ------------------------------------------------
            # OVERLAP
            # Son cümleleri yeni chunk'a taşı (küçük overlap,
            # bağlamın tamamen kopmaması için).
            # ------------------------------------------------

            overlap_sentences = []
            overlap_length = 0

            for previous_sentence in reversed(current_sentences):

                if overlap_length + len(previous_sentence) + 1 > overlap:
                    break

                overlap_sentences.insert(0, previous_sentence)
                overlap_length += len(previous_sentence) + 1

            current_sentences = overlap_sentences
            current_length = overlap_length

        current_sentences.append(sentence)
        current_length += sentence_length + 1

    if current_sentences:
        emit()

    return chunks


# ============================================================
# MAIN CHUNK FUNCTION
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = 280,
    overlap: int = 40,
    max_sentences: int = 2
) -> List[Dict]:
    """
    Tarih / makale / düz metin PDF'leri için ana chunking fonksiyonu.

    Özellikler:
    - Sayfa bilgilerini korur.
    - Her chunk en fazla `max_sentences` cümle içerir (topically
      focused olması için).
    - ~280 karakterlik chunklar oluşturur (700 yerine — bkz.
      split_generic_section docstring'i).
    - Küçük bir overlap kullanır.
    - Her chunk'a HISTORY section ekler.
    """

    if not text or not text.strip():
        return []

    pages = split_into_pages(text)

    all_chunks = []

    for page_data in pages:

        page_chunks = split_generic_section(
            page_data["text"],
            page_number=page_data["page"],
            chunk_size=chunk_size,
            overlap=overlap,
            max_sentences=max_sentences
        )

        all_chunks.extend(page_chunks)

    return all_chunks


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    from pathlib import Path
    from document_loader import load_pdf

    project_root = Path(__file__).resolve().parents[2]

    pdf_path = project_root / "historag" / "documents" / "ata.pdf"

    print(f"Loading: {pdf_path}")

    text = load_pdf(str(pdf_path))

    print(f"Characters extracted: {len(text)}")

    chunks = chunk_text(text)

    print()
    print(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):

        print()
        print("=" * 70)
        print(f"CHUNK {i}")
        print(f"PAGE: {chunk['page']}")
        print(f"SECTION: {chunk['section']}")
        print(f"CHARS: {len(chunk['text'])}")
        print("=" * 70)
        print(chunk["text"])