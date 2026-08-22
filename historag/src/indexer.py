from pathlib import Path

from document_loader import load_pdf

from chunker import chunk_text

from embedding import (
    load_embedding_model,
    create_embeddings
)

from database import (
    initialize_database,
    clear_documents,
    insert_document
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_DIR = (
    PROJECT_ROOT
    / "finrag"
    / "documents"
)


# ============================================================
# INDEX DOCUMENTS
# ============================================================

def index_documents():

    print("=" * 60)
    print("LOCAL RAG INDEXER")
    print("=" * 60)

    # ========================================================
    # 1. DATABASE
    # ========================================================

    print()
    print("[1/4] SQLite database hazırlanıyor...")

    initialize_database()

    clear_documents()

    # ========================================================
    # 2. PDF FILES
    # ========================================================

    pdf_files = list(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    print()
    print(
        f"Bulunan PDF sayısı: {len(pdf_files)}"
    )

    if not pdf_files:

        print(
            "PDF bulunamadı."
        )

        return

    # ========================================================
    # 3. EMBEDDING MODEL
    # ========================================================

    print()
    print(
        "[2/4] Embedding modeli hazırlanıyor..."
    )

    embedding_client = load_embedding_model()

    # ========================================================
    # 4. PROCESS DOCUMENTS
    # ========================================================

    total_chunks = 0

    for pdf_file in pdf_files:

        print()
        print("=" * 60)
        print(
            f"Processing: {pdf_file.name}"
        )
        print("=" * 60)

        # ----------------------------------------------------
        # PDF -> TEXT
        # ----------------------------------------------------

        print(
            "\nPDF metni çıkarılıyor..."
        )

        text = load_pdf(
            str(pdf_file)
        )

        print(
            f"Characters extracted: {len(text)}"
        )

        # ----------------------------------------------------
        # TEXT -> CHUNKS
        # ----------------------------------------------------

        print(
            "\nChunking yapılıyor..."
        )

        chunks = chunk_text(
            text,
            chunk_size=700,
            overlap=100
        )

        print(
            f"Chunks: {len(chunks)}"
        )

        if not chunks:

            print(
                "Chunk bulunamadı."
            )

            continue

        # ----------------------------------------------------
        # CHUNKS -> EMBEDDINGS
        # ----------------------------------------------------

        print(
            "\n[3/4] Embedding oluşturuluyor..."
        )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = create_embeddings(
            embedding_client,
            texts
        )

        print(
            f"Embedding sayısı: {len(embeddings)}"
        )

        print(
            f"Vector boyutu: {len(embeddings[0])}"
        )

        # ----------------------------------------------------
        # SAVE TO SQLITE
        # ----------------------------------------------------

        print(
            "\n[4/4] SQLite'a kaydediliyor..."
        )

        for chunk, embedding in zip(
            chunks,
            embeddings
        ):

            insert_document(
                content=chunk["text"],
                source=pdf_file.name,
                page=chunk["page"],
                section=chunk["section"],
                embedding=embedding
            )

        total_chunks += len(chunks)

        print(
            f"{len(chunks)} chunk SQLite'a kaydedildi."
        )

    # ========================================================
    # COMPLETED
    # ========================================================

    print()
    print("=" * 60)
    print("INDEXING COMPLETED")
    print("=" * 60)

    print(
        f"Toplam chunk: {total_chunks}"
    )

    database_path = (
        PROJECT_ROOT
        / "historag"
        / "data"
        / "historag.db"
    )

    print(
        f"Database: {database_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    index_documents()