import sqlite3
from pathlib import Path
import json


def get_database_path():
    """
    SQLite database dosyasının yolunu döndürür.
    """

    project_root = Path(__file__).resolve().parents[2]

    database_dir = (
        project_root
        / "historag"
        / "data"
    )

    database_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return database_dir / "historag.db"


def get_connection():
    """
    SQLite bağlantısı oluşturur.
    """

    database_path = get_database_path()

    connection = sqlite3.connect(
        str(database_path)
    )

    return connection


def initialize_database():
    """
    RAG sistemi için gerekli SQLite tablosunu oluşturur.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            content TEXT NOT NULL,

            source TEXT,

            page INTEGER,

            section TEXT,

            embedding TEXT NOT NULL

        )
        """
    )

    connection.commit()

    connection.close()

    print(
        f"SQLite database hazır: {get_database_path()}"
    )


def clear_documents():
    """
    Daha önce kaydedilmiş chunk'ları siler.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM documents"
    )

    connection.commit()

    connection.close()

    print("Eski dokümanlar temizlendi.")


def insert_document(
    content,
    source,
    page,
    section,
    embedding
):
    """
    Bir document chunk ve embedding'i SQLite'a kaydeder.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO documents (
            content,
            source,
            page,
            section,
            embedding
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            content,
            source,
            page,
            section,
            json.dumps(embedding)
        )
    )

    connection.commit()

    connection.close()


def get_all_documents():
    """
    SQLite içerisindeki bütün chunk'ları getirir.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            content,
            source,
            page,
            section,
            embedding
        FROM documents
        """
    )

    rows = cursor.fetchall()

    connection.close()

    documents = []

    for row in rows:

        documents.append(
            {
                "id": row[0],
                "document": row[1],
                "metadata": {
                    "source": row[2],
                    "page": row[3],
                    "section": row[4]
                },
                "embedding": json.loads(row[5])
            }
        )

    return documents


if __name__ == "__main__":

    print("=" * 60)
    print("SQLite DATABASE TEST")
    print("=" * 60)

    initialize_database()

    documents = get_all_documents()

    print(
        f"Database içerisindeki chunk sayısı: {len(documents)}"
    )