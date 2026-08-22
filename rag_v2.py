
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -----------------------------
# Belgeyi oku
# -----------------------------
text = Path("docs/rag_nedir.txt").read_text(encoding="utf-8")

chunks = [p.strip() for p in text.split("\n\n") if p.strip()]

print(f"Toplam chunk: {len(chunks)}")

# -----------------------------
# Embedding modelini yükle
# -----------------------------
print("Embedding modeli yükleniyor...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Chunk embedding'leri
chunk_embeddings = embedder.encode(chunks)

# -----------------------------
# Kullanıcı sorusu
# -----------------------------
question = "RAG sisteminin temel adımları nelerdir?"

question_embedding = embedder.encode([question])

# Benzerlik hesapla
scores = cosine_similarity(question_embedding, chunk_embeddings)[0]

# En iyi 2 chunk
top_indices = np.argsort(scores)[::-1][:2]

print("\nEn ilgili chunk'lar:\n")

selected_chunks = []

for idx in top_indices:
    print(f"Skor: {scores[idx]:.3f}")
    print(chunks[idx])
    print("-" * 50)
    selected_chunks.append(chunks[idx])

# -----------------------------
# RAG context oluştur
# -----------------------------
context = "\n\n".join(selected_chunks)

print("\n=== MODELE GÖNDERİLECEK CONTEXT ===\n")
print(context)

