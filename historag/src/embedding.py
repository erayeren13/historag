import time

from foundry_local_sdk import Configuration, FoundryLocalManager
from foundry_local_sdk.exception import FoundryLocalException


MODEL_NAME = "qwen3-embedding-0.6b"

# FIX: sending all chunks in a single generate_embeddings() call
# (e.g. 92 texts at once) triggered "Operation was cancelled" even
# on the CPU variant — this isn't the WebGPU f16 issue seen
# elsewhere, it's the batch itself being too large for one request.
# Splitting into small batches with a short retry avoids it.
BATCH_SIZE = 8
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 3


def load_embedding_model():

    print(f"Embedding modeli yükleniyor: {MODEL_NAME}")

    config = Configuration(app_name="historag")

    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model(MODEL_NAME)

    print("Embedding modeli indiriliyor/yükleniyor...")

    model.download(
        lambda progress: print(
            f"\rDownloading: {progress:.2f}%",
            end="",
            flush=True
        )
    )

    print()

    model.load()

    print("Embedding modeli hazır.")

    return model.get_embedding_client()


def _embed_batch(client, texts):

    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):

        try:
            response = client.generate_embeddings(texts)
            return [item.embedding for item in response.data]

        except FoundryLocalException as e:

            last_error = e

            print(
                f"\n  Batch başarısız (deneme {attempt}/{MAX_ATTEMPTS}): {e}"
            )

            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)

    raise last_error


def create_embeddings(client, texts):

    print(f"Creating embeddings... ({len(texts)} chunks, batch={BATCH_SIZE})")

    all_embeddings = []

    for start in range(0, len(texts), BATCH_SIZE):

        batch = texts[start:start + BATCH_SIZE]

        batch_num = start // BATCH_SIZE + 1
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunk)...")

        embeddings = _embed_batch(client, batch)

        all_embeddings.extend(embeddings)

    return all_embeddings


def create_query_embedding(client, query):

    response = client.generate_embedding(query)

    return response.data[0].embedding


if __name__ == "__main__":

    client = load_embedding_model()

    test_texts = [
        "Mustafa Kemal Atatürk 1 Nisan 1916'da tümgeneralliğe yükseldi.",
        "Atatürk Çanakkale Savaşı'ndan sonra Edirne ve Diyarbakır'da görev yaptı."
    ]

    embeddings = create_embeddings(client, test_texts)

    print()
    print("Embedding oluşturuldu.")
    print(f"Embedding sayısı: {len(embeddings)}")
    print(f"Vector boyutu: {len(embeddings[0])}")

    query_embedding = create_query_embedding(
        client,
        "Atatürk ne zaman tümgeneral oldu?"
    )

    print(f"Query vector boyutu: {len(query_embedding)}")