# Yerel Tarih RAG Asistanı

Tamamen çevrimdışı çalışan, Microsoft Foundry Local üzerinde yerel embedding ve LLM
kullanan bir RAG (Retrieval-Augmented Generation) sistemi. PDF dokümanlarından
(örnek: Atatürk'ün hayatı) alınan bilgilerle, dokümanda olmayan hiçbir bilgiyi
uydurmadan soru cevaplar.

## Özellikler

- **Sıfır internet bağımlılığı** — tüm model çıkarımları yerel cihazda çalışır,
  hiçbir bulut servisi veya harici API kullanılmaz.
- **Microsoft Foundry Local** ile model indirme, yönetme ve çalıştırma.
- **Embedding**: `qwen3-embedding-0.6b`
- **LLM**: `Phi-3.5 Mini` (CPU varyantı, düşük kaynak tüketimi için)
- **SQLite** tabanlı, sunucusuz vektör depolama.
- **Kosinüs benzerliği** ile saf semantik arama (hibrit arama yok).
- **Halüsinasyon yasağı**: model yalnızca getirilen bağlamdaki bilgiyi kullanır,
  bilgi yoksa "Bu bilgi sağlanan dokümanlarda bulunmuyor" der.
- **Kaynak atfı**: her cevap, hangi sayfadan geldiğini belirtir.
- **Streamlit tabanlı web arayüzü**.

## Mimari

```
document_loader.py   PDF -> düz metin (sayfa bilgisiyle)
        |
        v
   chunker.py         Metni küçük, konu-odaklı parçalara böler
        |
        v
  embedding.py         Parçaları vektöre çevirir (Foundry Local)
        |
        v
   indexer.py           Parça + vektörleri SQLite'a kaydeder
        |
        v
  database.py            SQLite okuma/yazma katmanı
        |
        v
  retriever.py       Soru vektörü <-> kayıtlı vektörler (kosinüs benzerliği)
        |
        v
  generator.py        En alakalı parçaları bağlam olarak LLM'e verir
        |
        v
     rag.py              Her şeyi birleştiren orkestratör / CLI giriş noktası
        |
        v
     app.py              Streamlit web arayüzü
```

## Kurulum

### Gereksinimler

- Python 3.10+
- [Microsoft Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/) kurulu olmalı:

  ```powershell
  winget install --id Microsoft.FoundryLocal
  ```

### Proje kurulumu

```powershell
git clone <bu-repo-url>
cd rag-assistant

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

PDF dokümanını `finrag/documents/` klasörüne koy (örnek: `ata.pdf`).

## Kullanım

### 1. Dokümanı indeksle

```powershell
python finrag/src/indexer.py
```

PDF'i okuyup parçalara ayırır, embedding'lerini çıkarır ve `finrag/data/finrag.db`
SQLite veritabanına kaydeder.

### 2. Komut satırından soru sor

```powershell
python finrag/src/rag.py
```

### 3. Web arayüzünden soru sor

```powershell
streamlit run finrag/src/app.py
```

Tarayıcıda otomatik olarak açılan sayfadan (genelde `http://localhost:8501`)
sohbet tarzında soru sorabilirsin. Her cevabın altındaki "Kaynaklar" panelinden
hangi sayfadan alındığını görebilirsin.

## Proje yapısı

```
rag-assistant/
├── venv/                      (git'e dahil değil)
├── requirements.txt
├── .gitignore
├── README.md
└── finrag/
    ├── documents/
    │   └── ata.pdf            (kaynak PDF)
    ├── data/
    │   └── finrag.db          (indexer.py tarafından üretilir, git'e dahil değil)
    └── src/
        ├── document_loader.py
        ├── chunker.py
        ├── embedding.py
        ├── database.py
        ├── indexer.py
        ├── retriever.py
        ├── generator.py
        ├── rag.py
        └── app.py
```

## Notlar / Sınırlamalar

- İlk çalıştırmada Foundry Local, modelleri indirir (birkaç GB); bu adım
  internet gerektirir, sonrasında tamamen çevrimdışı çalışır.
- `Phi-3.5 Mini` küçük bir model olduğu için karmaşık çok adımlı çıkarımlarda
  sınırlı kalabilir; sistem bunu bağlam dışına çıkmadan, doğrudan dokümandaki
  bilgiyle cevap vererek dengeler.
- Yeni bir PDF eklediğinde `indexer.py`'yi tekrar çalıştırman gerekir.