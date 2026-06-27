# 📄 Chat with your PDF — RAG demo

A small, fully self-contained **Retrieval-Augmented Generation** pipeline.

- **🔴 Live demo:** https://huggingface.co/spaces/barisgovercin/chat-with-pdf

```
PDF → text → chunks → MiniLM embeddings → cosine retrieval → Flan-T5 grounded answer
```

Upload a PDF and ask questions. The retrieved passages are shown alongside each
answer so you can verify what the response is grounded in.

## How it works

| Stage | Component |
|-------|-----------|
| Parse | `pypdf` extracts text from the PDF |
| Chunk | Sliding window (~480 chars, 80 overlap) |
| Embed | `sentence-transformers/all-MiniLM-L6-v2` |
| Retrieve | Cosine similarity, top-k passages |
| Generate | `google/flan-t5-base`, answer grounded in retrieved context |

Runs entirely on CPU with open models — no API keys, always free. The point is
the **RAG architecture**, which mirrors the retrieval setup in my MSc
dissertation (a multimodal Vision-Language Model + RAG for skin-cancer
diagnosis).

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Stack

Gradio · sentence-transformers · Transformers (Flan-T5) · pypdf · NumPy
