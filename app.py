"""
Chat with your PDF — a small, fully self-contained Retrieval-Augmented
Generation (RAG) demo.

Pipeline:  PDF -> text -> chunks -> MiniLM embeddings -> cosine retrieval
           -> Flan-T5 grounded answer (with the retrieved passages shown).

Everything runs locally on CPU (no external API, no keys), so the demo is free
and always available — the focus is the RAG architecture, not a frontier LLM.
"""

import gradio as gr
import numpy as np
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-base")
_GEN_MODEL = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").eval()


def generate(prompt: str) -> str:
    inputs = _TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out_ids = _GEN_MODEL.generate(**inputs, max_new_tokens=200)
    return _TOKENIZER.decode(out_ids[0], skip_special_tokens=True)

# in-memory index for the currently loaded document
STATE = {"chunks": [], "embeddings": None, "name": None}


def chunk_text(text: str, size: int = 480, overlap: int = 80):
    text = " ".join(text.split())
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks


def process_pdf(file):
    if file is None:
        return "⚠️ Please upload a PDF first."
    reader = PdfReader(file.name)
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    if not text.strip():
        return "⚠️ No extractable text found — this looks like a scanned/image PDF."
    chunks = chunk_text(text)
    STATE["chunks"] = chunks
    STATE["embeddings"] = EMBEDDER.encode(chunks, normalize_embeddings=True)
    STATE["name"] = file.name.split("/")[-1].split("\\")[-1]
    return f"✅ Indexed **{len(chunks)}** chunks from **{len(reader.pages)}** pages. Ask a question below."


def answer(question: str, k: int = 3):
    if STATE["embeddings"] is None:
        return "Upload and process a PDF first.", ""
    if not question.strip():
        return "Type a question.", ""

    q = EMBEDDER.encode([question], normalize_embeddings=True)[0]
    sims = STATE["embeddings"] @ q
    top = sims.argsort()[::-1][:k]
    context = "\n\n".join(STATE["chunks"][i] for i in top)

    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    out = generate(prompt)

    sources = "\n\n".join(
        f"**[{rank + 1}]** (score {sims[i]:.2f}) {STATE['chunks'][i][:280]}…"
        for rank, i in enumerate(top)
    )
    return out, sources


with gr.Blocks(title="Chat with your PDF (RAG)") as demo:
    gr.Markdown(
        "# 📄 Chat with your PDF — RAG demo\n"
        "Upload a PDF, then ask questions. Retrieval (MiniLM embeddings + cosine "
        "search) grounds a Flan-T5 answer in the most relevant passages, which are "
        "shown so you can verify the source.\n\n"
        "_Runs fully on CPU — no API keys. Small model, so keep questions specific._"
    )
    with gr.Row():
        pdf = gr.File(label="PDF", file_types=[".pdf"])
        status = gr.Markdown()
    process_btn = gr.Button("Process PDF", variant="primary")
    question = gr.Textbox(label="Your question", placeholder="What is this document about?")
    ask_btn = gr.Button("Ask")
    response = gr.Markdown(label="Answer")
    with gr.Accordion("Retrieved passages (sources)", open=False):
        sources = gr.Markdown()

    process_btn.click(process_pdf, inputs=pdf, outputs=status)
    ask_btn.click(answer, inputs=question, outputs=[response, sources])
    question.submit(answer, inputs=question, outputs=[response, sources])

if __name__ == "__main__":
    demo.launch()
