import os
import json
from pathlib import Path

import numpy as np
import faiss
import gradio as gr
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# -----------------------------
# Paths (edit if your layout differs)
# -----------------------------
FAISS_PATH = Path(os.getenv("FAISS_PATH", "faiss.index"))
CHUNKS_PATH = Path(os.getenv("CHUNKS_PATH", "chunks.jsonl"))

# Embedding model must match what you used when building the FAISS index
EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# Phi-3 model (pick the one you want)
MODEL_ID = os.getenv("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")

# Cache to /tmp for serverless environments (Cloud Run)
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# -----------------------------
# Load KB (chunks) + FAISS
# -----------------------------
def load_chunks_jsonl(path: Path):
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


if not FAISS_PATH.exists():
    raise FileNotFoundError(f"FAISS index not found at: {FAISS_PATH.resolve()}")
if not CHUNKS_PATH.exists():
    raise FileNotFoundError(f"chunks.jsonl not found at: {CHUNKS_PATH.resolve()}")

chunks = load_chunks_jsonl(CHUNKS_PATH)
index = faiss.read_index(str(FAISS_PATH))

if index.ntotal != len(chunks):
    raise ValueError(
        f"Mismatch: FAISS index has {index.ntotal} vectors but chunks.jsonl has {len(chunks)} lines. "
        "They must be built from the same ordered chunks."
    )


# -----------------------------
# Embedder (CPU by default to save VRAM for Phi-3)
# -----------------------------
embed_device = os.getenv("EMBED_DEVICE", "cpu")
embedder = SentenceTransformer(EMBED_MODEL, device=embed_device)


# -----------------------------
# Phi-3 4-bit (bitsandbytes)
# -----------------------------
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not available. Phi-3 4-bit with bitsandbytes needs an NVIDIA GPU runtime. "
        "Deploy on GCP with GPU (e.g., Cloud Run L4) or use a CPU-friendly model."
    )

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.eval()


# -----------------------------
# Retrieval + RAG helpers
# -----------------------------
def retrieve(query: str, k: int = 5):
    q_vec = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, idxs = index.search(q_vec, k)
    out = []
    for score, idx in zip(scores[0], idxs[0]):
        c = chunks[int(idx)]
        out.append(
            {
                "score": float(score),
                "idx": int(idx),
                "chunk_id": c.get("chunk_id", ""),
                "lang": c.get("lang", ""),
                "intent": c.get("intent", ""),
                "category": c.get("category", ""),
                "user_query": c.get("user_query", ""),
                "bot_response": c.get("bot_response", ""),
                "text": c.get("text", ""),
            }
        )
    return out


def build_context(retrieved, max_sources=4, max_chars_per_source=700):
    blocks = []
    for i, r in enumerate(retrieved[:max_sources], 1):
        txt = r["text"] or (f"User: {r['user_query']}\nAgent: {r['bot_response']}")
        txt = txt.strip().replace("\u0000", "")
        if len(txt) > max_chars_per_source:
            txt = txt[:max_chars_per_source] + "…"

        meta = f"(score={r['score']:.3f}, lang={r['lang']}, category={r['category']}, chunk_id={r['chunk_id']})"
        blocks.append(f"[Source {i}] {meta}\n{txt}")

    return "\n\n".join(blocks)


SYSTEM_PROMPT = """You are a customer-support assistant.
Use ONLY the provided sources to answer the user.
If the sources do not contain the answer, say you don't have enough information and ask a brief follow-up question.
Be concise and actionable.
"""


def generate_phi3_answer(user_query: str, retrieved, max_new_tokens=220, temperature=0.4, top_p=0.9):
    context = build_context(retrieved)

    user_prompt = f"""SOURCES:
{context}

USER QUESTION:
{user_query}

INSTRUCTIONS:
- Answer using only the SOURCES.
- If not answerable, say you don't know based on sources.
- If helpful, provide steps/bullets.
"""

    # Prefer model chat template (Phi-3 supports it)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback prompt format
        prompt = SYSTEM_PROMPT + "\n\n" + user_prompt + "\n\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=(temperature > 0),
            temperature=float(temperature),
            top_p=float(top_p),
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return answer


def format_sources_md(retrieved, max_sources=5):
    lines = ["\n\n---\n**Sources used**"]
    for i, r in enumerate(retrieved[:max_sources], 1):
        uq = (r["user_query"] or "").strip()
        br = (r["bot_response"] or "").strip()
        uq = (uq[:120] + "…") if len(uq) > 120 else uq
        br = (br[:140] + "…") if len(br) > 140 else br
        lines.append(
            f"{i}. score={r['score']:.3f} | lang={r['lang']} | category={r['category']} | id={r['chunk_id']}\n"
            f"   - user: {uq}\n"
            f"   - bot : {br}"
        )
    return "\n".join(lines)


# -----------------------------
# Gradio UI
# -----------------------------
def chat_turn(message, history, top_k, max_sources, max_new_tokens, temperature, top_p):
    retrieved = retrieve(message, k=int(top_k))
    answer = generate_phi3_answer(
        message,
        retrieved,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    answer_with_sources = answer + format_sources_md(retrieved, max_sources=int(max_sources))
    history = history + [(message, answer_with_sources)]
    return history, ""


with gr.Blocks(title="Phi-3 4-bit RAG (FAISS + chunks.jsonl)") as demo:
    gr.Markdown("# Phi-3 (4-bit) RAG Chat\nUses FAISS retrieval over `chunks.jsonl` and answers using Phi-3 with citations.")

    chatbot = gr.Chatbot(height=520)
    msg = gr.Textbox(label="Ask", placeholder="Type your question…")

    with gr.Row():
        top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K retrieval")
        max_sources = gr.Slider(1, 6, value=4, step=1, label="Sources to include")
    with gr.Row():
        max_new_tokens = gr.Slider(32, 512, value=220, step=8, label="Max new tokens")
        temperature = gr.Slider(0.0, 1.2, value=0.4, step=0.05, label="Temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")

    send = gr.Button("Send")
    clear = gr.Button("Clear")

    state = gr.State([])

    send.click(
        fn=chat_turn,
        inputs=[msg, chatbot, top_k, max_sources, max_new_tokens, temperature, top_p],
        outputs=[chatbot, msg],
    )
    msg.submit(
        fn=chat_turn,
        inputs=[msg, chatbot, top_k, max_sources, max_new_tokens, temperature, top_p],
        outputs=[chatbot, msg],
    )
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    demo.launch(server_name="0.0.0.0", server_port=port)
