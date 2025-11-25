import gradio as gr
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# ---------------------------
# LOAD ENV
# ---------------------------
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

COLLECTION = "tuyensinh"

# Model LLM
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.5-flash")

# Embedding
embedder = SentenceTransformer("intfloat/multilingual-e5-base")

# Qdrant client
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# ---------------------------
# RAG SEARCH FUNCTION
# ---------------------------
def search_qdrant(query, top_k=5):
    query_vec = embedder.encode(query).tolist()

    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vec,
        limit=top_k
    )

    contexts = []
    for h in hits:
        text = h.payload.get("text", "")
        url = h.payload.get("url", "")
        contexts.append(f"- {text}\n(Source: {url})")

    return "\n\n".join(contexts)


def generate_answer(question):
    context = search_qdrant(question, top_k=5)

    prompt = f"""
Bạn là trợ lý tư vấn tuyển sinh của Đại học Đông Á.

Dữ liệu dưới đây là các đoạn thông tin được tìm thấy từ hệ thống:

{context}

Hãy trả lời câu hỏi của thí sinh dựa trên thông tin có thật trong dữ liệu trên.
Nếu dữ liệu không đủ, hãy nói: "Hiện tại tôi chưa có thông tin trong hệ thống."

Câu hỏi: {question}
"""

    response = llm.generate_content(prompt)
    return response.text


# ---------------------------
# GRADIO UI
# ---------------------------
def chat_fn(message, history):
    answer = generate_answer(message)
    history.append((message, answer))
    return history, history


with gr.Blocks(title="Chatbot Tư Vấn Tuyển Sinh - Đại Học Đông Á") as demo:
    gr.Markdown("# 🎓 Chatbot Tư Vấn Tuyển Sinh – Đại Học Đông Á\nHỏi tôi bất kỳ thông tin nào về tuyển sinh!")

    chatbot = gr.Chatbot(height=600)

    msg = gr.Textbox(label="Nhập câu hỏi của bạn...")

    msg.submit(chat_fn, [msg, chatbot], [chatbot, chatbot])
    
    gr.Markdown("**Ví dụ câu hỏi:**\n- Hồ sơ xét tuyển cần những gì?\n- Học phí năm 2025 bao nhiêu?\n- Đại học Đông Á có bao nhiêu phương thức xét tuyển?")

demo.launch()
