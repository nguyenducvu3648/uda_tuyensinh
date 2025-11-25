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
# GRADIO CHAT FUNCTION
# ---------------------------
def chat_fn(user_message, history):
    if history is None:
        history = []
    bot_message = generate_answer(user_message)
    # Thêm dictionary thay vì tuple
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": bot_message})
    return history


# ---------------------------
# GRADIO INTERFACE
# ---------------------------
with gr.Blocks(title="Chatbot Tư Vấn Tuyển Sinh - Đại Học Đông Á") as demo:

    # HEADER
    gr.HTML("""
    <div style="padding:20px; background-color:#f5f5f5; border-radius:10px;">
        <h1>🎓 Chatbot Tư Vấn Tuyển Sinh – Đại Học Đông Á</h1>
        <p>Hỏi tôi bất kỳ thông tin nào về tuyển sinh!</p>
    </div>
    """)

    with gr.Row():
        # CHATBOT COLUMN
        with gr.Column(scale=8):
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(
                placeholder="Nhập câu hỏi của bạn...",
                label="Câu hỏi",
                lines=1
            )
            msg.submit(chat_fn, [msg, chatbot], chatbot)

        # EXAMPLE QUESTIONS COLUMN
        with gr.Column(scale=4):
            gr.HTML("""
            <div style="background-color:#f0f8ff; padding:15px; border-radius:10px;">
                <h3>💡 Ví dụ câu hỏi:</h3>
                <ul>
                    <li>Hồ sơ xét tuyển cần những gì?</li>
                    <li>Học phí năm 2025 bao nhiêu?</li>
                    <li>Đại học Đông Á có bao nhiêu phương thức xét tuyển?</li>
                </ul>
            </div>
            """)

    # FOOTER
    gr.HTML("""
    <div style="text-align:center; padding:10px; color:#888;">
        © 2025 Đại học Đông Á - Hệ thống Chatbot tư vấn tuyển sinh
    </div>
    """)

demo.launch()
