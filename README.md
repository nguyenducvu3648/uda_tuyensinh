# Chatbot Tư Vấn Tuyển Sinh – Đại Học Đông Á

Ứng dụng chatbot giúp tư vấn thông tin tuyển sinh dựa trên dữ liệu từ Qdrant và sử dụng LLM Gemini của Google.

---

## Yêu cầu
- Python 3.10 trở lên
- Pip

---

## Cài đặt

1. **Clone repository**:
```bash
git clone <link-repo-của-bạn>
cd GenAI
2. ****Cài đặt thư viện cần thiết:**
pip install -r requirements.txt

3. Thiết lập biến môi trường

Tạo file .env trong thư mục gốc với nội dung:

QDRANT_URL=<url của Qdrant>
QDRANT_API_KEY=<api key của Qdrant>
GEMINI_API_KEY=<api key của Gemini LLM>

4. Chạy
python main.py
