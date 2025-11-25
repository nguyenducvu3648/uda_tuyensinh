import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
from tqdm import tqdm
import re
import uuid

BASE = "https://donga.edu.vn"
MENU_URL = "https://donga.edu.vn/tuyensinh"


def clean_text(text):
    """Xóa khoảng trắng thừa, newline, tab"""
    return re.sub(r'\s+', ' ', text).strip()


def filter_text(text):
    """
    Loại bỏ các dòng rác: menu, quảng cáo, ngày tháng lặp, dòng quá ngắn
    """
    # split thành câu/dấu chấm
    lines = re.split(r'(?<=[.!?]) +', text)
    clean_lines = []

    for line in lines:
        line = line.strip()
        # loại bỏ dòng quá ngắn
        if len(line) < 20:
            continue
        # loại bỏ dòng bắt đầu bằng ngày tháng kiểu dd/mm/yyyy
        if re.match(r'^\d{2}/\d{2}/\d{4}', line):
            continue
        # loại bỏ quảng cáo/menu cố định
        if re.search(r'Trang chủ|Giới thiệu|Tuyển sinh|Tin tức|Thông báo|Đại học|Sinh viên|CB - GV|Liên hệ', line):
            continue
        clean_lines.append(line)

    return ". ".join(clean_lines)


def extract_article(url):
    """Crawl 1 trang tuyển sinh & trích xuất dữ liệu"""
    try:
        res = requests.get(url, timeout=15)
        soup = BeautifulSoup(res.text, "html.parser")

        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else soup.title.get_text(strip=True) if soup.title else "No Title"

        # Lấy vùng content chính ưu tiên
        content_div = soup.find("div", class_="article-body") \
                      or soup.find("div", class_="post-content") \
                      or soup.find("div", class_="main-content") \
                      or soup.find("div", class_="content") \
                      or soup

        raw_text = clean_text(content_div.get_text(" "))
        content_text = filter_text(raw_text)

        images = []
        for img in content_div.find_all("img"):
            src = img.get("src")
            if src:
                images.append(urljoin(BASE, src))

        return {
            "url": url,
            "title": title,
            "content_text": content_text,
            "html": str(content_div),
            "images": images
        }

    except Exception as e:
        return {"url": url, "error": str(e), "title": "No Title"}


def extract_menu_links():
    """Crawl page Tuyển sinh để lấy toàn bộ link trong menu"""
    res = requests.get(MENU_URL)
    soup = BeautifulSoup(res.text, "html.parser")

    links = []

    for a in soup.find_all("a"):
        label = a.get_text(strip=True)
        href = a.get("href")
        if not label:
            continue
        if "tuyen" in label.lower() or "đại" in label.lower() or "sinh" in label.lower():
            if href and href != "#":
                full = urljoin(BASE, href)
            else:
                full = None
            links.append({"label": label, "url": full})

    return links


def filter_valid_links(links):
    """Loại bỏ null + domain ngoài + trùng"""
    unique = set()
    valid = []
    for item in links:
        if item["url"] and item["url"].startswith(BASE):
            if item["url"] not in unique:
                unique.add(item["url"])
                valid.append(item)
    return valid


def save_json(data, filename="tuyensinh_raw.json"):
    with open(filename, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Saved:", filename)


def chunk_text(text, size=800):
    """
    Chia text thành chunks khoảng size ký tự,
    cố gắng không cắt câu giữa chừng
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def save_jsonl(docs, filename="tuyensinh_chunks.jsonl"):
    with open(filename, "w", encoding="utf8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("Saved:", filename)


def main():
    print("▶ Extracting menu links...")
    menu_links = extract_menu_links()
    print("Found raw menu items:", len(menu_links))

    valid_links = filter_valid_links(menu_links)
    print("Valid internal links:", len(valid_links))

    results = []

    print("▶ Crawling pages...")
    for item in tqdm(valid_links):
        url = item["url"]
        data = extract_article(url)
        data["label"] = item["label"]
        results.append(data)

    save_json(results)

    # Chunking
    print("▶ Chunking content...")
    jsonl_docs = []

    for doc in results:
        if "content_text" not in doc or not doc["content_text"]:
            continue
        chunks = chunk_text(doc["content_text"], size=800)

        for idx, chunk in enumerate(chunks):
            jsonl_docs.append({
                "id": str(uuid.uuid4()),  # ID hợp lệ cho Qdrant
                "url": doc["url"],
                "title": doc.get("title", "No Title"),
                "chunk_id": idx,
                "text": chunk
            })

    save_jsonl(jsonl_docs)


if __name__ == "__main__":
    main()
