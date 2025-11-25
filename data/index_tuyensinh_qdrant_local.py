import json
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ------------------------------
# Load ENV
# ------------------------------
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "tuyensinh"

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=120
)

# ------------------------------
# 2) Load Sentence Transformer
# ------------------------------
print("Loading embedding model...")
model = SentenceTransformer("intfloat/multilingual-e5-base")  # 768 dims

VECTOR_SIZE = model.get_sentence_embedding_dimension()
print("Vector size:", VECTOR_SIZE)


# ------------------------------
# 3) Create collection
# ------------------------------
def create_collection():
    print(f"Creating collection '{COLLECTION_NAME}'...")

    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

    print("✔ Collection created!")


# ------------------------------
# 4) Import JSONL chunks
# ------------------------------
def import_jsonl():
    print("Loading chunks from tuyensinh_chunks.jsonl ...")

    points = []
    with open("tuyensinh_chunks.jsonl", "r", encoding="utf8") as f:
        for line in tqdm(f):
            item = json.loads(line)

            embedding = model.encode(item["text"]).tolist()

            point = PointStruct(
                id=item["id"],
                vector=embedding,
                payload={
                    "text": item["text"],
                    "title": item["title"],
                    "url": item["url"],
                    "chunk_id": item["chunk_id"]
                }
            )

            points.append(point)

    print(f"Uploading {len(points)} vectors to Qdrant...")

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print("✔ Upload completed!")


if __name__ == "__main__":
    create_collection()
    import_jsonl()
