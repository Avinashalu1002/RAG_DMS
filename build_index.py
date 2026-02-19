import psycopg2
import requests
import numpy as np
import faiss

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "nomic-embed-text"

DB_CONFIG = {
    "host": "localhost",
    "database": "pharma",
    "user": "postgres",
    "password": "1234",
    "port": "5433"
}
# Connect DB
connection = psycopg2.connect(**DB_CONFIG)
cursor = connection.cursor()

print("Loading approved chunks from DB...")

cursor.execute("""
    SELECT c.chunk_id, c.chunk_text
    FROM dms_chunk c
    JOIN dms_document d ON c.doc_id = d.doc_id
    WHERE d.status = 'Approved'
""")

rows = cursor.fetchall()

if not rows:
    print("No approved chunks found!")
    exit()

chunk_ids = []
embeddings = []

print("Generating embeddings ")

for chunk_id, chunk_text in rows:

    response = requests.post(OLLAMA_URL,json={ "model": MODEL_NAME,"prompt": chunk_text })

    if response.status_code != 200:
        print("Embedding error:", response.text)
        exit()

    vector = response.json()["embedding"]

    chunk_ids.append(chunk_id)
    embeddings.append(vector)

print("Creating FAISS index...")

embed_matrix = np.array(embeddings).astype("float32")

faiss.normalize_L2(embed_matrix)

dimension = embed_matrix.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embed_matrix)

faiss.write_index(index, "faiss_index.bin")

cursor.close()
connection.close()
