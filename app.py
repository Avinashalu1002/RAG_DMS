
import json
import psycopg2
import requests
import numpy as np
import faiss
from flask import Flask, request, jsonify

ollama_embedding = "http://localhost:11434/api/embeddings"
ollama_llm = "http://localhost:11434/api/generate"

embeding_model = "nomic-embed-text"
llm_model = "gemma3:1b"

K = 5

database_details = {
    "host": "localhost",
    "database": "pharma",
    "user": "postgres",
    "password": "1234",
    "port": "5433"
}

# Initialize flask

app = Flask(__name__)

try:
    index = faiss.read_index("faiss_index.bin")
    print("FAISS  loaded")
except:
    index = None
    print("FAISS not found")

# data base connection

def database_connection():
    return psycopg2.connect(**database_details)

# validation layer 
def validate_query(query):
    if not query or not query.strip():
        return "Query cannot be empty."
    if len(query.strip()) < 5:
        return "Query too short."
    return None

# using OLLAMA
def start_embedding(text):

    response = requests.post(
        ollama_embedding,
        json={
            "model": embeding_model,
            "prompt": text
        }
    )

    if response.status_code != 200:
        raise Exception("error: " + response.text)

    return np.array([response.json()["embedding"]], dtype="float32")

# chunks

def load_all_chunks():

    connection = database_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT chunk_id, chunk_text
        FROM dms_chunk
        ORDER BY chunk_id
    """)

    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    return rows

#governance layer

def is_chunk_governed(chunk_id):

    connection = database_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT d.status
        FROM dms_chunk c
        JOIN dms_document d ON c.doc_id = d.doc_id
        WHERE c.chunk_id = %s
    """, (chunk_id,))

    result = cursor.fetchone()

    cursor.close()
    connection.close()

    if result and result[0] == "Approved":
        return True

    return False

# OLLAMA LLM answer generation

def gen_answer(question, context_chunks):

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a Pharma DMS assistant.

Answer ONLY from the context below.
If answer not present, say:
" No evidence in approved in database."

Question:
{question}

Context:
{context}

Answer:
"""

    response = requests.post(
        ollama_llm,
        json={
            "model": llm_model,
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code != 200:
        raise Exception("llm error: " + response.text)

    return response.json()["response"]

# flask router

@app.route("/")
def home():
    return "RAG Running"

@app.route("/query", methods=["POST"])
def query():

    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Query field required"}), 400

    user_query = data["query"]

    error = validate_query(user_query)
    if error:
        return jsonify({"error": error}), 400

    try:

        if index is None:
            return jsonify({"error": "FAISS  not loaded"}), 500

        # Embed query
        query_embed = start_embedding(user_query)

        # Loading  chunks
        rows = load_all_chunks()

        if not rows:
            return jsonify({
                "answer": "No documents found.",
                "citations": []
            })

        chunk_ids = [row[0] for row in rows]
        chunk_texts = [row[1] for row in rows]

        # FAISS Search
        D, I = index.search(query_embed, K)

        citations = []
        context_chunks = []

        for i, idx in enumerate(I[0]):

            if idx >= len(chunk_ids):
               continue

            distance = D[0][i]
            print("Distance:", distance)

            citations.append({
        "chunk_id": chunk_ids[idx],
        "distance": float(distance),
        "snippet": chunk_texts[idx][:200]
    })

            context_chunks.append(chunk_texts[idx])

        # Refuse 
        if not citations:
            return jsonify({
                "answer": "No evidence in approved documents.",
                "citations": []
            })

        # LLM answer
        final_answer = gen_answer(user_query, context_chunks)

        return jsonify({
            "answer": final_answer,
            "citations": citations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# running flask
if __name__ == "__main__":
    app.run(debug=True, port=5000)


