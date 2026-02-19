import json
import requests

API_URL = "http://127.0.0.1:5000/query"

with open("evaluation_questions.json", "r") as f:
    questions = json.load(f)

total = len(questions)
correct_answers = 0
refusals = 0
correct_retrieval = 0

for item in questions:

    response = requests.post(API_URL, json={"query": item["question"]})

    result = response.json()

    answer = result.get("answer", "")
    citations = result.get("citations", [])

    print("\nQuestion:", item["question"])
    print("Answer:", answer)

    # Refusal check
    if "No evidence" in answer:
        refusals += 1

    # Retrive check
    if item["expected_doc"]:
        for c in citations:
            if item["expected_doc"] in c["chunk_id"]:
                correct_retrieval += 1
                break

    # Simple accuracy check
    if item["expected_doc"] is None and "No" in answer:
        correct_answers += 1
    elif item["expected_doc"] and citations:
        correct_answers += 1

# Metrics
print(" Questions:", total)
print("Accuracy:", round(correct_answers / total, 2))
print("Retrive Recall:", round(correct_retrieval / total, 2))
print("Refusal Rate:", round(refusals / total, 2))
