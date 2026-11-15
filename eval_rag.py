import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# Define custom test dataset with expected ground-truth keywords and source references
EVAL_DATASET = [
    {
        "query": "What are the core infrastructure requirements for RAG microservices?",
        "expected_keywords": ["containerized", "Docker", "Qdrant", "768-dimensional"],
        "expected_source": "sample_doc.pdf"
    },
    {
        "query": "What distance metric is used for vector indexing in Qdrant?",
        "expected_keywords": ["Cosine"],
        "expected_source": "sample_doc.pdf"
    }
]


def run_evaluation():
    print("🧪 Starting RAG Pipeline Evaluation (Recall & Faithfulness)...\n")

    total_tests = len(EVAL_DATASET)
    retrieval_passes = 0
    faithfulness_passes = 0

    for i, test in enumerate(EVAL_DATASET, 1):
        print(f"--- Test Case {i}/{total_tests} ---")
        print(f"Query: \"{test['query']}\"")

        try:
            # Call your FastAPI query endpoint
            response = requests.post(f"{BASE_URL}/query", json={"query": test["query"]})
            if response.status_code != 200:
                print(f"[✗] API Error: Status {response.status_code}\n")
                continue

            data = response.json()
            answer = data.get("answer", "")
            contexts = data.get("retrieved_contexts", [])

            # 1. Evaluate Retrieval Recall (Did we retrieve chunks and match the source?)
            retrieved_sources = [ctx.get("source", "") for ctx in contexts]
            source_matched = any(test["expected_source"] in src for src in retrieved_sources)
            has_contexts = len(contexts) > 0

            if has_contexts and source_matched:
                retrieval_passes += 1
                print(
                    f"[✓] Retrieval Recall: PASSED (Found source '{test['expected_source']}' in top-{len(contexts)} chunks)")
            else:
                print(f"[✗] Retrieval Recall: FAILED (Retrieved sources: {retrieved_sources})")

            # 2. Evaluate LLM Faithfulness (Did the answer include expected keywords grounded in context?)
            keywords_found = [kw.lower() in answer.lower() for kw in test["expected_keywords"]]
            faithfulness_score = sum(keywords_found) / len(test["expected_keywords"])

            # If at least 75% of expected keywords appear in the generated answer, count as faithful/accurate
            if faithfulness_score >= 0.75:
                faithfulness_passes += 1
                print(f"[✓] LLM Faithfulness: PASSED (Keyword match score: {faithfulness_score * 100:.0f}%)")
            else:
                print(f"[✗] LLM Faithfulness: FAILED (Keyword match score: {faithfulness_score * 100:.0f}%)")
                print(f"    - Generated Answer: {answer}")

            print("-" * 40 + "\n")

        except Exception as e:
            print(f"[✗] Exception during test execution: {e}\n")

    # Summary Report
        # Summary Report
        print("📊 === EVALUATION SUMMARY ===")
        print(f"Total Test Cases: {total_tests}")
        print(f"Retrieval Recall Rate: {(retrieval_passes / total_tests) * 100:.1f}%")
        print(f"LLM Faithfulness Rate: {(faithfulness_passes / total_tests) * 100:.1f}%")
        print("=============================")

if __name__ == "__main__":
    run_evaluation()