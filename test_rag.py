import requests
import os

BASE_URL = "http://127.0.0.1:8000"


def test_full_pipeline():
    print("🚀 Starting RAG Pipeline Integration Tests...\n")

    # 1. Verify Health
    health_res = requests.get(f"{BASE_URL}/health")
    assert health_res.status_code == 200, f"Health check failed with status {health_res.status_code}"
    health_data = health_res.json()
    assert health_data.get("status") == "healthy" or "qdrant" in health_data
    print(f"[✓] Health Check Passed: {health_data}")

    # 2. Test Ingestion (Using sample_doc.pdf)
    sample_pdf = "sample_doc.pdf"
    if os.path.exists(sample_pdf):
        with open(sample_pdf, "rb") as f:
            files = {"file": (sample_pdf, f, "application/pdf")}
            ingest_res = requests.post(f"{BASE_URL}/upload-pdf", files=files)

        assert ingest_res.status_code == 200, f"Ingestion failed with status {ingest_res.status_code}"
        ingest_data = ingest_res.json()
        print(f"[✓] Ingestion Passed: Filename={ingest_data.get('filename')}, Status={ingest_data.get('status')}")
    else:
        print(f"[!] Warning: '{sample_pdf}' not found locally. Skipping upload test.")

    # 3. Test Grounded Query
    query_payload = {"query": "What are the core infrastructure requirements for RAG microservices?"}
    query_res = requests.post(f"{BASE_URL}/query", json=query_payload)

    assert query_res.status_code == 200, f"Query endpoint failed with status {query_res.status_code}"
    query_data = query_res.json()

    # Assertions on response structure
    assert "answer" in query_data, "Response missing 'answer' field"
    assert "retrieved_contexts" in query_data, "Response missing 'retrieved_contexts' field"
    assert len(query_data["retrieved_contexts"]) > 0, "No context chunks retrieved from Qdrant"

    print(f"[✓] Query & Generation Passed!")
    print(f"    - Answer: {query_data['answer'][:120]}...")
    print(f"    - Retrieved Chunks Count: {len(query_data['retrieved_contexts'])}")

    print("\n🎉 All integration tests successfully verified!")


if __name__ == "__main__":
    test_full_pipeline()