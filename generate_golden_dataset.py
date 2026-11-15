import json
import random
from pathlib import Path
import pypdf
from langchain_ollama import ChatOllama

# 1. Initialize the local Ollama model (change "llama3.2:latest" if your ollama list shows a different name)
llm = ChatOllama(model="llama3.2:latest", temperature=0.3, format="json")

PROMPT_TEMPLATE = """
You are creating a strict RAG evaluation dataset from the document excerpt below.
Generate 3 distinct question-and-answer pairs in valid JSON format with a root key "items":
1. "single_hop": A specific factual question directly answered in one sentence of the text.
2. "multi_hop": A reasoning question requiring two separate facts from the text to answer completely.
3. "unanswerable": A plausible-sounding question related to the topic whose answer is NOT present anywhere in the text. Set its ground_truth to "I cannot answer this based on the provided documents."

Return ONLY JSON in this exact schema:
{{
  "items": [
    {{"type": "single_hop", "question": "...", "ground_truth": "..."}},
    {{"type": "multi_hop", "question": "...", "ground_truth": "..."}},
    {{"type": "unanswerable", "question": "...", "ground_truth": "..."}}
  ]
}}

Document Excerpt:
{chunk}
"""


def extract_chunks(pdf_dir: str, chunk_size: int = 2500) -> list[str]:
    chunks = []
    pdf_path_dir = Path(pdf_dir)
    if not pdf_path_dir.exists():
        print(f"Warning: Directory '{pdf_dir}' not found.")
        return chunks

    for pdf_path in pdf_path_dir.glob("*.pdf"):
        reader = pypdf.PdfReader(pdf_path)
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        for i in range(0, len(full_text), chunk_size):
            segment = full_text[i: i + chunk_size].strip()
            if len(segment) > 500:
                chunks.append(segment)
    return chunks


def main():
    # Points to your uploaded_docs folder in the project root
    chunks = extract_chunks("./uploaded_docs")
    if not chunks:
        print("No PDF text chunks found! Make sure your PDFs are inside the 'uploaded_docs' folder.")
        return

    sampled_chunks = random.sample(chunks, min(10, len(chunks)))

    dataset = []
    for idx, chunk in enumerate(sampled_chunks, 1):
        print(f"Generating Q&A pairs from excerpt {idx}/{len(sampled_chunks)}...")
        response = llm.invoke(PROMPT_TEMPLATE.format(chunk=chunk))
        try:
            parsed = json.loads(response.content)
            for item in parsed.get("items", []):
                item["source_excerpt"] = chunk[:200] + "..."
                dataset.append(item)
        except json.JSONDecodeError:
            print(f"Skipped malformed JSON on chunk {idx}")

    with open("golden_dataset_draft.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Success! Saved {len(dataset)} draft items to golden_dataset_draft.json")


if __name__ == "__main__":
    main()