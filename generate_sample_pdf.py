from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_sample_pdf(filename="sample_doc.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(54, height - 54, "Enterprise AI Architecture & Security Standards (v3.2)")

    # Body Content
    c.setFont("Helvetica", 11)
    text_lines = [
        "Section 1 (Core Infrastructure): All production RAG microservices must run containerized",
        "using Docker on isolated networks. Vector embeddings must be stored in Qdrant utilizing",
        "768-dimensional dense vectors with Cosine distance indexing.",
        "",
        "Section 2 (Security & Governance): User prompts must pass through PII masking pipelines",
        "to redact sensitive identifiers before reaching vector stores or LLM endpoints.",
        "",
        "Section 3 (Compliance & Auditing): Every retrieval operation must log source filenames,",
        "page numbers, and cosine similarity scores for auditing and traceability."
    ]

    y = height - 100
    for line in text_lines:
        c.drawString(54, y, line)
        y -= 20

    c.save()
    print(f"[*] Generated sample document: {filename}")


if __name__ == "__main__":
    create_sample_pdf()