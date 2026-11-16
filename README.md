# 🌑 TSUKI // Spatial Enterprise RAG Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-HNSW%20Vector%20DB-FF5722?style=flat&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM%20(Qwen%202.5)-000000?style=flat&logo=ollama&logoColor=white)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **TSUKI (月)** is a production-grade, containerized "local-first" Retrieval-Augmented Generation (RAG) system paired with a glassmorphic spatial OS dashboard. Built for high-performance AI engineering portfolios, it bridges robust backend vector infrastructure with zero-latency spatial UI/UX.
<img width="1920" height="965" alt="Screenshot (37)" src="https://github.com/user-attachments/assets/05613959-5801-4ac8-bf03-c16d30f2ec22" />
<img width="1920" height="964" alt="Screenshot (38)" src="https://github.com/user-attachments/assets/c1e6fe05-cd3f-47a6-9548-d138df38e55c" />


---

## 🏛️ System Architecture

```text
[ PDF Document ] ---> [ Sliding Window Chunking (300/30) ] ---> [ Ollama (nomic-embed-text) ]
                                                                             │
[ TSUKI Glassmorphic UI ] <--- [ FastAPI Backend / Query Engine ] <──────────┘
                                            │
                                            ▼
                            [ Qdrant Docker (HNSW / Cosine) ]
                                            │
                                            ▼
                             [ Ollama (qwen2.5:7b) ] ---> [ Grounded Response ]
