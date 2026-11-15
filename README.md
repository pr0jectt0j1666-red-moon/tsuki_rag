# 🌑 TSUKI // Spatial Enterprise RAG Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-HNSW%20Vector%20DB-FF5722?style=flat&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM%20(Qwen%202.5)-000000?style=flat&logo=ollama&logoColor=white)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **TSUKI (月)** is a production-grade, containerized "local-first" Retrieval-Augmented Generation (RAG) system paired with a glassmorphic spatial OS dashboard. Built for high-performance AI engineering portfolios, it bridges robust backend vector infrastructure with zero-latency spatial UI/UX.

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