# 🧠 Advanced RAG System – Insurelm Knowledge Assistant

This project implements an advanced Retrieval-Augmented Generation (RAG) system that answers everything related to the company "Insurelm".

It focuses on improving retrieval quality, ranking accuracy, and multi-document reasoning using modern RAG optimization techniques.

---

# 🚀 System Architecture

Documents → Chunking → Embeddings → ChromaDB  
                                    ↓  
                               Retriever  
                                    ↓  
                                   LLM  
                                    ↓  
                                Final Answer  

LangChain provides two key abstractions:

- Retriever → Returns most relevant chunks
- LLM → Uses retrieved context to generate final answer

---

# 📂 Project Structure

- ingest.py → Document loading, chunking, embedding, and storing in ChromaDB
- answer.py → Retrieval pipeline + LLM answer generation
- evals/ → Evaluation metrics & curated test set

---

# 📄 Document Processing

### 1️⃣ Loading
- LangChain document loaders

### 2️⃣ Chunking
- RecursiveCharacterTextSplitter
- Tuned chunk sizes for optimal retrieval
- Semantic chunking (split by meaning, not only characters)

### 3️⃣ Preprocessing
- LLM-based rewriting for cleaner retrieval
- Query-optimized text before storing in vector DB

---

# 🧮 Embedding Models Tested

1. HuggingFace SentenceTransformer  
   - all-MiniLM-L6-v2

2. OpenAI text-embedding-small

3. OpenAI text-embedding-3-large  
   - 3072 dimensions  
   - 413 vectors stored  
   - Best retrieval performance

Embeddings stored in ChromaDB and visualized using t-SNE.

---

# 🔍 Retrieval Enhancements

Several techniques were implemented to improve RAG performance:

## Query Optimization
- Query rewriting (LLM reformulates question)
- Query expansion (multiple RAG queries generated)

## Retrieval Improvements
- Larger chunk retrieval (increase K)
- Better embedding model selection
- Prompt engineering improvements

## Re-ranking
- LLM-based re-ranking to select best chunks from retrieved set

## Hierarchical Summarization
- Multi-level summarization
- Effective for questions spanning multiple documents

---

# 📊 Evaluation Metrics

A curated evaluation dataset was created.

Metrics used:

- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG)
- Recall@K
- Precision@K

These metrics helped quantify improvements from:
- Better chunking strategy
- Stronger embedding models
- Query rewriting & expansion
- Re-ranking techniques

---

# 🧪 Vector Store Details

- Vector DB: ChromaDB
- Embedding dimension (OpenAI large): 3072
- Number of stored vectors: 413
- Visualization: t-SNE projection of embedding space

---

# 🎯 Key Improvements Over Basic RAG

- Semantic chunking instead of fixed-size splits
- LLM-assisted document rewriting before storage
- Query rewriting and expansion
- LLM-based re-ranking
- Hierarchical summarization for cross-document reasoning
- Quantitative evaluation using ranking metrics

---

# 🛠 Tech Stack

- LangChain
- ChromaDB
- OpenAI Embeddings
- HuggingFace Sentence Transformers
- Python
- t-SNE (for embedding visualization)

---

# 📌 Future Work

- Hybrid search (BM25 + vector)
- Cross-encoder re-ranking
- Dynamic chunk sizing
- Production deployment with streaming responses
- Automated regression evaluation pipeline

---

# ⭐ Summary

This project demonstrates how to move from a basic RAG pipeline to an advanced, evaluation-driven, production-ready retrieval system with measurable ranking improvements.