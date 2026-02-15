# BFSI Call Center AI Assistant

📌 Project Overview

The BFSI Call Center AI Assistant is a lightweight, compliant AI system designed to handle common customer queries in the Banking, Financial Services, and Insurance (BFSI) domain.

Use cases:
1. Semantic Retrieval (Sentence Transformers)
2. Policy-Grounded RAG Architecture
3. Local Small Language Model (SLM) via Ollama
4. Controlled Prompt Engineering
5. Threshold-Based Fallback Logic

The assistant supports queries related to:
1. Loan eligibility and application status
2. EMI details and schedules
3. Interest rate information
4. Payment and transaction queries
5. Basic account support

📌 System Architecture

User Query
    ↓
Query Embedding (all-mpnet-base-v2)
    ↓
Similarity Check Against Alpaca Dataset
    ↓
[Tier 1] Strong Match?
    ├── YES → Return Stored Response
    └── NO  → Local Fine-Tuned SLM
                    ↓
             [Tier 3] Complex Policy Query?
                    ├── YES → RAG Retrieval Layer
                    └── NO  → Direct SLM Response
    ↓
Final Response

📌 Core Components

1️⃣ Dataset Layer (Primary Response Layer)
1. 150+ BFSI conversation samples
2. Alpaca format (Instruction, Input, Output)
3. Professional and compliant tone
4. Embedded using all-mpnet-base-v2

Role:
If cosine similarity score exceeds threshold →
Return stored response directly.
This ensures:
1. Deterministic answers
2. Zero hallucination
3. Policy-safe responses

2️⃣ Small Language Model (Local SLM)
1. Lightweight instruction-based model
2. Runs locally via Ollama
3. Model: phi3:mini

Role:
Used only when:
No strong dataset similarity match is found
This provides flexibility while maintaining safety.

3️⃣ RAG Layer (Knowledge Retrieval)
Used for complex financial or policy-related queries such as:
1. Interest rate explanations
2. EMI breakdowns
3. Prepayment penalties
4. Foreclosure policies

Implementation:
1. Structured knowledge base (knowledge_base.json)
2. Embedded into vectors
3. Retrieved via cosine similarity
4. Injected into controlled prompt

Role:
Provide grounded, policy-based responses.

📌 Response Priority Logic

Tier	  Condition	Action
Tier 1	Strong dataset similarity	Return stored response
Tier 2	No strong match	Generate via local SLM
Tier 3	Complex financial query	Use RAG retrieval

This ensures a safety-first design.

📌 Guardrails & Compliance

The assistant enforces strict BFSI safety rules:
1. No guessing financial numbers
2. No fake interest rates
3. No fabricated policies
4. No exposure of customer-specific data
5. Reject out-of-domain queries

Low-temperature generation (0.3) is used to reduce randomness.

📌 Technical Stack

Embeddings → Sentence Transformers

Model → all-mpnet-base-v2

Local → LLM	Ollama

SLM Example	→ phi3:mini

Similarity → Cosine Similarity

Data Format	→ Alpaca

📌 Project Structure

main.py                     → Orchestrates full pipeline

generate_embeddings.py      → Creates dataset embeddings

prepare_rag_embedding.py    → Creates knowledge base embeddings

rag_embedding_service.py    → Semantic retrieval logic

prompt_builder.py           → Structured prompt construction

slm_service.py              → Local model inference via Ollama

similarity_check.py         → Dataset similarity evaluation

knowledge_base.json         → Policy knowledge documents

alpaca_bfsi_sample.json     → BFSI dataset

📌 Outcome

A modular, production-style AI assistant architecture demonstrating:
1. RAG implementation
2. Safe LLM deployment
3. Semantic retrieval
4. AI system design for regulated industries



