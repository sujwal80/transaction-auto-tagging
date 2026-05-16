# Financial Tagging & Categorization Agent MVP

A robust, three-pass financial transaction tagging pipeline designed for high accuracy and zero-tolerance for silent errors. This MVP demonstrates a "Maker-Critic" architecture with RAG-based context assembly and adversarial guardrails.

## 🚀 Architecture Overview

The system processes transactions through three sequential passes to ensure maximum reliability:

1.  **Pass 1: Deterministic Rules Engine**
    - High-speed, zero-latency filtering.
    - Uses MCC (Merchant Category Code) mapping and exact merchant string matching.
    - Bypasses AI entirely for known patterns to ensure 100% precision.

2.  **Pass 2: Maker LLM (Gemini Flash 1.5)**
    - Performs RAG-based categorization using a tenant-specific Chart of Accounts (CoA).
    - Assembles context from a mock vector index.
    - emulates ReAct reasoning steps for complex transactions.

3.  **Pass 3: Robust Guardrail Pipeline (Audit Critic)**
    - **Materiality Filter:** Forces manual review for transactions exceeding $5,000.
    - **Confidence Filter:** Flags predictions with low confidence scores.
    - **OOD Filter:** Detects Out-of-Distribution anomalies via vector distance.
    - **Adversarial Audit Critic (Claude Opus):** Validates the Maker's output against negative tenant policies (e.g., travel spend limits).

## 🛠️ Setup & Installation

### 1. Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

## 🏃 Running the Simulation

The end-to-end simulation processes 5 distinct edge cases (Standard Match, Low Confidence, High Materiality, Policy Violation, and a "Poison Pill" for DLQ testing).

```powershell
python main.py
```

## 🧪 Running Tests

The project includes a comprehensive suite of unit tests for all architectural layers.

```powershell
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
python -m pytest -v
```

## 📁 File Structure

- `engine/`: The core processing logic (Pass 1, 2, and 3).
- `models/`: Data models for payloads and Chart of Accounts.
- `worker/`: The asynchronous worker daemon and queue orchestration.
- `config.py`: Centralized thresholds and environment settings.
- `main.py`: Entry point for the simulation.
- `*_test.py`: Unit tests located alongside the modules.
