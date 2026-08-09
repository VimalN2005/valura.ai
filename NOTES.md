# Notes - Valura AI Arena Multi-Agent Assignment
**Candidate Email**: vimalnasit20@gmail.com

---

## 1. How to Build, Run, and Test

### Prerequisites
- Python 3.11+
- Pip

### Local Setup & Execution
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Export your Valura API Key to the environment:
   ```powershell
   # Windows PowerShell
   $env:VALURA_API_KEY="vlr_9MgCCpiMEKBico7RT_N2aqRP-Du-cYOO"
   ```
3. Run in practice mode:
   ```bash
   python -m harness.run --mode practice
   ```

### Running with Docker
1. Build the Docker image:
   ```bash
   docker build -t valura-agents .
   ```
2. Run the container against the qualifying/graded API:
   ```bash
   docker run --rm -e ASSESSMENT_KEY="vlr_9MgCCpiMEKBico7RT_N2aqRP-Du-cYOO" -e ASSESSMENT_URL="https://ai-arena.twocc.in" valura-agents --mode qualifying
   ```

### Running Tests
Execute the automated test suite to verify masking, temporal as-of filtering, and coverage bounds:
```bash
python -m unittest tests/test_harness.py
```

---

## 2. Architecture & Division of Labor

Our multi-agent ecosystem utilizes a hybrid programmatic-agentic orchestrator:
- **Router (valura-fast)**: Extracts temporal `as_at` dates (e.g. `2025-06-30`) via regex, initializes the thread-local execution context, classifies the prompt's intent, and dispatches the query sequentially to the required specialists. It keeps track of the active specialists to build the `agents` execution path (e.g., `["router", "book_qa"]`).
- **Specialists**: 
  - `book_qa` (`valura-fast`): Computes transaction counts, balances, and allocations.
  - `kyc_profile` (`valura-fast`): Handles DOB, address, and profile queries.
  - `notes_desk` (`valura-deep`): Reads and summarizes ops memos.
  - `market_desk` (`valura-fast`): Retrieves prices and news.
  - `compliance` (`valura-fast`): Enforces policy refusals.
- **Division of Labor**: The LLM's role is strictly limited to natural language comprehension, intent routing, and final response synthesis. All math, decimal quantizations, KYC masking (`****XXXX`), temporal filters, conflict checks, and market coverage boundary checks are executed deterministically in Python tools.

---

## 3. Decisions and Inquiries

- **Portfolio Drift**: Calculated drift using total portfolio value (cash + stock value) as the denominator because target allocations sum to 100%. Unspecified assets default to a 0% target.
- **Citation Compression**: Implemented the citation limit check in Python tools. If the count of referenced records exceeds 6, we automatically compress the citations list to `[client_id]`.
- **Inquiries**: We would have asked:
  1. Are there any other KYC attributes or national rules that make a client out-of-scope?
  2. Are historical prices guaranteed to cover every transaction date, or do we carry forward the latest available price? (We assumed carrying forward is correct).

---

## 4. Specific Technical Questions

### Q1: Deciding on Unanswerable Questions
Our Python tools explicitly throw an `UncoveredSymbolException` if a requested symbol is not in `meta.covered_symbols` or if a record is missing from the local book. When caught, the router directly outputs `abstained=True` and `answer_value=null`. This decision occurs at the Python level, ensuring it is not the model's subjective "uncertainty" but a hard, rules-based boundary.

### Q2: Neutralizing Adversarial Notes Injections
The note-injection attack (e.g., instructing the system to leak all accounts or unmask data) is neutralized at two layers:
1. **Scope Restriction (Python)**: The note tool only loads data for the current `client_id` context. If the model attempts to query another client's notes, the tool asserts a scope mismatch and raises a `PermissionError`.
2. **Deterministic Masking (Python)**: Bank accounts and PANs are masked (`****XXXX`) in the KYC tool before the LLM receives them.
*To fail*: Both our Python scope checks and masking functions would have to be deleted, and the LLM's system instructions would need to be successfully bypassed.

### Q3: Provider Down (LLM Blackout)
- **Unaffected**: KYC lookups, cash balances, transaction lists, and fee summaries. The router intercepts LLM failures and runs `generate_heuristic_answer` in Python to fetch, compute, and format these queries directly from `book.json`.
- **Worse**: Free-text note summary and complex multi-agent questions. These fail back to a clean decline with `flags=["upstream_issue"]`.
- **Slower**: Transient connection drops, which are resolved via HTTP client exponential backoffs.

### Q4: Agno Framework Experience
- **Easy**: System prompt configuration and tool definition.
- **Hard**: Tracking sequential agent handoffs and extracting execution paths.
- **Source Reading**: We read `agno/models/openai/chat.py` to figure out how to pass the custom proxy `base_url` to the underlying client client library as it wasn't documented clearly.

---

## 5. Weaknesses & Future Improvements

- **Weakness**: Complex queries (e.g., historical portfolio valuations or free-text notes) cannot be resolved during a blackout and must be declined.
- **Future Improvements**:
  1. Integrate a tiny local model (like Llama-3-8B-Instruct via llama-cpp-python) inside the Docker container to act as a fallback generator when the LLM proxy is down.
  2. Implement a post-generation grounding verifier to check the LLM's final response against the raw JSON records.
