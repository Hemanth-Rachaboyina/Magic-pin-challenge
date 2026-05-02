# Magicpin AI Challenge: Implementation Plan
**Objective:** Build a 50/50 scoring, deterministic, high-compulsion message engine (Vera) that perfectly adheres to the judge's scoring rubric. Winning is the only option.

## Phase 1: Foundation & API Scaffolding
*Goal: Establish a rock-solid, zero-error API contract that passes the test harness warmup phase effortlessly.*

- [ ] **Environment Setup:** Initialize a Python virtual environment and install core dependencies (`fastapi`, `uvicorn`, `pydantic`, `google-genai`, `python-dotenv`).
- [ ] **Data Validation (`schemas.py`):** Define strict Pydantic models mapping exactly to the `api-call-examples.md` contract. This guarantees we never fail a test due to missing keys or malformed JSON.
- [ ] **State Management (`store.py`):** Implement an extremely fast in-memory `ContextStore`.
    - Must handle API idempotency strictly (ignore duplicate pushes, overwrite on higher versions).
    - Must track accurate counts for the `/v1/healthz` endpoint.
    - Must manage conversational history for multi-turn replies.
- [ ] **API Routing (`main.py`):** Implement the 5 required endpoints:
    - `GET /v1/healthz`
    - `GET /v1/metadata`
    - `POST /v1/context`
    - `POST /v1/tick`
    - `POST /v1/reply`

## Phase 2: The LLM Engine ("Vera's Brain")
*Goal: Build the cognitive engine using Gemini 2.5 Pro, heavily anchoring it to the provided data to maximize Specificity, Decision Quality, and Engagement scores.*

- [ ] **LLM Integration (`engine.py`):** Setup the Google GenAI SDK.
- [ ] **Implement Structured Outputs:** Bind our Pydantic response models directly to the Gemini API calls. This forces the LLM to return valid JSON with `body`, `cta`, and `rationale` every single time.
- [ ] **Prompt Engineering for `/v1/tick`:**
    - Construct a prompt that injects the Merchant, Category, and Trigger JSON safely.
    - Explicitly write rules in the prompt targeting the scoring rubric (e.g., *"CRITICAL: You must extract and use at least one specific numerical metric from the performance data. Do not use generic greetings."*).
- [ ] **Prompt Engineering for `/v1/reply`:**
    - Implement an Intent Classifier. The LLM must look at the conversation history and the new message to classify the scenario:
        - **Hostile/Stop:** Output Action `end`.
        - **Auto-Reply:** Output Action `wait`.
        - **Engaged:** Output Action `send` + draft follow up.
        - **Curveball/Out-of-scope:** Output Action `send` + polite pivot.

## Phase 3: Benchmarking & Iteration (The Crucible)
*Goal: Achieve an elite score on the local `judge_simulator.py` without overfitting.*

- [ ] **Dry Run:** Boot the FastAPI server locally (`uvicorn main:app`).
- [ ] **Simulator Execution:** Run `python judge_simulator.py` against `http://localhost:8000`.
- [ ] **Score Analysis:** Review the output matrix. Where are we losing points? (Usually Specificity or Category Fit).
- [ ] **Iterative Prompt Refinement:** Tweak the system prompts to fix weaknesses. If it's hallucinating, harden the negative constraints. If it's boring, improve the engagement rules.
- [ ] **Test Edge Cases:** Ensure graceful handling when optional data (like `customer` context) is missing.

## Phase 4: Productionization & Submission
*Goal: Ensure the bot is unkillable during the real judge's dynamic injection scenarios.*

- [ ] **Resilience:** Add global exception handlers in FastAPI. If the LLM times out or fails, return a safe fallback empty action `{ "actions": [] }` rather than crashing the test slot.
- [ ] **Documentation:** Write the required one-page `README.md` explaining the architecture, the choice of Gemini 2.5 Pro (for its superior instruction following and context window), and the structured output strategy.
- [ ] **Deployment:** Host the application on a fast, reliable public cloud provider (e.g., Render, Railway).
- [ ] **Final Verification:** Run the local judge one last time against the *public* URL to verify network latency is within the judge's budget.
- [ ] **Submit:** Provide the public bot URL to magicpin.
