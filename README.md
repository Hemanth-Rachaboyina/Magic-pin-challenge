# Vera Message Engine - AI Architects

## Overview
Vera is a deterministic, highly-optimized AI message engine built for the magicpin AI Challenge. It acts as an autonomous growth manager for merchants, generating highly specific, engaging, and category-appropriate WhatsApp campaigns.

## Architecture

The project is built around **FastAPI** for low-latency routing and idempotency handling, with a modular **Pydantic** schema layer guaranteeing strict JSON compliance with the judge's API contract.

- `main.py`: Handles API routing, health checks, context ingestion, and safe exception fallbacks to ensure the judge never crashes.
- `store.py`: A fast, thread-safe, in-memory context store. It handles version-based idempotency to ignore duplicate pushes and updates state globally.
- `engine.py`: The "brain" of the operation. Contains meticulously crafted system prompts that target the 5 grading dimensions: Specificity, Category Fit, Merchant Fit, Decision Quality, and Engagement Compulsion.
- `models.py`: An abstraction layer for LLM generation. It forces `gpt-4o` to output validated JSON arrays that exactly match the expected schema by passing strict `response_format` boundaries.

## Model Choice: GPT-4o
We initially experimented with Gemini 2.5 Pro, but shifted to **OpenAI GPT-4o** due to rate limit constraints on free tiers and its superior ability to perfectly adhere to aggressive, highly-constrained system instructions.

By completely separating the `system` constraints from the `user` context payload in the API call, GPT-4o acts entirely deterministically, eliminating hallucinations and ensuring perfect extraction of numerical data and offer names.

## Running Locally

1. `python -m venv venv`
2. `venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. Set `OPENAI_API_KEY` in your `.env` file.
5. Run the server: `python main.py`
6. Run the judge: `python judge_simulator.py`

## Deployment
This FastAPI app is production-ready. It can be instantly deployed to Render or Railway using the included `requirements.txt` and setting the startup command to:
`uvicorn main:app --host 0.0.0.0 --port $PORT`
