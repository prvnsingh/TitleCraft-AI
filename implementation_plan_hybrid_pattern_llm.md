# Implementation Plan — Pattern-Profiling + LLM Generation

## Overview
This document describes a pragmatic, production-minded implementation for the Electrify take-home task given the small dataset (~211 rows across 3 channels). The recommended approach is a **hybrid pipeline** that extracts lightweight, interpretable patterns from high-performing titles per channel and uses those patterns to condition a prompt-driven LLM generator. This balances data-grounded reasoning, creativity, and robustness when training data is limited.

---

## Goals
- Learn what makes *high-performing titles* per channel using only the provided CSV.
- Generate 3–5 candidate titles for a new video idea for any channel.
- Provide *per-title, data-grounded reasoning* explaining why each title aligns with the channel’s high-performing patterns.
- Deliver as a **local FastAPI** endpoint with runnable setup.

---

## High-level architecture

# Diagram (logical)

1. **Offline Data Profiler** (batch)
   - Input: `training CSV`
   - Output: `channel_profiles.json`
   - Responsibilities: compute channel-level statistics & extract high-performer features

2. **Optional Offline Embedding Indexer** (batch, small)
   - Input: `title+summary` rows
   - Output: embeddings store (e.g., small FAISS / in-memory dict)
   - Responsibilities: enable optional retrieval of semantically similar past videos

3. **Runtime Service (FastAPI)**
   - Endpoint: `/generate_titles` (POST) — accepts `channel_id` and `idea`
   - Flow:
     1. Load channel profile
     2. Optionally retrieve top-n similar examples by embeddings
     3. Build structured LLM prompt (pattern profile + examples + idea)
     4. Call LLM (remote API or local model) to generate 3–5 titles + reasoning
     5. Post-process & return JSON

4. **Utilities**
   - Caching layer (simple file / in-memory cache) for repeated requests
   - Logging & telemetry for evaluation during demo

---

## Component details

### 1) Data Profiler (offline)
**Purpose:** extract robust, interpretable features from a small dataset and save a `channel_profiles.json` used at runtime.

**Key computations (per channel):**
- `n_videos` — total videos
- `views_stats` — min, 25%, median, mean, 75%, max
- `high_perf_threshold` — 80th percentile of `views_in_period` (tunable)
- `high_perf_examples` — top 5 titles (and their summaries) above threshold
- `avg_title_length_words` and `std` — tokenized by whitespace
- `numeric_ratio` — fraction of titles containing digits (e.g., "5", "10")
- `question_ratio` — fraction of titles that end with `?` or start with `Why/How/What`
- `punctuation_usage` — presence of `:`, `-`, `!`, `?`
- `top_ngrams` — top bigrams/trigrams from high-performers (use simple count + min_df)
- `top_keywords` — TF-IDF top keywords from high-performers
- `tone_markers` — heuristic list of frequent curiosity/emotion words (why, how, shocking, crazy, revealed)

**Output file:** `channel_profiles.json` example:
```json
{
  "UCxxxxx": {
    "n_videos": 107,
    "views_stats": {"median": 2101, "max": 336440},
    "high_perf_threshold": 5000,
    "avg_title_length_words": 9.8,
    "numeric_ratio": 0.18,
    "question_ratio": 0.12,
    "top_ngrams": ["how to", "top 5", "why does"],
    "high_perf_examples": [
      {"title": "...", "summary": "...", "views": 336440}
    ]
  }
}
```

**Implementation notes:**
- Use `pandas`, `sklearn.feature_extraction.text.TfidfVectorizer`, `collections.Counter` for ngram counts.
- Keep heuristics conservative to avoid noise from small-sample artifacts (e.g., require ngram count ≥ 2).

---

### 2) Optional Embedding Indexer (small, helpful for grounding)
**Purpose:** find semantically similar past videos to the incoming `idea` and show them to the LLM to ground suggestions.

**Approach:**
- Use a lightweight embedding model like `sentence-transformers/all-MiniLM-L6-v2` (small and fast).
- Precompute embeddings for `title + summary` per row and save in a local pickle or small FAISS index.
- At runtime, compute embedding for `idea` and retrieve top-3 similar high-performing examples (by cosine similarity).

**Why optional:**
- Useful to show the LLM real examples; but with 200 rows, gains are modest. It adds complexity but improves contextual relevance.

---

### 3) Runtime Service (FastAPI)
**Endpoint:** `POST /generate_titles`
**Request body:**
```json
{ "channel_id": "UC...", "idea": "Short idea sentence", "n_titles": 4 }
```

**Response body:**
```json
{
  "channel_id": "UC...",
  "idea": "...",
  "titles": [
    { "title": "...", "reasoning": "...", "score_components": {...} },
    ...
  ],
  "profile_used": { /* small slice of profile for transparency */ }
}
```

**Runtime steps:**
1. Validate input & load `channel_profiles.json`.
2. If channel profile missing: fallback to global profile (aggregate across channels).
3. Optionally retrieve top-k similar examples with embeddings.
4. Build the LLM prompt using a compact, structured template (see Prompt Templates section).
5. Call LLM API (e.g., OpenAI / Anthropic) or local model. Use temperature=0.7 to allow creativity but maintain reproducibility with `seed` if supported.
6. Parse and validate the LLM output (ensure JSON schema compliance). If parsing fails, re-call or attempt a simple deterministic post-processor.
7. Return results.

**Operational considerations:**
- Enforce request timeouts and circuit-breakers for the LLM API.
- Implement a simple cache keyed by `(channel_id, idea)`.

---

## Prompt templates (examples)
Keep prompts small and structured. Use explicit instruction + bullets + examples.

**Template (compact):**
```
You are a YouTube title strategist.
Channel profile:
- avg_title_length_words: {avg_title_length}
- numeric_ratio: {numeric_ratio}
- question_ratio: {question_ratio}
- top_ngrams: {top_ngrams}
Top-performing examples:
1. {ex1_title} — {ex1_views}
2. {ex2_title} — {ex2_views}

Task: Given this new video idea: "{idea}", generate {n_titles} concise, click-worthy titles for this channel.
Constraints:
- Follow the channel style (length ~ {avg_title_length} words; use numbers if numeric_ratio > 0.15; prefer questions if question_ratio > 0.2)
- For each title, provide a 1–2 sentence reasoning that references the channel profile or examples.

Output: Strict JSON array: [ {"title":"...", "reasoning":"..."}, ... ]
```

**Notes:**
- Keep prompts under token limits; include only 2–3 example titles to save tokens.
- For channels with very few videos, include aggregated/global patterns.

---

## Post-processing & Safety
- Trim excessive length (> 12 words) optionally by heuristic.
- Sanitize outputs for punctuation/emoji.
- Validate reasoning length (1–2 sentences) and remove hallucinated metrics (don’t allow LLM to invent views numbers).

---

## Evaluation & Demo Plan
- **Manual check**: for demo, use 3 unseen test ideas (provided during interview) and produce 3–5 titles each.
- **Quality criteria**:
  - Alignment with channel style (qualitative)
  - Diversity among titles (avoid paraphrases)
  - Reasoning cites channel features or examples
- **Metrics (informal)**:
  - Human rating (1–5) from your internal reviewer
  - Precision of following constraints (binary pass/fail)

---

## Folder structure (suggested)
```
project-root/
├─ data/
│  ├─ electrify_training.csv
│  └─ channel_profiles.json
├─ src/
│  ├─ profiler/
│  │  └─ build_profiles.py
│  ├─ indexer/
│  │  └─ build_embeddings.py
│  ├─ api/
│  │  ├─ main.py   # FastAPI app
│  │  └─ prompts.py
│  ├─ utils/
│  │  └─ text.py
│  └─ config.py
├─ tests/
├─ README.md
├─ requirements.txt
└─ run_local.sh
```

---

## Implementation checklist (priority-ordered)
1. **Profiler**: compute channel profiles & save JSON. (1–2 hours)
2. **FastAPI skeleton**: implement endpoint and local dev run. (30–60 mins)
3. **Prompt engineering**: craft stable prompt templates & parse outputs. (30–60 mins)
4. **LLM integration**: connect to your preferred API (or local model) with retries and timeouts. (30–60 mins)
5. **Optional embeddings**: compute & index embeddings for retrieval. (30–60 mins)
6. **Demo & caching**: prepare three example ideas and test. (30–60 mins)

---

## Notes on the small dataset and why this design works
- The dataset is too small for reliable supervised training or fine-tuning; pattern extraction avoids overfitting and yields interpretable signals.
- LLMs are good at composition: handing them a few succinct channel-specific hints yields better creative output than training on a few hundred examples.
- The hybrid system gives demonstrable, data-grounded reasoning while remaining simple to run locally.

---

## Extensions (future)
- Auto-tune `high_perf_threshold` per channel with cross-validation-like holdout.
- Learn a small reranker (lightweight classifier) from handcrafted scoring features to rank LLM candidates; train on pairwise preferences if more data becomes available.
- Add A/B testing hooks and collect real-world feedback (CTR/views) to iteratively improve.

---

## Run & demo instructions (short)
1. `pip install -r requirements.txt`
2. `python src/profiler/build_profiles.py --input data/electrify_training.csv --output data/channel_profiles.json`
3. Set `OPENAI_API_KEY` (or other LLM creds) in env.
4. `uvicorn src.api.main:app --reload --port 8000`
5. POST to `http://localhost:8000/generate_titles` with JSON payload `{ "channel_id": "UC...", "idea": "..." }`


---

## Appendix: Example prompt (final)
*(Keep this in `src/api/prompts.py` and programmatically fill fields.)*

```
You are an expert YouTube title strategist. Channel profile: avg length {avg_title_length} words; numeric_ratio {numeric_ratio}; question_ratio {question_ratio}; top_patterns {top_ngrams}. Top examples: {examples}.
Task: Given the new video idea: "{idea}", generate {n_titles} title suggestions and for each provide a 1–2 sentence reasoning grounded in the channel profile.
Output in JSON.
```

---

*End of document.*

