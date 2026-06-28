# Foundry Interview Prep

A resume-driven **interview practice application for the metal casting / foundry
industry**, built with **Streamlit** and the **OpenRouter API**. Upload a
candidate's resume and the application analyses it, generates tailored interview
questions, and runs a **live mock interview** with an **LLM-as-a-judge** that
scores answers.

**Author:** Eugen Miknevic — <https://github.com/eugenmik/foundry-interview-prep>

---

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [How the assignment requirements are met](#how-the-assignment-requirements-are-met)
- [Prompt engineering techniques](#prompt-engineering-techniques)
- [Security guards](#security-guards)
- [Optional tasks implemented](#optional-tasks-implemented-bonus)
- [Project structure](#project-structure)
- [Running locally](#running-locally)
- [Configuration](#configuration)
- [Deployment (Streamlit Community Cloud)](#deployment-streamlit-community-cloud)
- [Usage guide](#usage-guide)
- [Models](#models)
- [Design notes](#design-notes)
- [Known limitations and future work](#known-limitations-and-future-work)
- [License](#license)

---

## Overview

Most interview-prep tools ask you to pick a generic role and hand back generic
questions. This application starts from the **candidate's actual resume**: it
extracts the real foundry experience, estimates seniority, finds gaps, and builds
an interview that probes both strengths and weaknesses — all anchored to the
metal casting domain (sand / investment / die casting, alloys, melting and
metallurgy, gating and risering, casting defects, quality systems and foundry
safety).

It supports **two modes**:

- **Candidate (practice)** — upload your own resume, get tailored questions, and
  practise in a live chat mock interview. Ask the built-in judge to score your
  answers and suggest improvements.
- **Recruiter (prepare an interview)** — upload a candidate's resume and get a
  structured interview plan, a weighted scorecard and a list of red flags.

The interface is available in **English (default), German and Russian**. The
chosen language is also passed to the model, so all generated content matches.

---

## Features

- Resume ingestion from **PDF, DOCX or TXT**, or pasted text.
- **Structured resume analysis**: summary, estimated seniority, years of
  experience, key skills, strengths, gaps, likely topics and a study plan.
- **Tailored question generation** across Technical, Metallurgy, Quality/Safety
  and Behavioural categories, each with a "what a good answer covers" note.
- **Live multi-turn mock interview** (chatbot) with selectable interviewer
  persona (friendly / neutral / strict).
- **LLM-as-a-judge** scoring of answers (0–10) with strengths, improvements and
  a model answer.
- **Recruiter scorecard** and interview plan in recruiter mode.
- **Five switchable prompt-engineering techniques** for comparison.
- Full **model tuning**: model choice, temperature, top-p, frequency/presence
  penalty, max tokens, reasoning effort.
- **Three layered security guards**.
- **Light / dark theme** toggle and inline help on every control.
- **Live cost tracking** (token usage and estimated USD per session).
- **Downloadable JSON report** of the whole session.

---

## How the assignment requirements are met

| Requirement | Where |
|---|---|
| Streamlit front-end | `app.py` |
| OpenRouter API call with correct parameters | `src/openrouter_client.py` |
| At least 5 system prompts, different techniques | `src/prompts.py` → `TECHNIQUES` |
| Tune at least one model setting | sidebar sliders: temperature, top-p, frequency/presence penalty, max tokens, reasoning effort |
| At least one security guard | `src/security.py` (three guards) |

---

## Prompt engineering techniques

Selectable in the *Developer settings* panel; switch between them to compare
output quality on the same resume.

1. **Zero-shot** — direct instruction, no examples. Fast, cheap baseline.
2. **Few-shot** — worked foundry Q&A examples guide the style and depth.
3. **Chain-of-Thought** — the model reasons about the resume step by step before
   writing questions; improves relevance on messy resumes.
4. **Role / persona** — a veteran chief metallurgist conducts the interview;
   produces sharper, more realistic questions.
5. **Structured contract** — strict output specification with balanced coverage
   across question categories.

The app also distinguishes the three message roles correctly: a **system** prompt
sets the interviewer behaviour, **user** messages carry the resume and the
candidate's answers, and **assistant** messages carry the model's questions and
feedback in the multi-turn chat.

---

## Security guards

Located in `src/security.py` and applied to all untrusted input (the resume and
chat messages are inserted into prompts, so they are an injection surface):

1. **Input validation** — emptiness, minimum/maximum length and binary-content
   checks.
2. **Prompt-injection / jailbreak detection** — a regex screen for phrases such
   as "ignore previous instructions" or "reveal your system prompt".
3. **LLM moderation** (optional) — a low-cost classifier for abusive or
   manipulative content. It fails open, so a network blip never blocks a
   legitimate user.

---

## Optional tasks implemented (bonus)

**Hard**

- **Full chatbot** mock interview — a multi-turn conversation, not a one-shot call.
- **LLM-as-a-judge** — rubric scoring of answers (`src/judge.py`).
- **Vector database** — a persistent embedding store (`SeenStore` in `src/rag.py`)
  flags already-seen questions so the model is nudged to diversify.

**Medium**

- **RAG** — the resume is chunked and embedded; the most relevant chunks are
  retrieved to focus question generation (`retrieve_context` in `src/rag.py`).
- **Two or more structured JSON outputs** — resume analysis and the question list
  (plus the recruiter guide).
- **Choice of LLMs** — GPT-5 mini / nano / full in the sidebar.
- **Prompt cost** — live pricing pulled from the OpenRouter `/models` endpoint.
- **Developer/user separation** — model and prompt controls live in a collapsed
  *Developer settings* panel, hidden from the end user by default.

**Easy**

- Difficulty levels (easy / medium / hard) and interviewer personas
  (friendly / neutral / strict).
- Concise vs detailed control via the prompt technique and token budget.

---

## Project structure

```
interview_app/
├── app.py                    # Streamlit UI: modes, theme, developer settings
├── src/
│   ├── i18n.py               # EN / DE / RU strings and help texts
│   ├── openrouter_client.py  # chat, chat_json (retry), embeddings, live pricing
│   ├── prompts.py            # 5 prompt techniques + task message builders
│   ├── security.py           # three security guards
│   ├── resume_parser.py      # PDF / DOCX / TXT text extraction
│   ├── schemas.py            # defensive JSON parsing + normalisation
│   ├── rag.py                # embedding retrieval + persistent SeenStore
│   └── judge.py              # LLM-as-a-judge scoring
├── requirements.txt
├── .env.example
├── .streamlit/config.toml
└── README.md
```

---

## Running locally

```bash
git clone https://github.com/eugenmik/foundry-interview-prep.git
cd foundry-interview-prep

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then edit .env and add your key
streamlit run app.py
```

The app opens at <http://localhost:8501>.

---

## Configuration

Create a `.env` file (see `.env.example`):

```
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_APP_TITLE=Foundry Interview Prep   # optional, for OpenRouter attribution
OPENROUTER_APP_URL=https://github.com/eugenmik/foundry-interview-prep  # optional
```

Get a key at <https://openrouter.ai/keys>. The `.env` file is git-ignored and is
never committed. The app also reads the same keys from `st.secrets`, so it works
unchanged on Streamlit Community Cloud (see below).

---

## Deployment (Streamlit Community Cloud)

Yes — this repository can be deployed for public testing for free via
**Streamlit Community Cloud**, which connects directly to GitHub.

1. Push this repository to GitHub (already done).
2. Go to <https://share.streamlit.io> and sign in with your GitHub account.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Select:
   - **Repository:** `eugenmik/foundry-interview-prep`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Open **Advanced settings → Secrets** and paste:
   ```toml
   OPENROUTER_API_KEY = "sk-or-v1-..."
   OPENROUTER_APP_TITLE = "Foundry Interview Prep"
   OPENROUTER_APP_URL = "https://github.com/eugenmik/foundry-interview-prep"
   ```
6. Click **Deploy**. After the build you get a public URL like
   `https://foundry-interview-prep.streamlit.app`.

The application automatically bridges `st.secrets` into the environment, so no
code change is needed between local and cloud runs. Every push to `main`
redeploys the app automatically.

> Note: the API key is billed to your OpenRouter account, so a public deployment
> is best kept for demos/reviews. The built-in security guards reduce misuse.

Alternative hosts (Hugging Face Spaces with the Streamlit SDK, Render, a Docker
container) also work; only the secret-management step differs.

---

## Usage guide

1. Pick the **language**, **mode** and **theme** in the sidebar.
2. Upload a resume (PDF/DOCX/TXT) or paste the text.
3. Optionally set the **target role**, **seniority** and **question difficulty**.
4. Click **Analyze resume & build interview**.
5. Review the **analysis** and the **tailored questions**.
6. **Candidate mode:** answer the interviewer in the chat; click
   **Evaluate my last answer** for an AI score and feedback.
   **Recruiter mode:** review the interview plan, scorecard and red flags.
7. Download the full session as a JSON report at the bottom.

Developer/reviewer controls (model, prompt technique, sampling parameters,
reasoning effort, RAG, moderation, cost display) live in the collapsed
**Developer settings** panel.

---

## Models

- **Chat:** `openai/gpt-5-mini` (default), `openai/gpt-5-nano` (cheaper),
  `openai/gpt-5` (highest quality).
- **Embeddings:** `qwen/qwen3-embedding-8b` (RAG and the vector store).

`gpt-5` models spend hidden *reasoning tokens* that count toward `max_tokens`.
The app defaults to low reasoning effort and generous token budgets, and the
`chat_json` helper detects truncation and retries malformed JSON, so structured
outputs stay reliable.

---

## Design notes

- **Provider abstraction:** all network access goes through
  `src/openrouter_client.py`, keeping the rest of the code provider-agnostic.
- **Defensive JSON parsing:** `src/schemas.py` strips code fences and locates the
  JSON object even when the model adds prose; `chat_json` retries on failure.
- **Internationalisation:** every user-facing string is keyed in `src/i18n.py`,
  with English fallback.

---

## Known limitations and future work

- Live pricing depends on the OpenRouter `/models` endpoint being reachable.
- The injection guard is heuristic; a determined attacker may evade regex — the
  optional LLM moderation layer mitigates this.
- `SeenStore` is a single-file JSON store, fine for a demo but not for concurrent
  multi-user use; a managed vector database would be the next step.
- Resume parsing relies on text-based PDFs; scanned/image PDFs would need OCR.
- Possible extensions: image generation of a defective casting for visual
  questions, an LLM-as-a-judge comparison of the prompt techniques themselves,
  and a jailbreak test report.

---

## License

© 2026 Eugen Miknevic. All rights reserved.
