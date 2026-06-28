# 🔥 Foundry Interview Prep

A resume-driven **interview practice app for the metal casting / foundry industry**,
built with **Streamlit** and the **OpenRouter API**. Upload a candidate's resume and
the app analyses it, generates tailored interview questions, and runs a **live mock
interview** with an **LLM-as-a-judge** that scores answers.

---

## ✨ Features

Two modes (toggle in the sidebar):

- **Candidate (practice)** — upload your own resume, get tailored questions, and
  practise in a live chat mock interview. Ask the judge to score your answers.
- **Recruiter (prepare an interview)** — upload a candidate's resume and get a
  structured interview plan, a weighted scorecard, and red flags to watch for.

UI available in **English (default), German and Russian** — the chosen language is
also passed to the model, so all generated content matches.

### How the assignment requirements are met

| Requirement | Where |
|---|---|
| Streamlit front-end | `app.py` |
| OpenRouter API call with correct params | `src/openrouter_client.py` |
| **≥5 system prompts, different techniques** | `src/prompts.py` → `TECHNIQUES` |
| Tune ≥1 model setting | sidebar sliders: temperature, top-p, frequency/presence penalty, max tokens |
| ≥1 security guard | `src/security.py` (3 guards) |

### Prompt engineering techniques (selectable in *Developer settings*)

1. **Zero-shot** — direct instruction, no examples.
2. **Few-shot** — worked foundry Q&A examples guide style and depth.
3. **Chain-of-Thought** — reason about the resume step by step before writing questions.
4. **Role / persona** — a veteran chief metallurgist conducts the interview.
5. **Structured contract** — strict output spec with balanced category coverage.

Switch between them live to compare which works best for your resume.

### Security guards (`src/security.py`)

1. **Input validation** — length, emptiness and binary-content checks.
2. **Prompt-injection / jailbreak detection** — regex screen over the untrusted
   resume and chat text ("ignore previous instructions", "reveal your system prompt", …).
3. **LLM moderation** (optional) — a low-cost classifier for abusive / manipulative
   content. Fails open so a network blip never blocks a real user.

### Optional tasks implemented (for bonus points)

- **Hard — full chatbot** mock interview (multi-turn chat, not a one-shot call).
- **Hard — LLM-as-a-judge** rubric scoring of answers (`src/judge.py`).
- **Hard — vector database**: a persistent embedding store flags already-seen
  questions so the model is nudged to diversify (`src/rag.py` → `SeenStore`).
- **Medium — RAG**: resume is chunked and embedded; the most relevant chunks are
  retrieved to focus question generation (`src/rag.py` → `retrieve_context`).
- **Medium — ≥2 structured JSON outputs**: resume analysis + question list (+ recruiter guide).
- **Medium — choice of LLMs**: GPT-5 mini / nano / full in the sidebar.
- **Medium — prompt cost**: live pricing from the OpenRouter `/models` endpoint.
- **Medium — dev/user separation**: model & prompt controls live in a collapsed
  *Developer settings* panel, hidden from the end user by default.
- **Easy — difficulty levels**, **interviewer personas** (friendly/neutral/strict).

---

## 🚀 Run locally

```bash
git clone <this-repo>
cd interview_app

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then edit .env and add your key
streamlit run app.py
```

### Configuration

Create a `.env` file (see `.env.example`):

```
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_APP_TITLE=Foundry Interview Prep   # optional
OPENROUTER_APP_URL=https://github.com/...      # optional
```

Get a key at <https://openrouter.ai/keys>. The `.env` file is git-ignored.

---

## 🧱 Project structure

```
interview_app/
├── app.py                    # Streamlit UI: modes, tabs, developer settings
├── src/
│   ├── i18n.py               # EN / DE / RU strings
│   ├── openrouter_client.py  # chat, embeddings, live pricing
│   ├── prompts.py            # 5 prompt techniques + task message builders
│   ├── security.py           # 3 security guards
│   ├── resume_parser.py      # PDF / DOCX / TXT text extraction
│   ├── schemas.py            # defensive JSON parsing + normalisation
│   ├── rag.py                # embedding retrieval + persistent SeenStore
│   └── judge.py              # LLM-as-a-judge scoring
├── requirements.txt
├── .env.example
└── .streamlit/config.toml
```

## ⚙️ Models

- Chat: `openai/gpt-5-mini` (default), `openai/gpt-5-nano`, `openai/gpt-5`
- Embeddings: `qwen/qwen3-embedding-8b`

## 🔎 Known limitations / future work

- Pricing depends on the OpenRouter `/models` endpoint being reachable.
- The injection guard is heuristic; a determined attacker may evade regex —
  the optional LLM moderation layer mitigates this.
- The `SeenStore` is a single-file JSON store, fine for a demo but not concurrent use.
- Resume parsing relies on text-based PDFs; scanned/image PDFs need OCR (not included).

---

*Built with Streamlit + OpenRouter.*
