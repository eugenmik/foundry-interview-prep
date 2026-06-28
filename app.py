"""Foundry Interview Prep — a resume-driven interview practice app.

A Streamlit app for the metal casting / foundry industry. It analyses a
candidate resume, generates tailored interview questions, and runs a live
mock interview, with an LLM-as-a-judge to score answers. Two modes:
candidate (practice) and recruiter (prepare an interview). UI in EN / DE / RU.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json

import streamlit as st
from dotenv import load_dotenv

from src import judge as judge_mod
from src import openrouter_client as orc
from src import prompts, rag, schemas, security
from src.i18n import LANGUAGES, t

load_dotenv()

st.set_page_config(page_title="Foundry Interview Prep", page_icon="🔥", layout="wide")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "lang": "en",
        "resume_text": "",
        "analysis": None,
        "questions": None,
        "recruiter_guide": None,
        "chat": [],
        "interview_system": "",
        "judge_result": None,
        "total_cost": 0.0,
        "last_usage": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


def track_cost(result: orc.ChatResult) -> None:
    """Accumulate estimated USD cost and remember last token usage."""
    cost = orc.estimate_cost(result.model, result.prompt_tokens, result.completion_tokens)
    if cost is not None:
        st.session_state.total_cost += cost
    st.session_state.last_usage = {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost": cost,
    }


# ---------------------------------------------------------------------------
# Sidebar: language, mode, developer settings
# ---------------------------------------------------------------------------
with st.sidebar:
    lang = st.selectbox(
        "🌐 " + t(st.session_state.lang, "language"),
        options=list(LANGUAGES.keys()),
        format_func=lambda c: LANGUAGES[c],
        index=list(LANGUAGES.keys()).index(st.session_state.lang),
    )
    st.session_state.lang = lang

    mode = st.radio(
        t(lang, "mode"),
        options=["candidate", "recruiter"],
        format_func=lambda m: t(lang, f"mode_{m}"),
    )

    st.divider()
    with st.expander(t(lang, "developer_settings"), expanded=False):
        st.caption(t(lang, "dev_hint"))
        model_label = st.selectbox(t(lang, "model"), list(orc.CHAT_MODELS.keys()))
        model = orc.CHAT_MODELS[model_label]

        tech_labels = prompts.technique_labels()
        tech_label = st.selectbox(
            t(lang, "prompt_technique"),
            list(tech_labels.keys()),
            index=list(tech_labels.values()).index(prompts.DEFAULT_TECHNIQUE),
        )
        technique = tech_labels[tech_label]
        st.caption(prompts.TECHNIQUES[technique]["description"])

        temperature = st.slider(t(lang, "temperature"), 0.0, 2.0, 0.7, 0.1)
        top_p = st.slider(t(lang, "top_p"), 0.0, 1.0, 1.0, 0.05)
        frequency_penalty = st.slider(t(lang, "frequency_penalty"), -2.0, 2.0, 0.0, 0.1)
        presence_penalty = st.slider(t(lang, "presence_penalty"), -2.0, 2.0, 0.0, 0.1)
        # Minimum kept high: gpt-5 models spend hidden reasoning tokens, so a
        # low cap can leave no room for the actual (esp. JSON) answer.
        max_tokens = st.slider(t(lang, "max_tokens"), 512, 6000, 2000, 128)
        reasoning_effort = st.selectbox(
            "Reasoning effort (gpt-5)", ["low", "medium", "high"], index=0,
            help="Hidden reasoning depth. 'low' is faster and cheaper.",
        )
        n_questions = st.slider("Number of questions", 4, 12, 8, 1)
        persona = st.selectbox(
            "Interviewer persona", list(prompts.INTERVIEWER_PERSONAS.keys())
        )
        use_rag = st.checkbox(t(lang, "use_rag"), value=True)
        use_llm_guard = st.checkbox("LLM moderation guard", value=False)
        show_cost = st.checkbox(t(lang, "show_cost"), value=True)

    if st.button(t(lang, "reset")):
        for key in ("analysis", "questions", "recruiter_guide", "chat",
                    "interview_system", "judge_result", "total_cost", "last_usage"):
            st.session_state[key] = None if key not in ("chat", "total_cost") else (
                [] if key == "chat" else 0.0)
        st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title(t(lang, "app_title"))
st.caption(t(lang, "app_caption"))

if not orc.get_api_key():
    st.error(t(lang, "no_api_key"))
    st.stop()


def run_guards(text: str, *, min_chars: int, field: str) -> bool:
    """Run security guards; show an error and return False if blocked."""
    result = security.screen(text, min_chars=min_chars, field=field)
    if result.ok and use_llm_guard:
        result = security.llm_moderation(text)
    if not result.ok:
        st.error(f"{t(lang, 'security_blocked')}: {result.reason}")
        return False
    return True


# ---------------------------------------------------------------------------
# Section 1 — resume input
# ---------------------------------------------------------------------------
st.subheader(t(lang, "resume_section"))
st.caption(t(lang, "resume_help"))

col_a, col_b = st.columns(2)
with col_a:
    uploaded = st.file_uploader(t(lang, "upload_resume"), type=["pdf", "docx", "txt", "md"])
    if uploaded is not None:
        from src import resume_parser
        try:
            st.session_state.resume_text = resume_parser.extract_text(
                uploaded.name, uploaded.getvalue()
            )
            st.success(f"✓ {uploaded.name} ({len(st.session_state.resume_text)} chars)")
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
with col_b:
    pasted = st.text_area(t(lang, "or_paste"), value="", height=160)
    if pasted.strip():
        st.session_state.resume_text = pasted

col_c, col_d, col_e = st.columns(3)
with col_c:
    role = st.text_input(t(lang, "target_role"), placeholder=t(lang, "target_role_ph"))
with col_d:
    seniority = st.selectbox(t(lang, "seniority"), ["Junior", "Mid", "Senior", "Lead"])
with col_e:
    difficulty = st.selectbox(t(lang, "difficulty"), ["easy", "medium", "hard"])

resume_text = st.session_state.resume_text


# ---------------------------------------------------------------------------
# Analyze button — runs analysis + question generation (and recruiter guide)
# ---------------------------------------------------------------------------
if st.button("🚀 " + t(lang, "analyze_btn"), type="primary"):
    if not resume_text or not resume_text.strip():
        st.warning(t(lang, "no_resume"))
    elif run_guards(resume_text, min_chars=security.MIN_RESUME_CHARS, field="resume"):
        rag_ctx = None
        if use_rag:
            rag_ctx = rag.retrieve_context(resume_text, role or "foundry role")
        try:
            # JSON output #1: resume analysis
            with st.spinner(t(lang, "analyzing")):
                data, res = orc.chat_json(
                    prompts.analysis_messages(resume_text, role, lang),
                    parser=schemas.parse_json,
                    model=model, temperature=temperature, top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    max_tokens=max_tokens, reasoning_effort=reasoning_effort,
                )
                track_cost(res)
                st.session_state.analysis = schemas.normalize_analysis(data)

            # JSON output #2: tailored questions (technique-driven)
            with st.spinner(t(lang, "generating")):
                data, res = orc.chat_json(
                    prompts.questions_messages(
                        resume_text, role, seniority, difficulty, lang,
                        technique, n_questions, rag_ctx,
                    ),
                    parser=schemas.parse_json,
                    model=model, temperature=temperature, top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    max_tokens=max_tokens, reasoning_effort=reasoning_effort,
                )
                track_cost(res)
                questions = schemas.normalize_questions(data)
                # Hard task: vector store de-duplication of prep items.
                seen_store = rag.SeenStore()
                fresh = set(seen_store.check_and_add([q["question"] for q in questions]))
                for q in questions:
                    q["is_new"] = q["question"] in fresh if fresh else True
                st.session_state.questions = questions

            # Recruiter mode: also build interviewer guide + scorecard
            if mode == "recruiter":
                with st.spinner(t(lang, "generating")):
                    data, res = orc.chat_json(
                        prompts.recruiter_guide_messages(resume_text, role, lang),
                        parser=schemas.parse_json,
                        model=model, temperature=temperature, max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort,
                    )
                    track_cost(res)
                    st.session_state.recruiter_guide = data

            # Seed the mock interview system prompt for the chat below.
            st.session_state.interview_system = prompts.interview_system_prompt(
                resume_text, role, seniority, difficulty, lang, persona, rag_ctx
            )
            st.session_state.chat = []
            st.session_state.judge_result = None
        except orc.OpenRouterError as exc:
            st.error(f"OpenRouter error: {exc}")
        except ValueError as exc:
            st.error(f"Could not parse model output: {exc}")


# ---------------------------------------------------------------------------
# Section 2 — analysis
# ---------------------------------------------------------------------------
analysis = st.session_state.analysis
if analysis:
    st.subheader(t(lang, "analysis_section"))
    st.write(analysis["summary"])
    m1, m2 = st.columns(2)
    m1.metric(t(lang, "seniority_est"), analysis["estimated_seniority"])
    m2.metric(t(lang, "experience_years"), analysis["experience_years"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{t(lang, 'skills')}**")
        st.write(", ".join(analysis["key_skills"]) or "—")
        st.markdown(f"**{t(lang, 'strengths')}**")
        for s in analysis["strengths"]:
            st.markdown(f"- {s}")
    with c2:
        st.markdown(f"**{t(lang, 'gaps')}**")
        for g in analysis["gaps"]:
            st.markdown(f"- {g}")
        st.markdown(f"**{t(lang, 'topics')}**")
        st.write(", ".join(analysis["likely_topics"]) or "—")

    st.markdown(f"**{t(lang, 'study_plan')}**")
    for step in analysis["study_plan"]:
        st.markdown(f"- {step}")


# ---------------------------------------------------------------------------
# Section 3 — questions
# ---------------------------------------------------------------------------
questions = st.session_state.questions
if questions:
    st.subheader(t(lang, "questions_section"))
    for i, q in enumerate(questions, 1):
        badge = "🆕 " if q.get("is_new") else "♻️ "
        with st.expander(f"{badge}{i}. [{q['category']}] {q['question']}"):
            st.markdown(f"*{t(lang, 'what_to_listen')}:* {q['what_to_listen_for']}")


# ---------------------------------------------------------------------------
# Recruiter guide (recruiter mode only)
# ---------------------------------------------------------------------------
guide = st.session_state.recruiter_guide
if mode == "recruiter" and guide:
    st.subheader(t(lang, "recruiter_guide"))
    for step in guide.get("interview_plan", []):
        st.markdown(f"- {step}")
    if guide.get("scorecard"):
        st.markdown(f"**{t(lang, 'scorecard')}**")
        st.dataframe(guide["scorecard"], use_container_width=True)
    if guide.get("red_flags"):
        st.markdown("**🚩**")
        for rf in guide["red_flags"]:
            st.markdown(f"- {rf}")


# ---------------------------------------------------------------------------
# Section 4 — mock interview chat (candidate mode)
# ---------------------------------------------------------------------------
if mode == "candidate" and st.session_state.interview_system:
    st.subheader(t(lang, "interview_section"))
    st.caption(t(lang, "interview_intro"))

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Kick off with the interviewer's first question.
    if not st.session_state.chat:
        try:
            with st.spinner(t(lang, "thinking")):
                res = orc.chat(
                    [{"role": "system", "content": st.session_state.interview_system},
                     {"role": "user", "content": "Begin the interview with your first question."}],
                    model=model, temperature=temperature, top_p=top_p,
                    max_tokens=max_tokens, reasoning_effort=reasoning_effort,
                )
                track_cost(res)
                st.session_state.chat.append({"role": "assistant", "content": res.text})
                st.rerun()
        except orc.OpenRouterError as exc:
            st.error(f"OpenRouter error: {exc}")

    user_msg = st.chat_input(t(lang, "your_answer"))
    if user_msg:
        if run_guards(user_msg, min_chars=1, field="message"):
            st.session_state.chat.append({"role": "user", "content": user_msg})
            try:
                with st.spinner(t(lang, "thinking")):
                    messages = [{"role": "system", "content": st.session_state.interview_system}]
                    messages += [{"role": m["role"], "content": m["content"]}
                                 for m in st.session_state.chat]
                    res = orc.chat(
                        messages, model=model, temperature=temperature, top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty, max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort,
                    )
                    track_cost(res)
                    st.session_state.chat.append({"role": "assistant", "content": res.text})
                st.rerun()
            except orc.OpenRouterError as exc:
                st.error(f"OpenRouter error: {exc}")

    # LLM-as-a-judge: evaluate the candidate's most recent answer.
    last_user = next((m["content"] for m in reversed(st.session_state.chat)
                      if m["role"] == "user"), None)
    last_question = None
    for m in st.session_state.chat:
        if m["role"] == "assistant":
            last_question = m["content"]
        elif m["role"] == "user":
            pass
    if last_user and st.button("🧑‍⚖️ " + t(lang, "evaluate_btn")):
        try:
            with st.spinner(t(lang, "judging")):
                # Find the assistant question immediately preceding the last answer.
                q_for_answer = ""
                for idx in range(len(st.session_state.chat) - 1, -1, -1):
                    if st.session_state.chat[idx]["role"] == "user":
                        if idx > 0 and st.session_state.chat[idx - 1]["role"] == "assistant":
                            q_for_answer = st.session_state.chat[idx - 1]["content"]
                        break
                st.session_state.judge_result = judge_mod.judge_answer(
                    q_for_answer, last_user, resume_text, lang, model, reasoning_effort
                )
        except (orc.OpenRouterError, ValueError) as exc:
            st.error(f"Judge error: {exc}")

    jr = st.session_state.judge_result
    if jr:
        st.markdown(f"### {t(lang, 'judge_section')}")
        st.metric(t(lang, "score"), f"{jr['score']}/10")
        st.markdown(f"**{t(lang, 'verdict')}:** {jr['verdict']}")
        jc1, jc2 = st.columns(2)
        with jc1:
            st.markdown(f"**{t(lang, 'strengths_ans')}**")
            for s in jr["strengths"]:
                st.markdown(f"- {s}")
        with jc2:
            st.markdown(f"**{t(lang, 'improve_ans')}**")
            for s in jr["improvements"]:
                st.markdown(f"- {s}")
        with st.expander(t(lang, "model_answer")):
            st.write(jr["model_answer"])


# ---------------------------------------------------------------------------
# Cost panel + report download
# ---------------------------------------------------------------------------
if show_cost and st.session_state.last_usage:
    u = st.session_state.last_usage
    st.divider()
    cc1, cc2 = st.columns(2)
    cc1.metric(
        t(lang, "tokens_label"),
        f"{u['prompt_tokens']} / {u['completion_tokens']}",
    )
    cost_txt = f"${st.session_state.total_cost:.5f}" if st.session_state.total_cost else "—"
    cc2.metric(t(lang, "cost_label") + " (session)", cost_txt)

if analysis or questions:
    report = {
        "mode": mode,
        "language": lang,
        "role": role,
        "seniority": seniority,
        "difficulty": difficulty,
        "analysis": analysis,
        "questions": questions,
        "recruiter_guide": guide,
        "chat": st.session_state.chat,
        "judge_result": st.session_state.judge_result,
    }
    st.download_button(
        "⬇️ " + t(lang, "download_report"),
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name="interview_report.json",
        mime="application/json",
    )

st.divider()
st.caption(t(lang, "footer"))
