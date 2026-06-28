"""Prompt engineering library for the Foundry Interview Prep app.

This module is the heart of the project. It contains:

* FIVE interchangeable system prompts for the interview engine, each using a
  different prompting technique (zero-shot, few-shot, chain-of-thought,
  role/persona, and structured-contract). The developer can switch between
  them in the sidebar to compare which works best.
* Message builders for the three LLM tasks: resume analysis, question
  generation and the multi-turn mock interview.

All prompts are domain-anchored to the metal casting / foundry industry.
"""

from __future__ import annotations

from .i18n import language_name

# ---------------------------------------------------------------------------
# Shared domain context injected into every system prompt. Keeping the domain
# anchor in one place makes the foundry focus consistent across techniques.
# ---------------------------------------------------------------------------
DOMAIN_ANCHOR = (
    "You are an expert technical interviewer for the METAL CASTING / FOUNDRY "
    "industry. You know sand casting, investment casting (lost-wax), die "
    "casting (HPDC/LPDC), permanent-mould and centrifugal casting; alloys "
    "(grey/ductile iron, steel, aluminium, bronze); melting and metallurgy "
    "(induction/cupola furnaces, inoculation, degassing); gating and risering; "
    "casting defects (shrinkage, gas porosity, cold shut, misrun, hot tears, "
    "inclusions, sand burn-on); quality systems (IATF 16949, PPAP, NDT, "
    "spectrometry) and foundry health & safety."
)

# A small few-shot bank reused by the few-shot technique and as guidance.
FEWSHOT_EXAMPLES = """\
Example interview questions for foundry roles (use as style reference, do NOT copy verbatim):

[Technical · Process Engineer]
Q: You see recurring gas porosity in an aluminium HPDC part. Walk me through how
   you isolate the root cause across melt, die and process.
What a good answer covers: degassing/H2 level, die venting and vacuum, lubricant
   burn-off, fill speed/intensification pressure, X-ray confirmation.

[Behavioural · Melt Shop Supervisor]
Q: Tell me about a time a furnace charge went wrong on shift. What did you do?
What a good answer covers: STAR structure, safety first, containment of scrap,
   root-cause, communication with the next shift.

[Metallurgy · Quality Engineer]
Q: A ductile iron casting fails the nodularity spec. What checks do you run?
What a good answer covers: Mg fading/inoculation timing, spectro + micrograph,
   pouring temperature/time window, treatment ladle practice.
"""

# ---------------------------------------------------------------------------
# FIVE system-prompt techniques for the interview question engine.
# Each value is a template with {lang}, {role}, {seniority}, {difficulty}.
# ---------------------------------------------------------------------------
TECHNIQUES: dict[str, dict[str, str]] = {
    "zero_shot": {
        "name": "Zero-shot (direct instruction)",
        "description": (
            "Plain, direct instruction with no examples and no reasoning "
            "scaffold. Fast and cheap; good baseline."
        ),
        "template": (
            DOMAIN_ANCHOR + "\n\n"
            "Task: produce interview questions tailored to the candidate's "
            "resume for the role of {role} at {seniority} level, at {difficulty} "
            "difficulty. Cover technical, metallurgical and behavioural areas. "
            "Respond in {lang}."
        ),
    },
    "few_shot": {
        "name": "Few-shot (learning from examples)",
        "description": (
            "Provides several worked examples so the model imitates the style, "
            "depth and structure of strong foundry interview questions."
        ),
        "template": (
            DOMAIN_ANCHOR + "\n\n" + FEWSHOT_EXAMPLES + "\n"
            "Now generate NEW questions in the same style, tailored to the "
            "candidate's resume for {role} ({seniority}, {difficulty} "
            "difficulty). Respond in {lang}."
        ),
    },
    "chain_of_thought": {
        "name": "Chain-of-Thought (reason then answer)",
        "description": (
            "Asks the model to reason step by step about the resume before "
            "writing questions. Improves relevance for messy resumes."
        ),
        "template": (
            DOMAIN_ANCHOR + "\n\n"
            "Think step by step BEFORE answering: (1) identify the candidate's "
            "real foundry experience and seniority from the resume; (2) map it to "
            "the {role} role; (3) find gaps worth probing; (4) only then write "
            "questions at {seniority}/{difficulty} level that test both strengths "
            "and gaps. Keep your private reasoning brief and do not expose it in "
            "the final structured output. Respond in {lang}."
        ),
    },
    "role_persona": {
        "name": "Role / persona (expert interviewer)",
        "description": (
            "Strong persona framing — a veteran foundry chief metallurgist who "
            "interviews rigorously but fairly. Produces sharper, realistic questions."
        ),
        "template": (
            "You are Dr. Keller, a chief metallurgist with 25 years in iron and "
            "aluminium foundries who has hired dozens of engineers and operators. "
            + DOMAIN_ANCHOR + "\n\n"
            "Interview the candidate for {role} ({seniority}). Ask the kind of "
            "incisive, practical questions you would actually ask on the shop "
            "floor at {difficulty} difficulty, grounded in their resume. "
            "Respond in {lang}."
        ),
    },
    "structured_contract": {
        "name": "Structured contract (strict output spec)",
        "description": (
            "Emphasises an explicit output contract and balanced coverage across "
            "categories. Best paired with JSON output mode."
        ),
        "template": (
            DOMAIN_ANCHOR + "\n\n"
            "Generate interview questions for {role} ({seniority}, {difficulty}). "
            "Follow this contract strictly: balanced coverage across the "
            "categories Technical, Metallurgy, Quality/Safety and Behavioural; "
            "every question must be answerable from or motivated by the resume; "
            "each question gets a short 'what a good answer covers' note. "
            "Respond in {lang}."
        ),
    },
}

DEFAULT_TECHNIQUE = "few_shot"


def technique_labels() -> dict[str, str]:
    """Return {label: key} for the technique selector."""
    return {v["name"]: k for k, v in TECHNIQUES.items()}


def _ctx(role: str, seniority: str, difficulty: str) -> dict[str, str]:
    return {
        "role": role.strip() or "a foundry role matching the resume",
        "seniority": seniority,
        "difficulty": difficulty,
    }


# ---------------------------------------------------------------------------
# Task 1: resume analysis (structured JSON output #1)
# ---------------------------------------------------------------------------
def analysis_messages(resume_text: str, role: str, lang: str) -> list[dict[str, str]]:
    """Build messages that analyse the resume and return structured JSON."""
    system = (
        DOMAIN_ANCHOR + "\n\n"
        "You analyse a candidate resume for foundry/metallurgy roles. "
        "Return ONLY a JSON object with this exact shape:\n"
        "{\n"
        '  "summary": string,                       // 2-3 sentences\n'
        '  "estimated_seniority": "Junior"|"Mid"|"Senior"|"Lead",\n'
        '  "experience_years": number,\n'
        '  "key_skills": string[],                  // foundry-relevant skills\n'
        '  "strengths": string[],\n'
        '  "gaps": string[],                        // risks / things to probe\n'
        '  "likely_topics": string[],               // interview topics to cover\n'
        '  "study_plan": string[]                   // 3-6 concrete prep actions\n'
        "}\n"
        f"Write all string values in {language_name(lang)}."
    )
    user = (
        f"Target role: {role or '(infer the best-fit foundry role)'}\n\n"
        f"RESUME:\n\"\"\"\n{resume_text}\n\"\"\""
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Task 2: question generation (structured JSON output #2, technique-driven)
# ---------------------------------------------------------------------------
def questions_messages(
    resume_text: str,
    role: str,
    seniority: str,
    difficulty: str,
    lang: str,
    technique: str,
    n_questions: int = 8,
    rag_context: str | None = None,
) -> list[dict[str, str]]:
    """Build messages that generate tailored questions using the chosen technique."""
    tech = TECHNIQUES.get(technique, TECHNIQUES[DEFAULT_TECHNIQUE])
    system = tech["template"].format(lang=language_name(lang), **_ctx(role, seniority, difficulty))
    system += (
        "\n\nReturn ONLY a JSON object of the form:\n"
        '{ "questions": [ { "category": string, "question": string, '
        '"what_to_listen_for": string } ] }\n'
        f"Produce exactly {n_questions} questions."
    )
    context_block = resume_text
    if rag_context:
        context_block = (
            "Most relevant resume excerpts (retrieved):\n" + rag_context +
            "\n\nFull resume follows for reference:\n" + resume_text
        )
    user = f"CANDIDATE RESUME / CONTEXT:\n\"\"\"\n{context_block}\n\"\"\""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Task 3: multi-turn mock interview (chat)
# ---------------------------------------------------------------------------
INTERVIEWER_PERSONAS = {
    "friendly": "warm and encouraging, you put the candidate at ease",
    "neutral": "professional and neutral, you neither praise nor criticise",
    "strict": "demanding and probing, you push back and ask hard follow-ups",
}


def interview_system_prompt(
    resume_text: str,
    role: str,
    seniority: str,
    difficulty: str,
    lang: str,
    persona: str = "neutral",
    rag_context: str | None = None,
) -> str:
    """System prompt that turns the model into a live foundry interviewer."""
    persona_desc = INTERVIEWER_PERSONAS.get(persona, INTERVIEWER_PERSONAS["neutral"])
    context = rag_context or resume_text
    return (
        DOMAIN_ANCHOR + "\n\n"
        f"You are conducting a LIVE mock interview for {role or 'a foundry role'} "
        f"at {seniority} level, {difficulty} difficulty. Persona: {persona_desc}.\n"
        "Rules:\n"
        "- Ask ONE question at a time, then wait for the candidate's answer.\n"
        "- Base questions on the resume context below; probe both strengths and gaps.\n"
        "- After each answer give one short line of feedback, then ask the next question.\n"
        "- If the candidate asks for a hint, give a brief one without revealing a full answer.\n"
        "- Stay strictly on foundry/metallurgy/interview topics.\n"
        f"- Always respond in {language_name(lang)}.\n\n"
        f"RESUME CONTEXT:\n\"\"\"\n{context}\n\"\"\""
    )


# ---------------------------------------------------------------------------
# Recruiter mode: interviewer guide / scorecard (structured JSON)
# ---------------------------------------------------------------------------
def recruiter_guide_messages(resume_text: str, role: str, lang: str) -> list[dict[str, str]]:
    """Build messages producing an interviewer guide + scorecard for recruiters."""
    system = (
        DOMAIN_ANCHOR + "\n\n"
        "You help a foundry recruiter run a structured interview from a resume. "
        "Return ONLY JSON of the form:\n"
        "{\n"
        '  "interview_plan": string[],          // ordered stages of the interview\n'
        '  "scorecard": [ { "criterion": string, "what_good_looks_like": string, '
        '"weight": number } ],\n'
        '  "red_flags": string[]\n'
        "}\n"
        f"Write all text in {language_name(lang)}."
    )
    user = (
        f"Role: {role or '(infer)'}\n\nRESUME:\n\"\"\"\n{resume_text}\n\"\"\""
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
