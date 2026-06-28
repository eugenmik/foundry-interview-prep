"""LLM-as-a-judge: score a candidate's interview answer (Hard optional task).

Given the question, the candidate's answer and the resume context, an LLM
rubric-scores the answer and returns structured feedback. Temperature is kept
low for consistent grading.
"""

from __future__ import annotations

from typing import Any

from . import openrouter_client as orc
from . import schemas
from .i18n import language_name
from .prompts import DOMAIN_ANCHOR


def judge_answer(
    question: str,
    answer: str,
    resume_context: str,
    lang: str,
    model: str = "openai/gpt-5-mini",
) -> dict[str, Any]:
    """Score one answer and return structured feedback.

    Returns a dict: {score (0-10), verdict, strengths[], improvements[], model_answer}.
    """
    system = (
        DOMAIN_ANCHOR + "\n\n"
        "You are a strict but fair interview grader. Score the candidate's "
        "answer against what a strong foundry professional would say. "
        "Return ONLY JSON:\n"
        "{\n"
        '  "score": number,            // 0-10\n'
        '  "verdict": string,          // one-line overall judgement\n'
        '  "strengths": string[],\n'
        '  "improvements": string[],\n'
        '  "model_answer": string      // a concise strong answer\n'
        "}\n"
        f"Write all text in {language_name(lang)}."
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"CANDIDATE ANSWER:\n{answer}\n\n"
        f"RESUME CONTEXT (for relevance):\n{resume_context[:2000]}"
    )
    result = orc.chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
        temperature=0.1,      # low temperature -> consistent grading
        top_p=0.9,
        max_tokens=700,
        json_mode=True,
    )
    data = schemas.parse_json(result.text)
    return {
        "score": data.get("score", "—"),
        "verdict": str(data.get("verdict", "")),
        "strengths": schemas._as_list(data.get("strengths")),
        "improvements": schemas._as_list(data.get("improvements")),
        "model_answer": str(data.get("model_answer", "")),
        "_usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "model": result.model,
        },
    }
