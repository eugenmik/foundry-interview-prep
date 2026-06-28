"""Helpers for parsing and validating the structured JSON the LLM returns.

The app uses two main structured output formats (resume analysis and the
question list) plus the recruiter guide and judge verdict. The LLM is asked
for JSON, but models occasionally wrap it in code fences or add prose, so we
parse defensively.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from a possibly-noisy LLM response."""
    if not text:
        raise ValueError("Empty response.")
    candidate = text.strip()

    # 1. Strip a markdown code fence if present.
    fence = _FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()

    # 2. Try a straight parse.
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 3. Fall back to the first {...} block in the text.
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(candidate[start:end + 1])

    raise ValueError("Response did not contain valid JSON.")


def normalize_analysis(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure the resume-analysis dict has all expected keys with safe defaults."""
    return {
        "summary": str(data.get("summary", "")),
        "estimated_seniority": str(data.get("estimated_seniority", "—")),
        "experience_years": data.get("experience_years", "—"),
        "key_skills": _as_list(data.get("key_skills")),
        "strengths": _as_list(data.get("strengths")),
        "gaps": _as_list(data.get("gaps")),
        "likely_topics": _as_list(data.get("likely_topics")),
        "study_plan": _as_list(data.get("study_plan")),
    }


def normalize_questions(data: dict[str, Any]) -> list[dict[str, str]]:
    """Return a clean list of {category, question, what_to_listen_for}."""
    items = data.get("questions", []) if isinstance(data, dict) else []
    out: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append({
            "category": str(item.get("category", "General")),
            "question": str(item.get("question", "")).strip(),
            "what_to_listen_for": str(item.get("what_to_listen_for", "")).strip(),
        })
    return [q for q in out if q["question"]]


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]
