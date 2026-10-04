"""Criteria Manager — turns a free-text comment into a new versioned criteria row.

Strategy: LLM merges the user's comment with the currently active criteria,
producing an updated `structured_rules` JSON and `nl_addendum`. We never lose
prior versions — each update appends.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from assistant.llm import get_router
from assistant.memory.criteria_store import append_version, get_active
from assistant.state import GraphState
from assistant.storage.index_lock import resource_lock

_SYSTEM = """You manage curation criteria for a single-user research paper assistant.
You will be given (1) the currently active criteria for a domain and (2) a new
user comment. Produce an UPDATED criteria object that incorporates the comment.

Respond as JSON with this exact shape:
{
  "structured_rules": {
    "include_keywords": [...],
    "exclude_keywords": [...],
    "must_have": [...],
    "min_year": <int or null>
  },
  "nl_addendum": "<short natural-language guidance reflecting the user's intent>"
}

Rules:
- Preserve existing rules unless the comment contradicts them.
- Keep nl_addendum under 4 sentences.
- include_keywords / exclude_keywords should be concrete (e.g. "diffusion", "survey").
"""


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Could not parse LLM output as JSON: {raw[:200]}")


def criteria_node(state: GraphState) -> GraphState:
    domain = state.get("domain")
    comment = state.get("user_comment", "").strip()
    if not domain or not comment:
        return {"new_criteria_version": None}

    with resource_lock(f"criteria:{domain}"):
        return _update_criteria(domain, comment)


def _update_criteria(domain: str, comment: str) -> GraphState:

    current = get_active(domain)
    current_blob = {
        "structured_rules": current.structured_rules if current else {},
        "nl_addendum": current.nl_addendum if current else "",
    }

    llm = get_router().chat("criteria")
    msg = llm.invoke(
        [
            SystemMessage(content=_SYSTEM),
            HumanMessage(
                content=(
                    f"Domain: {domain}\n\n"
                    f"Current criteria:\n{json.dumps(current_blob, indent=2)}\n\n"
                    f"User comment:\n{comment}"
                )
            ),
        ]
    )
    parsed = _parse_json(getattr(msg, "content", str(msg)))
    version = append_version(
        domain_name=domain,
        structured_rules=parsed.get("structured_rules", {}),
        nl_addendum=str(parsed.get("nl_addendum", "")).strip(),
        source_comment=comment,
    )
    return {"new_criteria_version": version}
