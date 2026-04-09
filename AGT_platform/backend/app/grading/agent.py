import json
from typing import Any

from .llm_router import ChatClient
from .prompts import (
    CONSISTENCY_CHECKER,
    CRITERION_SCORER,
    EVIDENCE_EXTRACTOR,
    GRADER,
    PLANNER,
    SYSTEM,
)


def plan(client: ChatClient, modality: str) -> dict:
    return client.chat_json(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{PLANNER}\nModality: {modality}"},
        ]
    )


def grade(
    client: ChatClient,
    rubric: list,
    assignment_prompt: str,
    submission_context: dict,
    *,
    temperature: float | None = None,
) -> dict:
    payload = {
        "rubric": rubric,
        "assignment_prompt": assignment_prompt,
        "submission": submission_context,
    }
    return client.chat_json(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": GRADER + "\n" + json.dumps(payload)},
        ],
        temperature=temperature,
    )


def extract_evidence(
    client: ChatClient,
    normalized_json: str,
    assignment_prompt: str,
    rubric_outline: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "normalized_submission": normalized_json,
        "assignment_prompt": assignment_prompt,
        "rubric_outline": rubric_outline,
    }
    return client.chat_json(
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": EVIDENCE_EXTRACTOR + "\n" + json.dumps(payload, default=str),
            },
        ]
    )


def score_criterion(
    client: ChatClient,
    criterion_spec: dict[str, Any],
    evidence_slice: dict[str, Any],
    assignment_prompt: str,
) -> dict[str, Any]:
    payload = {
        "criterion": criterion_spec,
        "evidence_slice": evidence_slice,
        "assignment_prompt": assignment_prompt,
    }
    return client.chat_json(
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": CRITERION_SCORER + "\n" + json.dumps(payload, default=str),
            },
        ]
    )


def check_consistency(
    client: ChatClient,
    criteria_list: list[dict[str, Any]],
    evidence_bundle: dict[str, Any],
    rule_issues: list[str],
) -> dict[str, Any]:
    payload = {
        "criteria": criteria_list,
        "evidence_bundle": evidence_bundle,
        "deterministic_rule_issues": rule_issues,
    }
    return client.chat_json(
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": CONSISTENCY_CHECKER
                + "\n"
                + json.dumps(payload, default=str),
            },
        ]
    )
