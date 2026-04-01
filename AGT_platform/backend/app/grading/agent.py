import json

from .prompts import SYSTEM, PLANNER, GRADER
from .llm_router import ChatClient


def plan(client: ChatClient, modality: str) -> dict:
    return client.chat_json(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{PLANNER}\nModality: {modality}"},
        ]
    )


def grade(
    client: ChatClient, rubric: list, assignment_prompt: str, submission_context: dict
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
        ]
    )
