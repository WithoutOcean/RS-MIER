
import importlib.util
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import requests


_BASE_REWARD_PATH = Path(__file__).with_name("reward_rule.py")
_BASE_SPEC = importlib.util.spec_from_file_location("base_reward", _BASE_REWARD_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError(f"Cannot load the base reward from {_BASE_REWARD_PATH}.")

_BASE_REWARD = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE_REWARD)

REWARD_NAME = "four_part"
REWARD_TYPE = "batch"

_FORMAT_PATTERN = re.compile(
    r"^\s*"
    r"<perception>\s*\S(?:.*?)</perception>\s*"
    r"<think>\s*\S(?:.*?)</think>\s*"
    r"<answer>\s*\S(?:.*?)</answer>"
    r"\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)
_QUESTION_PATTERN = re.compile(
    r"\bQuestion\s*:\s*(.*?)"
    r"(?=\n\s*(?:First\b|Then\b|Finally\b|Your response\b|Analyze\b)|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)

_JUDGE_SYSTEM_PROMPT = """You are a strict reasoning judge for remote-sensing change questions.
You receive one question and an untrusted visual-evidence description written by another model. Infer the answer to the question using ONLY that evidence description. You do not see the images, the other model's reasoning, its final answer, or the reference answer.
Briefly show how the answer follows from the supplied visual evidence, then give the derived answer. Return exactly this structure and no outside text:
<think>reasoning from the supplied visual evidence</think>
<answer>derived answer</answer>"""


def _normalize_tags(response: str) -> str:
   
    return re.sub(
        r"<\s*(/?)\s*(perception|think|answer)\s*>",
        lambda match: f"<{match.group(1)}{match.group(2).lower()}>",
        str(response),
        flags=re.IGNORECASE,
    )


def _extract_section(response: str, tag: str) -> Optional[str]:
    match = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        _normalize_tags(response),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None

    content = match.group(1).strip()
    return content if content else None


def extract_question(prompt: str) -> str:
    """Extract the question without forwarding answer-format instructions."""
    match = _QUESTION_PATTERN.search(str(prompt))
    return match.group(1).strip() if match is not None else ""


def format_reward(response: str) -> float:
    """Return 0 for exactly three valid non-empty sections, otherwise -1."""
    normalized = _normalize_tags(response)
    for tag in ("perception", "think", "answer"):
        if normalized.lower().count(f"<{tag}>") != 1 or normalized.lower().count(f"</{tag}>") != 1:
            return -1.0

    return 0.0 if _FORMAT_PATTERN.fullmatch(normalized) is not None else -1.0


def _literal_answer_reward(response: str, ground_truth: str) -> float:
    """Compare extracted answer text using only case and outer-whitespace normalization."""
    answer = _BASE_REWARD.extract_answer(_normalize_tags(response))
    predicted = str(answer).strip().casefold()
    target = str(ground_truth).strip().casefold()
    return 1.0 if predicted and predicted == target else 0.0


def final_answer_reward(response: str, ground_truth: str) -> float:
    """Grade the student's final answer with conservative literal comparison."""
    return _literal_answer_reward(response, ground_truth)


def _judge_request(
    *,
    question: str,
    perception: str,
    judge_url: str,
    judge_model: str,
    judge_api_key: str,
    judge_timeout: float,
    judge_retries: int,
) -> str:
    headers = {"Content-Type": "application/json"}
    if judge_api_key:
        headers["Authorization"] = f"Bearer {judge_api_key}"

    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nVisual evidence only:\n<perception>{perception}</perception>",
            },
        ],
        "temperature": 0.0,
        "max_tokens": 128,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    last_error: Optional[Exception] = None
    for attempt in range(judge_retries + 1):
        try:
            response = requests.post(judge_url, headers=headers, json=payload, timeout=judge_timeout)
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("judge returned an empty completion")
            return content.strip()
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt < judge_retries:
                time.sleep(min(0.25 * (2**attempt), 1.0))

    raise RuntimeError(
        f"Qwen3-32B judge request failed after {judge_retries + 1} attempt(s): {last_error}"
    ) from last_error


def _judge_batch(
    items: list[tuple[str, str]],
    *,
    judge_urls: list[str],
    judge_model: str,
    judge_api_key: str,
    judge_timeout: float,
    judge_retries: int,
    judge_max_workers: int,
) -> list[str]:
    """Submit independent requests concurrently so vLLM can batch them."""
    if not items:
        return []

    outputs = [""] * len(items)
    max_workers = min(max(1, judge_max_workers), len(items))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _judge_request,
                question=question,
                perception=perception,
              
                judge_url=judge_urls[index % len(judge_urls)],
                judge_model=judge_model,
                judge_api_key=judge_api_key,
                judge_timeout=judge_timeout,
                judge_retries=judge_retries,
            ): index
            for index, (question, perception) in enumerate(items)
        }
        for future in as_completed(futures):
            outputs[futures[future]] = future.result()

    return outputs


def _resolve_judge_urls(judge_urls: Any, judge_url: Optional[str]) -> list[str]:
    """Resolve one or more judge endpoints while retaining single-URL compatibility."""
    configured = judge_urls
    if configured is None and judge_url:
        configured = judge_url
    if configured is None:
        configured = os.getenv("JUDGE_URLS")
    if configured is None:
        configured = os.getenv(
            "JUDGE_URL",
            "http://127.0.0.1:8001/v1/chat/completions",
        )

    if isinstance(configured, str):
        urls = [part.strip() for part in configured.split(",") if part.strip()]
    elif isinstance(configured, (list, tuple)):
        urls = [str(part).strip() for part in configured if str(part).strip()]
    else:
        raise TypeError(f"judge_urls must be a string, list, or tuple, got {type(configured).__name__}.")

    if not urls:
        raise ValueError("At least one Qwen3-32B judge URL is required.")
    if any(not re.match(r"^https?://", url, flags=re.IGNORECASE) for url in urls):
        raise ValueError(f"Every judge URL must start with http:// or https://, got {urls}.")
    return urls


def perception_reward(judge_answer: str, ground_truth: str) -> float:
    """Grade the judge's answer with the same conservative literal comparison."""
    return _literal_answer_reward(judge_answer, ground_truth)


def reasoning_reward(
    perception_score: float,
    final_answer_score: float,
    think: Optional[str],
    perception_evaluated: bool = True,
) -> float:
    """Reward matching correctness states: both correct or both incorrect."""
    if not perception_evaluated or not think:
        return 0.0

    perception_correct = perception_score > 0.0
    final_answer_correct = final_answer_score > 0.0
    return 1.0 if perception_correct == final_answer_correct else 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
    perception_weight: float = 0.4,
    reasoning_weight: float = 0.0,
    answer_weight: float = 0.3,
    judge_urls: Any = None,
    judge_url: Optional[str] = None,
    judge_model: Optional[str] = None,
    judge_api_key: Optional[str] = None,
    judge_timeout: float = 120.0,
    judge_retries: int = 2,
    judge_max_workers: int = 128,
) -> list[dict[str, float]]:
   
    weights = (format_weight, perception_weight, reasoning_weight, answer_weight)
    weight_sum = sum(weights)
    if any(weight < 0.0 for weight in weights) or weight_sum <= 0.0:
        raise ValueError(f"Reward weights must be non-negative with a positive sum, got {weights}.")
    if judge_timeout <= 0.0:
        raise ValueError(f"judge_timeout must be positive, got {judge_timeout}.")
    if judge_retries < 0:
        raise ValueError(f"judge_retries must be non-negative, got {judge_retries}.")
    if judge_max_workers <= 0:
        raise ValueError(f"judge_max_workers must be positive, got {judge_max_workers}.")

    resolved_judge_urls = _resolve_judge_urls(judge_urls, judge_url)
    judge_model = judge_model or os.getenv("JUDGE_MODEL", "qwen3-32b-judge")
    judge_api_key = judge_api_key if judge_api_key is not None else os.getenv("JUDGE_API_KEY", "")

    prepared: list[dict[str, Any]] = []
    judge_items: list[tuple[str, str]] = []
    judge_indices: list[int] = []

    for index, reward_input in enumerate(reward_inputs):
        response = _normalize_tags(str(reward_input["response"]))
        ground_truth = str(reward_input["ground_truth"])
        perception = _extract_section(response, "perception")
        think = _extract_section(response, "think")
        question = extract_question(str(reward_input.get("prompt", "")))

        prepared.append(
            {
                "response": response,
                "ground_truth": ground_truth,
                "perception": perception,
                "think": think,
                "question": question,
            }
        )

        if perception is not None:
            if not question:
                raise ValueError(
                    "The four-part reward requires the original prompt so the judge can identify the question."
                )
            judge_indices.append(index)
            judge_items.append((question, perception))

    judge_outputs = _judge_batch(
        judge_items,
        judge_urls=resolved_judge_urls,
        judge_model=judge_model,
        judge_api_key=judge_api_key,
        judge_timeout=judge_timeout,
        judge_retries=judge_retries,
        judge_max_workers=judge_max_workers,
    )
    judge_output_by_index = dict(zip(judge_indices, judge_outputs))

    scores = []
    use_reasoning = reasoning_weight > 0.0
    for index, item in enumerate(prepared):
        format_score = format_reward(item["response"])
        answer_score = final_answer_reward(item["response"], item["ground_truth"])
        perception_score = 0.0
        if index in judge_output_by_index:
            perception_score = perception_reward(judge_output_by_index[index], item["ground_truth"])
        weighted_score = (
            format_weight * format_score
            + perception_weight * perception_score
            + answer_weight * answer_score
        )
        score = {
            "format": format_score,
            "perception": perception_score,
            "final_answer": answer_score,
            
            "accuracy": answer_score,
        }
        if use_reasoning:
            reasoning_score = reasoning_reward(
                perception_score,
                answer_score,
                item["think"],
                perception_evaluated=index in judge_output_by_index,
            )
            score["reasoning"] = reasoning_score
            weighted_score += reasoning_weight * reasoning_score
        score["overall"] = weighted_score / weight_sum
        scores.append(score)

    return scores
