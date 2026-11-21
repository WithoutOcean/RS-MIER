"""Independent format + literal final-answer rewards. No judge/model calls."""
import math
import re

REWARD_NAME = "format_answer"
REWARD_TYPE = "batch"


def normalize_tags(response):
    return re.sub(r"<\s*(/?)\s*(perception|think|answer)\s*>",
                  lambda m: f"<{m.group(1)}{m.group(2).lower()}>",
                  str(response), flags=re.IGNORECASE)


def format_reward(response):
    text = normalize_tags(response)
    for name in ("perception", "think", "answer"):
        if text.count(f"<{name}>") != 1 or text.count(f"</{name}>") != 1:
            return -1.0
    match = re.fullmatch(
        r"\s*<perception>(.*?)</perception>\s*<think>(.*?)</think>"
        r"\s*<answer>(.*?)</answer>\s*", text, flags=re.DOTALL)
    return 0.0 if match and all(part.strip() for part in match.groups()) else -1.0


def answer_reward(response, ground_truth):
    text = normalize_tags(response)
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    predicted = (match.group(1) if match else text).strip().casefold()
    target = str(ground_truth).strip().casefold()
    return 1.0 if predicted and predicted == target else 0.0


def compute_score(reward_inputs, format_weight=0.1, answer_weight=0.9):
    weights = (float(format_weight), float(answer_weight))
    if not all(math.isfinite(w) and w >= 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Reward weights must be finite, nonnegative, with positive sum")
    scores = []
    for item in reward_inputs:
        fmt = format_reward(item["response"])
        answer = answer_reward(item["response"], item["ground_truth"])
        scores.append({"overall": weights[0] * fmt + weights[1] * answer,
                       "format": fmt, "answer": answer})
    return scores
