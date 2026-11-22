

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


# Metadata
REWARD_NAME = "math"
REWARD_TYPE = "batch"


OFFICIAL_LABELS = {
    "0",
    "0_to_10",
    "10_to_20",
    "20_to_30",
    "30_to_40",
    "40_to_50",
    "50_to_60",
    "60_to_70",
    "70_to_80",
    "80_to_90",
    "90_to_100",
    "NVG_surface",
    "buildings",
    "low_vegetation",
    "playgrounds",
    "trees",
    "water",
    "no",
    "yes",
    "green_house",
    "road",
    "bridge",
    "others",
}


CLASS_ALIASES = {
    "NVG_surface": {
        "nvg_surface",
        "nvg surface",
        "non-vegetated ground surface",
        "non vegetated ground surface",
        "nonvegetated ground surface",
        "non-vegetated ground",
        "non vegetated ground",
        "nonvegetated ground",
        "non-vegetated surface",
        "non vegetated surface",
        "bare ground",
        "bare ground surface",
        "bare surface",
    },
    "buildings": {
        "building",
        "buildings",
        "building area",
        "building areas",
        "building region",
        "building regions",
        "built-up area",
        "built up area",
        "built-up areas",
        "built up areas",
        "built-up region",
        "built up region",
    },
    "low_vegetation": {
        "low_vegetation",
        "low vegetation",
        "low-vegetation",
        "low vegetation area",
        "low vegetation areas",
        "low vegetation region",
        "low vegetation regions",
    },
    "playgrounds": {
        "playground",
        "playgrounds",
        "playground area",
        "playground areas",
        "playground region",
        "playground regions",
    },
    "trees": {
        "tree",
        "trees",
        "tree area",
        "tree areas",
        "tree region",
        "tree regions",
    },
    "water": {
        "water",
        "water area",
        "water areas",
        "water region",
        "water regions",
        "water body",
        "water bodies",
    },
    "green_house": {
        "green_house",
        "green house",
        "green houses",
        "greenhouse",
        "greenhouses",
        "greenhouse area",
        "greenhouse areas",
        "greenhouse region",
        "greenhouse regions",
    },
    "road": {
        "road",
        "roads",
        "road area",
        "road areas",
        "road region",
        "road regions",
    },
    "bridge": {
        "bridge",
        "bridges",
        "bridge area",
        "bridge areas",
        "bridge region",
        "bridge regions",
    },
    "others": {
        "other",
        "others",
        "other category",
        "other categories",
        "other class",
        "other classes",
        "other land cover",
        "other land-cover",
        "other land cover type",
        "other land-cover type",
    },
}


def normalize_text(text: str) -> str:
    """Conservative normalization used before rule-based canonicalization."""
    text = str(text).strip().lower()


    text = text.replace("_", " ")
    text = text.replace("-", " ")

    text = re.sub(r"[^\w\s%]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_answer(response: str) -> str:
    """
    Extract <answer>...</answer>.

    Accuracy and format are intentionally independent:
    if the answer tag is missing, the whole response is used for accuracy
    rather than forcing accuracy=0 solely because the format failed.
    """
    match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        str(response),
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    return str(response).strip()


def canonicalize_binary(text: str) -> str | None:
   
    text = normalize_text(text)

    # Explicit leading answer is preferred.
    if re.match(r"^yes\b", text):
        return "yes"

    if re.match(r"^no\b", text):
        return "no"

    # Very short standalone variants.
    positive_exact = {
        "yes",
        "true",
        "changed",
        "it changed",
        "there is a change",
        "there was a change",
    }

    negative_exact = {
        "no",
        "false",
        "unchanged",
        "no change",
        "not changed",
        "it did not change",
        "did not change",
        "has not changed",
        "there is no change",
        "there was no change",
    }

    if text in positive_exact:
        return "yes"

    if text in negative_exact:
        return "no"

    return None


def canonicalize_ratio(text: str) -> str | None:
   
    raw = str(text).strip().lower()

    # Exact official range label.
    match = re.fullmatch(r"\s*(\d+)_to_(\d+)\s*", raw)
    if match:
        low = int(match.group(1))
        high = int(match.group(2))
        candidate = f"{low}_to_{high}"
        return candidate if candidate in OFFICIAL_LABELS else None

    normalized = normalize_text(raw)

    # Exact zero forms.
    if normalized in {
        "0",
        "0%",
        "zero",
        "zero percent",
        "zero percentage",
    }:
        return "0"

    patterns = [
       
        r"\bbetween\s+(\d{1,3})\s*%?\s+and\s+(\d{1,3})\s*%?",

       
        r"\bfrom\s+(\d{1,3})\s*%?\s+to\s+(\d{1,3})\s*%?",

        r"\b(\d{1,3})\s*%?\s+to\s+(\d{1,3})\s*%?",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            low = int(match.group(1))
            high = int(match.group(2))
            candidate = f"{low}_to_{high}"
            return candidate if candidate in OFFICIAL_LABELS else None

    
    match = re.search(
        r"\b(\d{1,3})\s*%?\s*[-~–—]\s*(\d{1,3})\s*%?",
        raw,
    )
    if match:
        low = int(match.group(1))
        high = int(match.group(2))
        candidate = f"{low}_to_{high}"
        return candidate if candidate in OFFICIAL_LABELS else None

    return None


def canonicalize_class(text: str) -> str | None:

    normalized = normalize_text(text)
    matched_labels: list[str] = []

    for canonical, aliases in CLASS_ALIASES.items():
        aliases_normalized = {normalize_text(alias) for alias in aliases}

        if normalized in aliases_normalized:
            matched_labels.append(canonical)
            continue

      
        for alias in aliases_normalized:
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                matched_labels.append(canonical)
                break

    matched_labels = list(dict.fromkeys(matched_labels))

    if len(matched_labels) == 1:
        return matched_labels[0]

    return None


def canonicalize_answer(text: str, ground_truth: str) -> str | None:
    
    gt = str(ground_truth).strip()

    if gt not in OFFICIAL_LABELS:
       
        return normalize_text(text)

    if gt in {"yes", "no"}:
        return canonicalize_binary(text)

    if gt == "0" or re.fullmatch(r"\d+_to_\d+", gt):
        return canonicalize_ratio(text)

    return canonicalize_class(text)


def format_reward(response: str) -> float:
    """
    Format reward.

    Required structure:
      <perception>non-empty</perception>
      <think>non-empty</think>
      <answer>non-empty</answer>

    The three sections must appear exactly once, in this order, with no
    additional text outside them.
    """
    pattern = re.compile(
        r"\s*"
        r"<perception>\s*.+?\s*</perception>\s*"
        r"<think>\s*.+?\s*</think>\s*"
        r"<answer>\s*.+?\s*</answer>"
        r"\s*",
        flags=re.IGNORECASE | re.DOTALL,
    )

    return 1.0 if re.fullmatch(pattern, str(response)) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """
    Answer accuracy reward.

    It compares canonical labels, not raw character strings.
    """
    answer = extract_answer(response)

    pred_canonical = canonicalize_answer(answer, ground_truth)
    gt_canonical = canonicalize_answer(ground_truth, ground_truth)

    if pred_canonical is None or gt_canonical is None:
        return 0.0

    return 1.0 if pred_canonical == gt_canonical else 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
) -> list[dict[str, float]]:

    scores = []

    for reward_input in reward_inputs:
        
        response = re.sub(
            r"\s*(<|>|/)\s*",
            r"\1",
            str(reward_input["response"]),
        )

        format_score = format_reward(response)

        accuracy_score = accuracy_reward(
            response,
            str(reward_input["ground_truth"]),
        )

        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score
                + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
