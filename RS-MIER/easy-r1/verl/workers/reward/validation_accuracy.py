"""Final-answer validation only; independent of all training reward functions."""
import re


def extract_final_answer(response):
    text = re.sub(
        r"<\s*(/?)\s*answer\s*>",
        lambda match: f"<{match.group(1)}answer>",
        str(response), flags=re.IGNORECASE,
    )
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return (match.group(1) if match else text).strip()


def final_answer_correct(response, ground_truth):
    # No punctuation, internal whitespace, numeric or semantic normalization.
    answer = extract_final_answer(response).casefold()
    return bool(answer) and answer == str(ground_truth).strip().casefold()


class FinalAnswerAccuracy:
    """CPU-only evaluator: decode final answers and return one boolean per response."""

    def __init__(self, tokenizer, skip_special_tokens=True):
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens

    def compute_accuracy(self, data):
        responses = data.batch["responses"]
        lengths = data.batch["response_mask"].sum(dim=-1)
        references = data.non_tensor_batch["ground_truth"]
        if len(references) != len(responses):
            raise ValueError("Validation response/reference count mismatch")
        correct = []
        for index, reference in enumerate(references):
            response = self.tokenizer.decode(
                responses[index, :int(lengths[index].item())],
                skip_special_tokens=self.skip_special_tokens,
            )
            correct.append(final_answer_correct(response, reference))
        return correct
