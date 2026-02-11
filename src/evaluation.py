"""Response parsing and evaluation metrics."""
import re


def extract_number(text: str) -> int | None:
    """Extract the final numerical answer from a model response.
    Looks for the last number in the response (integer).
    """
    if not text or text.startswith("ERROR"):
        return None
    # Remove commas from numbers (e.g., 70,000 -> 70000)
    cleaned = text.replace(",", "")
    # Find all integers (possibly negative)
    numbers = re.findall(r'-?\d+', cleaned)
    if numbers:
        return int(numbers[-1])
    return None


def extract_choice(text: str) -> int | None:
    """Extract a multiple choice answer (1-4) from a model response.
    Looks for the last occurrence of 1, 2, 3, or 4.
    """
    if not text or text.startswith("ERROR"):
        return None
    # Look for standalone digits 1-4 (last occurrence)
    matches = re.findall(r'\b([1-4])\b', text)
    if matches:
        return int(matches[-1])
    return None


def compute_accuracy(predictions: list, gold: list) -> float:
    """Compute accuracy from predictions and gold labels.
    Ignores None predictions (counts as wrong).
    """
    if not predictions:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, gold) if p == g)
    return correct / len(gold)


def compute_democratization_score(lang_accuracies: dict) -> float:
    """Compute democratization score: mean accuracy / max accuracy.
    Score of 1.0 means perfect equity across languages.
    """
    if not lang_accuracies:
        return 0.0
    values = list(lang_accuracies.values())
    max_acc = max(values)
    if max_acc == 0:
        return 0.0
    return sum(values) / len(values) / max_acc
