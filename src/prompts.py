"""Prompt templates for different evaluation strategies."""

from src.config import MGSM_LANGUAGES, BELEBELE_LANGUAGES

# Language name lookup for prompt generation
LANG_NAMES = {**MGSM_LANGUAGES, **BELEBELE_LANGUAGES}


# =============================================================================
# MGSM (Math Reasoning) Prompts
# =============================================================================

def mgsm_direct_prompt(question: str, lang_code: str) -> str:
    """Direct inference: solve the math problem in its original language."""
    lang_name = LANG_NAMES.get(lang_code, lang_code)
    if lang_code == "en":
        return (
            f"Solve the following math problem. "
            f"Give ONLY the final numerical answer as a single number, nothing else.\n\n"
            f"Problem: {question}\n\nAnswer:"
        )
    return (
        f"Solve the following math problem written in {lang_name}. "
        f"Give ONLY the final numerical answer as a single number, nothing else.\n\n"
        f"Problem: {question}\n\nAnswer:"
    )


def mgsm_self_translate_prompt(question: str, lang_code: str) -> str:
    """Self-translate: translate to English first, then solve."""
    lang_name = LANG_NAMES.get(lang_code, lang_code)
    if lang_code == "en":
        return mgsm_direct_prompt(question, lang_code)
    return (
        f"The following math problem is written in {lang_name}. "
        f"First, translate the problem to English. "
        f"Then, solve the translated problem step by step. "
        f"Finally, give ONLY the final numerical answer as a single number on the last line.\n\n"
        f"Problem: {question}\n\n"
        f"Translation and solution:"
    )


def mgsm_english_cot_prompt(question: str, lang_code: str) -> str:
    """English CoT: use English chain-of-thought reasoning."""
    lang_name = LANG_NAMES.get(lang_code, lang_code)
    if lang_code == "en":
        return (
            f"Solve the following math problem step by step. "
            f"Think through each step carefully, then give ONLY the final numerical answer "
            f"as a single number on the last line.\n\n"
            f"Problem: {question}\n\nStep-by-step solution:"
        )
    return (
        f"The following math problem is written in {lang_name}. "
        f"Please reason through this problem step by step IN ENGLISH. "
        f"Show your work in English, then give ONLY the final numerical answer "
        f"as a single number on the last line.\n\n"
        f"Problem: {question}\n\nStep-by-step solution (in English):"
    )


# =============================================================================
# Belebele (Reading Comprehension) Prompts
# =============================================================================

def belebele_direct_prompt(passage: str, question: str, choices: list, lang_code: str) -> str:
    """Direct inference: answer the reading comprehension question."""
    lang_name = LANG_NAMES.get(lang_code, lang_code)
    choices_str = "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
    if lang_code == "eng":
        return (
            f"Read the passage and answer the question by choosing the correct option. "
            f"Reply with ONLY the option number (1, 2, 3, or 4), nothing else.\n\n"
            f"Passage: {passage}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{choices_str}\n\nAnswer:"
        )
    return (
        f"Read the following passage and question written in {lang_name}. "
        f"Choose the correct answer option. "
        f"Reply with ONLY the option number (1, 2, 3, or 4), nothing else.\n\n"
        f"Passage: {passage}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{choices_str}\n\nAnswer:"
    )


def belebele_self_translate_prompt(passage: str, question: str, choices: list, lang_code: str) -> str:
    """Self-translate: translate passage/question to English, then answer."""
    lang_name = LANG_NAMES.get(lang_code, lang_code)
    choices_str = "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
    if lang_code == "eng":
        return belebele_direct_prompt(passage, question, choices, lang_code)
    return (
        f"The following passage, question, and options are written in {lang_name}. "
        f"First, translate everything to English. "
        f"Then, answer the question by choosing the correct option. "
        f"End your response with ONLY the option number (1, 2, 3, or 4) on the last line.\n\n"
        f"Passage: {passage}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{choices_str}\n\n"
        f"Translation and answer:"
    )


def belebele_english_cot_prompt(passage: str, question: str, choices: list, lang_code: str) -> str:
    """English CoT: reason in English about the passage."""
    lang_name = LANG_NAMES.get(lang_code, lang_code)
    choices_str = "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
    if lang_code == "eng":
        return (
            f"Read the passage and answer the question. "
            f"Think step by step about what the passage says, then choose the correct option. "
            f"End your response with ONLY the option number (1, 2, 3, or 4) on the last line.\n\n"
            f"Passage: {passage}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{choices_str}\n\n"
            f"Step-by-step reasoning and answer:"
        )
    return (
        f"The following passage, question, and options are written in {lang_name}. "
        f"Please reason about this IN ENGLISH, step by step. "
        f"Analyze the passage content, then choose the correct option. "
        f"End your response with ONLY the option number (1, 2, 3, or 4) on the last line.\n\n"
        f"Passage: {passage}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{choices_str}\n\n"
        f"Step-by-step reasoning (in English) and answer:"
    )


# =============================================================================
# Prompt dispatcher
# =============================================================================

MGSM_PROMPT_FN = {
    "direct": mgsm_direct_prompt,
    "self_translate": mgsm_self_translate_prompt,
    "english_cot": mgsm_english_cot_prompt,
}

BELEBELE_PROMPT_FN = {
    "direct": belebele_direct_prompt,
    "self_translate": belebele_self_translate_prompt,
    "english_cot": belebele_english_cot_prompt,
}
