"""Data loading and sampling for multilingual evaluation."""
import random
import pandas as pd
from datasets import load_from_disk
from src.config import SEED, SAMPLES_PER_LANGUAGE, DATA_DIR, MGSM_LANGUAGES, BELEBELE_LANGUAGES


def load_mgsm_samples():
    """Load MGSM data and sample SAMPLES_PER_LANGUAGE items per language.

    Returns dict: {lang_code: [{"question": str, "answer": int}, ...]}
    """
    random.seed(SEED)
    samples = {}
    for lang_code in MGSM_LANGUAGES:
        path = f"{DATA_DIR}/mgsm/mgsm_{lang_code}.tsv"
        df = pd.read_csv(path, sep="\t", header=None, names=["question", "answer"])
        indices = random.sample(range(len(df)), min(SAMPLES_PER_LANGUAGE, len(df)))
        items = []
        for idx in indices:
            raw_answer = str(df.iloc[idx]["answer"]).replace(",", "").strip()
            items.append({
                "question": df.iloc[idx]["question"],
                "answer": int(raw_answer),
                "index": idx,
            })
        samples[lang_code] = items
    return samples


def load_belebele_samples():
    """Load Belebele data and sample SAMPLES_PER_LANGUAGE items per language.

    Returns dict: {lang_code: [{"passage": str, "question": str,
                                "choices": [str, str, str, str],
                                "correct": int (1-4)}, ...]}
    """
    random.seed(SEED)
    samples = {}

    # First, determine which indices to use (same across languages for parallel comparison)
    eng_ds = load_from_disk(f"{DATA_DIR}/belebele_eng")["test"]
    indices = random.sample(range(len(eng_ds)), min(SAMPLES_PER_LANGUAGE, len(eng_ds)))

    for lang_code in BELEBELE_LANGUAGES:
        dir_name = f"belebele_{lang_code}"
        if lang_code == "eng":
            dir_name = "belebele_eng"
        ds = load_from_disk(f"{DATA_DIR}/{dir_name}")["test"]
        items = []
        for idx in indices:
            row = ds[idx]
            items.append({
                "passage": row["flores_passage"],
                "question": row["question"],
                "choices": [
                    row["mc_answer1"],
                    row["mc_answer2"],
                    row["mc_answer3"],
                    row["mc_answer4"],
                ],
                "correct": int(row["correct_answer_num"]),
                "index": idx,
            })
        samples[lang_code] = items
    return samples


if __name__ == "__main__":
    mgsm = load_mgsm_samples()
    for lang, items in mgsm.items():
        print(f"MGSM {lang}: {len(items)} samples, first answer={items[0]['answer']}")

    belebele = load_belebele_samples()
    for lang, items in belebele.items():
        print(f"Belebele {lang}: {len(items)} samples, first correct={items[0]['correct']}")
