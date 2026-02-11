"""Main experiment runner for multilingual LLM evaluation."""
import json
import logging
import time
import sys
import os
from datetime import datetime
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODELS, STRATEGIES, MGSM_LANGUAGES, BELEBELE_LANGUAGES,
    RESULTS_DIR, SEED, SAMPLES_PER_LANGUAGE, LANGUAGE_RESOURCE_LEVEL,
)
from src.data_loader import load_mgsm_samples, load_belebele_samples
from src.prompts import MGSM_PROMPT_FN, BELEBELE_PROMPT_FN
from src.api_client import get_client, call_model
from src.evaluation import extract_number, extract_choice, compute_accuracy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger(__name__)


def run_mgsm_experiment(model_name: str, strategy: str, mgsm_data: dict, client=None):
    """Run MGSM experiment for one model and one strategy across all languages.

    Returns dict: {lang_code: {"predictions": [...], "gold": [...],
                                "responses": [...], "accuracy": float}}
    """
    prompt_fn = MGSM_PROMPT_FN[strategy]
    results = {}

    for lang_code, lang_name in MGSM_LANGUAGES.items():
        items = mgsm_data[lang_code]
        predictions = []
        gold = []
        responses = []

        desc = f"MGSM {model_name}/{strategy}/{lang_code}"
        for item in tqdm(items, desc=desc, leave=False):
            prompt = prompt_fn(item["question"], lang_code)
            response = call_model(model_name, prompt, client)
            pred = extract_number(response)
            predictions.append(pred)
            gold.append(item["answer"])
            responses.append(response)

        accuracy = compute_accuracy(predictions, gold)
        results[lang_code] = {
            "predictions": predictions,
            "gold": gold,
            "responses": responses,
            "accuracy": accuracy,
            "language": lang_name,
        }
        logger.info(f"MGSM {model_name}/{strategy}/{lang_code}: accuracy={accuracy:.3f}")

    return results


def run_belebele_experiment(model_name: str, strategy: str, belebele_data: dict, client=None):
    """Run Belebele experiment for one model and one strategy across all languages.

    Returns dict: {lang_code: {"predictions": [...], "gold": [...],
                                "responses": [...], "accuracy": float}}
    """
    prompt_fn = BELEBELE_PROMPT_FN[strategy]
    results = {}

    for lang_code, lang_name in BELEBELE_LANGUAGES.items():
        items = belebele_data[lang_code]
        predictions = []
        gold = []
        responses = []

        desc = f"Belebele {model_name}/{strategy}/{lang_code}"
        for item in tqdm(items, desc=desc, leave=False):
            prompt = prompt_fn(item["passage"], item["question"], item["choices"], lang_code)
            response = call_model(model_name, prompt, client)
            pred = extract_choice(response)
            predictions.append(pred)
            gold.append(item["correct"])
            responses.append(response)

        accuracy = compute_accuracy(predictions, gold)
        results[lang_code] = {
            "predictions": predictions,
            "gold": gold,
            "responses": responses,
            "accuracy": accuracy,
            "language": lang_name,
        }
        logger.info(f"Belebele {model_name}/{strategy}/{lang_code}: accuracy={accuracy:.3f}")

    return results


def run_all_experiments():
    """Run the complete experiment suite."""
    logger.info("=" * 60)
    logger.info("Starting Multilingual LLM Evaluation Experiments")
    logger.info(f"Seed: {SEED}, Samples/language: {SAMPLES_PER_LANGUAGE}")
    logger.info(f"Models: {list(MODELS.keys())}")
    logger.info(f"Strategies: {STRATEGIES}")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading datasets...")
    mgsm_data = load_mgsm_samples()
    belebele_data = load_belebele_samples()
    logger.info(f"MGSM: {sum(len(v) for v in mgsm_data.values())} total items")
    logger.info(f"Belebele: {sum(len(v) for v in belebele_data.values())} total items")

    # Store all results
    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "seed": SEED,
            "samples_per_language": SAMPLES_PER_LANGUAGE,
            "models": list(MODELS.keys()),
            "strategies": STRATEGIES,
        },
        "mgsm": {},
        "belebele": {},
    }

    start_time = time.time()

    for model_name in MODELS:
        logger.info(f"\n{'='*40}")
        logger.info(f"Running experiments for {model_name}")
        logger.info(f"{'='*40}")

        client = get_client(model_name)
        all_results["mgsm"][model_name] = {}
        all_results["belebele"][model_name] = {}

        for strategy in STRATEGIES:
            logger.info(f"\n--- Strategy: {strategy} ---")

            # MGSM
            mgsm_results = run_mgsm_experiment(model_name, strategy, mgsm_data, client)
            # Save without raw responses (too large for JSON)
            all_results["mgsm"][model_name][strategy] = {
                lang: {
                    "accuracy": r["accuracy"],
                    "language": r["language"],
                    "n_correct": sum(1 for p, g in zip(r["predictions"], r["gold"]) if p == g),
                    "n_total": len(r["gold"]),
                }
                for lang, r in mgsm_results.items()
            }

            # Save raw responses separately
            raw_path = f"{RESULTS_DIR}/raw/mgsm_{model_name}_{strategy}.json"
            with open(raw_path, "w") as f:
                # Convert to serializable format
                serializable = {
                    lang: {
                        "predictions": r["predictions"],
                        "gold": r["gold"],
                        "responses": r["responses"],
                    }
                    for lang, r in mgsm_results.items()
                }
                json.dump(serializable, f, indent=2, ensure_ascii=False)

            # Belebele
            belebele_results = run_belebele_experiment(model_name, strategy, belebele_data, client)
            all_results["belebele"][model_name][strategy] = {
                lang: {
                    "accuracy": r["accuracy"],
                    "language": r["language"],
                    "n_correct": sum(1 for p, g in zip(r["predictions"], r["gold"]) if p == g),
                    "n_total": len(r["gold"]),
                }
                for lang, r in belebele_results.items()
            }

            raw_path = f"{RESULTS_DIR}/raw/belebele_{model_name}_{strategy}.json"
            with open(raw_path, "w") as f:
                serializable = {
                    lang: {
                        "predictions": r["predictions"],
                        "gold": r["gold"],
                        "responses": r["responses"],
                    }
                    for lang, r in belebele_results.items()
                }
                json.dump(serializable, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    all_results["metadata"]["total_time_seconds"] = elapsed
    logger.info(f"\nTotal experiment time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save summary results
    results_path = f"{RESULTS_DIR}/experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return all_results


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/raw", exist_ok=True)
    results = run_all_experiments()

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    for task in ["mgsm", "belebele"]:
        print(f"\n--- {task.upper()} ---")
        for model in results[task]:
            for strategy in results[task][model]:
                accs = {
                    lang: r["accuracy"]
                    for lang, r in results[task][model][strategy].items()
                }
                avg = sum(accs.values()) / len(accs)
                print(f"  {model}/{strategy}: avg={avg:.3f} | " +
                      " | ".join(f"{l}={a:.3f}" for l, a in accs.items()))
