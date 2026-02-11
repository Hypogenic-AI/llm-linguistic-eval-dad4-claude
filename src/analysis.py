"""Statistical analysis and visualization for multilingual LLM evaluation."""
import json
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from scipy import stats
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    RESULTS_DIR, FIGURES_DIR, LANGUAGE_RESOURCE_LEVEL,
    MGSM_LANGUAGES, BELEBELE_LANGUAGES,
)
from src.evaluation import compute_democratization_score

# Consistent style
sns.set_theme(style="whitegrid", font_scale=1.2)
COLORS = sns.color_palette("Set2", 8)


def load_results():
    """Load experiment results from JSON."""
    with open(f"{RESULTS_DIR}/experiment_results.json") as f:
        return json.load(f)


def build_accuracy_dataframe(results):
    """Build a flat DataFrame of all accuracy results for analysis."""
    rows = []
    for task in ["mgsm", "belebele"]:
        lang_map = MGSM_LANGUAGES if task == "mgsm" else BELEBELE_LANGUAGES
        for model in results[task]:
            for strategy in results[task][model]:
                for lang_code, data in results[task][model][strategy].items():
                    lang_name = lang_map.get(lang_code, lang_code)
                    resource = LANGUAGE_RESOURCE_LEVEL.get(lang_name, "unknown")
                    rows.append({
                        "task": task,
                        "model": model,
                        "strategy": strategy,
                        "lang_code": lang_code,
                        "language": lang_name,
                        "resource_level": resource,
                        "accuracy": data["accuracy"],
                        "n_correct": data["n_correct"],
                        "n_total": data["n_total"],
                    })
    return pd.DataFrame(rows)


def analyze_h1_performance_gap(df):
    """H1: LLMs show reduced performance on non-English vs English."""
    print("\n" + "=" * 70)
    print("H1: Performance Gap Between English and Non-English")
    print("=" * 70)

    results_h1 = []
    for task in df["task"].unique():
        eng_code = "en" if task == "mgsm" else "eng"
        for model in df["model"].unique():
            for strategy in df["strategy"].unique():
                mask = (df["task"] == task) & (df["model"] == model) & (df["strategy"] == strategy)
                subset = df[mask]
                eng_acc = subset[subset["lang_code"] == eng_code]["accuracy"].values
                if len(eng_acc) == 0:
                    continue
                eng_acc = eng_acc[0]

                non_eng = subset[subset["lang_code"] != eng_code]
                for _, row in non_eng.iterrows():
                    gap = eng_acc - row["accuracy"]
                    results_h1.append({
                        "task": task,
                        "model": model,
                        "strategy": strategy,
                        "language": row["language"],
                        "resource_level": row["resource_level"],
                        "english_acc": eng_acc,
                        "lang_acc": row["accuracy"],
                        "gap": gap,
                    })

    h1_df = pd.DataFrame(results_h1)
    print("\nPerformance gaps (English - target language):")
    print(h1_df.groupby(["task", "model", "strategy"])["gap"].agg(["mean", "std", "min", "max"]).round(3))

    # Statistical test: one-sample t-test on gaps (H0: gap = 0)
    print("\nStatistical tests (H0: gap = 0):")
    for task in h1_df["task"].unique():
        for model in h1_df["model"].unique():
            for strategy in h1_df["strategy"].unique():
                mask = (h1_df["task"] == task) & (h1_df["model"] == model) & (h1_df["strategy"] == strategy)
                gaps = h1_df[mask]["gap"].values
                if len(gaps) < 2:
                    continue
                t_stat, p_val = stats.ttest_1samp(gaps, 0)
                d = np.mean(gaps) / (np.std(gaps, ddof=1) + 1e-10)  # Cohen's d
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"  {task}/{model}/{strategy}: mean_gap={np.mean(gaps):.3f}, t={t_stat:.2f}, p={p_val:.4f} {sig}, Cohen's d={d:.2f}")

    return h1_df


def analyze_h2_self_translate_effect(df):
    """H2: Self-translate and English CoT improve non-English performance."""
    print("\n" + "=" * 70)
    print("H2: Effect of English-Pivoting Strategies")
    print("=" * 70)

    results_h2 = []
    for task in df["task"].unique():
        eng_code = "en" if task == "mgsm" else "eng"
        for model in df["model"].unique():
            for lang_code in df[df["task"] == task]["lang_code"].unique():
                if lang_code == eng_code:
                    continue
                direct_mask = (df["task"] == task) & (df["model"] == model) & \
                             (df["strategy"] == "direct") & (df["lang_code"] == lang_code)
                direct_acc = df[direct_mask]["accuracy"].values
                if len(direct_acc) == 0:
                    continue
                direct_acc = direct_acc[0]

                for strategy in ["self_translate", "english_cot"]:
                    strat_mask = (df["task"] == task) & (df["model"] == model) & \
                                (df["strategy"] == strategy) & (df["lang_code"] == lang_code)
                    strat_acc = df[strat_mask]["accuracy"].values
                    if len(strat_acc) == 0:
                        continue
                    strat_acc = strat_acc[0]

                    lang_name = df[strat_mask]["language"].values[0]
                    resource = df[strat_mask]["resource_level"].values[0]
                    results_h2.append({
                        "task": task,
                        "model": model,
                        "strategy": strategy,
                        "language": lang_name,
                        "lang_code": lang_code,
                        "resource_level": resource,
                        "direct_acc": direct_acc,
                        "strategy_acc": strat_acc,
                        "lift": strat_acc - direct_acc,
                    })

    h2_df = pd.DataFrame(results_h2)
    print("\nSelf-translate / English CoT lift over direct inference:")
    print(h2_df.groupby(["task", "model", "strategy"])["lift"].agg(["mean", "std", "min", "max"]).round(3))

    # Test: is the average lift significantly > 0?
    print("\nStatistical tests (H0: lift = 0):")
    for task in h2_df["task"].unique():
        for model in h2_df["model"].unique():
            for strategy in h2_df["strategy"].unique():
                mask = (h2_df["task"] == task) & (h2_df["model"] == model) & (h2_df["strategy"] == strategy)
                lifts = h2_df[mask]["lift"].values
                if len(lifts) < 2:
                    continue
                t_stat, p_val = stats.ttest_1samp(lifts, 0)
                # One-sided test (we expect lift > 0)
                p_one_sided = p_val / 2 if t_stat > 0 else 1 - p_val / 2
                sig = "***" if p_one_sided < 0.001 else "**" if p_one_sided < 0.01 else "*" if p_one_sided < 0.05 else "ns"
                print(f"  {task}/{model}/{strategy}: mean_lift={np.mean(lifts):.3f}, t={t_stat:.2f}, p(one-sided)={p_one_sided:.4f} {sig}")

    return h2_df


def analyze_h3_resource_correlation(df):
    """H3: Performance gap correlates with language resource level."""
    print("\n" + "=" * 70)
    print("H3: Correlation Between Resource Level and Performance")
    print("=" * 70)

    resource_rank = {"high": 3, "medium": 2, "low": 1}

    for task in df["task"].unique():
        eng_code = "en" if task == "mgsm" else "eng"
        for model in df["model"].unique():
            mask = (df["task"] == task) & (df["model"] == model) & (df["strategy"] == "direct")
            subset = df[mask].copy()
            non_eng = subset[subset["lang_code"] != eng_code]
            if len(non_eng) < 3:
                continue
            ranks = non_eng["resource_level"].map(resource_rank)
            rho, p_val = stats.spearmanr(ranks, non_eng["accuracy"])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  {task}/{model} (direct): Spearman rho={rho:.3f}, p={p_val:.4f} {sig}")
            print(f"    High-resource avg: {non_eng[non_eng['resource_level']=='high']['accuracy'].mean():.3f}")
            print(f"    Medium-resource avg: {non_eng[non_eng['resource_level']=='medium']['accuracy'].mean():.3f}")
            print(f"    Low-resource avg: {non_eng[non_eng['resource_level']=='low']['accuracy'].mean():.3f}")


def compute_democratization_scores(df):
    """Compute democratization scores across conditions."""
    print("\n" + "=" * 70)
    print("Democratization Scores (avg/max language accuracy)")
    print("=" * 70)

    for task in df["task"].unique():
        for model in df["model"].unique():
            for strategy in df["strategy"].unique():
                mask = (df["task"] == task) & (df["model"] == model) & (df["strategy"] == strategy)
                subset = df[mask]
                lang_accs = dict(zip(subset["language"], subset["accuracy"]))
                dem_score = compute_democratization_score(lang_accs)
                print(f"  {task}/{model}/{strategy}: {dem_score:.3f}")


def plot_performance_by_language(df):
    """Create grouped bar charts of accuracy by language for each task."""
    for task in df["task"].unique():
        task_df = df[df["task"] == task]

        for model in task_df["model"].unique():
            model_df = task_df[task_df["model"] == model]

            fig, ax = plt.subplots(figsize=(14, 6))
            pivot = model_df.pivot_table(
                index="language", columns="strategy", values="accuracy"
            )
            # Sort by direct accuracy
            if "direct" in pivot.columns:
                pivot = pivot.sort_values("direct", ascending=False)

            pivot.plot(kind="bar", ax=ax, width=0.8)
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("Language")
            ax.set_title(f"{task.upper()} — {model}: Accuracy by Language and Strategy")
            ax.set_ylim(0, 1.05)
            ax.legend(title="Strategy", loc="lower left")
            ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="Random (Belebele)" if task == "belebele" else "")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            path = f"{FIGURES_DIR}/{task}_{model.replace('.', '_')}_by_language.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {path}")


def plot_performance_gap_heatmap(df):
    """Create a heatmap of performance gaps (English - target) across models and strategies."""
    for task in df["task"].unique():
        eng_code = "en" if task == "mgsm" else "eng"
        task_df = df[df["task"] == task]

        for model in task_df["model"].unique():
            model_df = task_df[task_df["model"] == model]
            eng_accs = {}
            for strategy in model_df["strategy"].unique():
                mask = (model_df["strategy"] == strategy) & (model_df["lang_code"] == eng_code)
                eng_accs[strategy] = model_df[mask]["accuracy"].values[0]

            gap_data = []
            for _, row in model_df.iterrows():
                if row["lang_code"] == eng_code:
                    continue
                gap_data.append({
                    "Language": row["language"],
                    "Strategy": row["strategy"],
                    "Gap": eng_accs[row["strategy"]] - row["accuracy"],
                })

            gap_df = pd.DataFrame(gap_data)
            pivot = gap_df.pivot_table(index="Language", columns="Strategy", values="Gap")

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", center=0,
                       ax=ax, vmin=-0.1, vmax=0.5)
            ax.set_title(f"{task.upper()} — {model}: Performance Gap (English - Target)")
            plt.tight_layout()
            path = f"{FIGURES_DIR}/{task}_{model.replace('.', '_')}_gap_heatmap.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {path}")


def plot_model_comparison(df):
    """Compare models side by side for each task (direct strategy)."""
    for task in df["task"].unique():
        direct_df = df[(df["task"] == task) & (df["strategy"] == "direct")]
        if direct_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        pivot = direct_df.pivot_table(index="language", columns="model", values="accuracy")
        if not pivot.empty:
            pivot = pivot.sort_values(pivot.columns[0], ascending=False)
        pivot.plot(kind="bar", ax=ax, width=0.7)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Language")
        ax.set_title(f"{task.upper()} — Model Comparison (Direct Inference)")
        ax.set_ylim(0, 1.05)
        ax.legend(title="Model")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path = f"{FIGURES_DIR}/{task}_model_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


def plot_strategy_lift(h2_df):
    """Plot the lift from self-translate and English CoT by language resource level."""
    if h2_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, strategy in enumerate(["self_translate", "english_cot"]):
        strat_df = h2_df[h2_df["strategy"] == strategy]
        if strat_df.empty:
            continue
        ax = axes[i]
        sns.boxplot(data=strat_df, x="resource_level", y="lift",
                   order=["high", "medium", "low"], ax=ax, palette="Set2")
        sns.stripplot(data=strat_df, x="resource_level", y="lift",
                     order=["high", "medium", "low"], ax=ax, color="black",
                     size=5, alpha=0.6)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax.set_title(f"Lift from {strategy.replace('_', ' ').title()}")
        ax.set_xlabel("Resource Level")
        ax.set_ylabel("Accuracy Lift (strategy - direct)")

    plt.suptitle("Effect of English-Pivoting Strategies by Resource Level", fontsize=14)
    plt.tight_layout()
    path = f"{FIGURES_DIR}/strategy_lift_by_resource.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_democratization_comparison(df):
    """Bar chart of democratization scores across conditions."""
    records = []
    for task in df["task"].unique():
        for model in df["model"].unique():
            for strategy in df["strategy"].unique():
                mask = (df["task"] == task) & (df["model"] == model) & (df["strategy"] == strategy)
                subset = df[mask]
                lang_accs = dict(zip(subset["language"], subset["accuracy"]))
                dem_score = compute_democratization_score(lang_accs)
                records.append({
                    "Task": task.upper(),
                    "Model": model,
                    "Strategy": strategy,
                    "Democratization": dem_score,
                })

    dem_df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=dem_df, x="Strategy", y="Democratization", hue="Model", ax=ax)
    ax.set_ylabel("Democratization Score (avg/max)")
    ax.set_title("Cross-Language Equity: Democratization Scores")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Perfect equity")
    ax.legend(title="Model")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/democratization_scores.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def run_full_analysis():
    """Run complete analysis pipeline."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading results...")
    results = load_results()

    print("Building dataframe...")
    df = build_accuracy_dataframe(results)
    print(f"Total records: {len(df)}")
    print(f"\nOverall accuracy summary:")
    print(df.groupby(["task", "model", "strategy"])["accuracy"].mean().round(3))

    # Save accuracy dataframe
    df.to_csv(f"{RESULTS_DIR}/accuracy_summary.csv", index=False)
    print(f"\nSaved accuracy summary to {RESULTS_DIR}/accuracy_summary.csv")

    # Hypothesis tests
    h1_df = analyze_h1_performance_gap(df)
    h2_df = analyze_h2_self_translate_effect(df)
    analyze_h3_resource_correlation(df)
    compute_democratization_scores(df)

    # Visualizations
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    plot_performance_by_language(df)
    plot_performance_gap_heatmap(df)
    plot_model_comparison(df)
    plot_strategy_lift(h2_df)
    plot_democratization_comparison(df)

    # Save analysis summary
    h1_df.to_csv(f"{RESULTS_DIR}/h1_performance_gaps.csv", index=False)
    h2_df.to_csv(f"{RESULTS_DIR}/h2_strategy_lifts.csv", index=False)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results in: {RESULTS_DIR}/")
    print(f"Figures in: {FIGURES_DIR}/")
    print("=" * 70)

    return df, h1_df, h2_df


if __name__ == "__main__":
    run_full_analysis()
