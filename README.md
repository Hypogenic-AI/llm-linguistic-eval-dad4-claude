# Multilingual LLM Evaluation: Cross-Language Equity and Implicit Translation Mechanisms

A systematic evaluation of GPT-4.1 and Claude Sonnet 4 across 8 languages, 3 prompting strategies, and 2 benchmarks (MGSM, Belebele) to assess multilingual performance equity and test for implicit internal translation mechanisms.

## Key Findings

- **Frontier LLMs have largely closed the multilingual gap.** Claude Sonnet 4 achieves 93-97% accuracy across all tested languages without any specialized prompting.
- **English-pivoting no longer helps.** Self-translate and English CoT strategies provide negligible benefit for current frontier models.
- **GPT-4.1 is format-sensitive, not language-biased.** Its low direct MGSM accuracy (0.60) stems from lack of chain-of-thought, not from language difficulty.
- **Democratization scores are high** (0.86-0.97), with Claude achieving near-perfect equity across languages.

See [REPORT.md](REPORT.md) for the full analysis.

## Project Structure

```
.
├── REPORT.md                # Full research report with findings
├── README.md                # This file
├── planning.md              # Research plan and experimental design
├── literature_review.md     # Literature survey
├── pyproject.toml           # Python dependencies
├── src/
│   ├── config.py            # Experiment configuration (models, languages, etc.)
│   ├── data_loader.py       # Dataset loading and sampling
│   ├── prompts.py           # Prompt templates for all strategies
│   ├── api_client.py        # LLM API client with retry logic
│   ├── evaluation.py        # Response parsing and metrics
│   ├── run_experiments.py   # Main experiment runner
│   └── analysis.py          # Statistical analysis and visualization
├── datasets/                # MGSM and Belebele data files
├── results/
│   ├── experiment_results.json   # Full experiment results
│   ├── accuracy_summary.csv      # Flat accuracy table
│   ├── h1_performance_gaps.csv   # H1 analysis data
│   ├── h2_strategy_lifts.csv     # H2 analysis data
│   └── raw/                      # Raw model responses
├── figures/                 # Generated visualizations (12 plots)
└── logs/                    # Experiment logs
```

## Experimental Design

**Factors:**
- 2 models: GPT-4.1, Claude Sonnet 4
- 3 strategies: direct, self-translate, English CoT
- 8 languages: English, Chinese, German, French, Russian, Japanese, Swahili, Bengali/Hindi
- 2 tasks: MGSM (math reasoning), Belebele (reading comprehension)
- 50 samples per language per task

**Total: 4,800 API calls**

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- API keys: `OPENAI_API_KEY`, `OPENROUTER_KEY`

### Setup

```bash
# Install dependencies
uv sync

# Set API keys
export OPENAI_API_KEY="your-openai-key"
export OPENROUTER_KEY="your-openrouter-key"
```

### Run Experiments

```bash
# Run full experiment suite (~4.5 hours)
python3 src/run_experiments.py

# Run analysis and generate figures
python3 src/analysis.py
```

### View Results

- **Results JSON:** `results/experiment_results.json`
- **Accuracy CSV:** `results/accuracy_summary.csv`
- **Figures:** `figures/*.png` (12 visualizations)
- **Full Report:** [REPORT.md](REPORT.md)

## Results Summary

### MGSM Accuracy (Math Reasoning)

| Model | Strategy | Avg | Range |
|-------|----------|-----|-------|
| GPT-4.1 | direct | 0.600 | 0.50-0.70 |
| GPT-4.1 | self_translate | 0.862 | 0.60-0.96 |
| GPT-4.1 | english_cot | 0.915 | 0.84-0.96 |
| Claude Sonnet 4 | direct | 0.935 | 0.90-0.98 |
| Claude Sonnet 4 | self_translate | 0.930 | 0.88-0.96 |
| Claude Sonnet 4 | english_cot | 0.938 | 0.84-1.00 |

### Belebele Accuracy (Reading Comprehension)

| Model | Strategy | Avg | Range |
|-------|----------|-----|-------|
| GPT-4.1 | direct | 0.938 | 0.82-1.00 |
| GPT-4.1 | self_translate | 0.948 | 0.80-1.00 |
| GPT-4.1 | english_cot | 0.950 | 0.82-1.00 |
| Claude Sonnet 4 | direct | 0.970 | 0.90-1.00 |
| Claude Sonnet 4 | self_translate | 0.968 | 0.88-1.00 |
| Claude Sonnet 4 | english_cot | 0.965 | 0.88-1.00 |

## License

Research project for academic purposes.
