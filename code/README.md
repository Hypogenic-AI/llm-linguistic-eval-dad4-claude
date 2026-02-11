# Code Repositories for Multilingual LLM Evaluation Research

This directory contains three key repositories studying how multilingual LLMs process
and represent different languages internally. Each tackles a different aspect of the
"Do LLMs think in English?" question.

---

## 1. `llm-latent-language/` — Logit Lens Analysis of Latent Language

**Paper:** "Do Llamas Work in English? On the Latent Language of Multilingual Transformers"
(Wendler et al., 2024, EPFL dlab) — [arXiv:2402.10588](https://arxiv.org/abs/2402.10588)

**What it does:** Uses the "logit lens" technique to decode intermediate hidden states of
Llama-2 at every layer, revealing which language the model uses internally (its "latent
language") when processing non-English prompts. Analyzes both translation and cloze
(fill-in-the-blank) tasks.

**Key files:**
- `llamawrapper.py` — Core Llama-2 wrapper with hooks for extracting intermediate activations; supports AttnWrapper for activation capture and LlamaHelper for logit lens decoding
- `utils.py` — Utility functions for language detection, plotting, and analysis
- `Translation.ipynb` — Notebook for translation task logit lens experiments
- `Cloze.ipynb` — Notebook for cloze task logit lens experiments
- `R/embed_unembed.r` — R scripts for statistical analysis and plotting
- `data/` — Pre-built language lists and datasets
- `plot_grid.sh` — Shell script for generating visualization grids

**Dependencies:** `requirements.txt` — PyTorch, transformers, accelerate, bitsandbytes, papermill, seaborn, scipy

**Usage:**
```bash
pip install -r requirements.txt
# Translation task (e.g., French to Chinese)
papermill Translation.ipynb out.ipynb -p input_lang fr -p target_lang zh
# Cloze task (e.g., French)
papermill Cloze.ipynb out.ipynb -p target_lang fr
```

**Precomputed data:** Available on [HuggingFace](https://huggingface.co/datasets/wendlerc/llm-latent-language)

---

## 2. `multilingual-analysis/` — PLND Neuron Detection & Multilingual Workflow

**Paper:** "How do Large Language Models Handle Multilingualism?"
(Zhao et al., NeurIPS 2024, DAMO-NLP-SG) — [arXiv:2402.18815](https://arxiv.org/abs/2402.18815)

**What it does:** Proposes a multilingual workflow (MWork) hypothesis where LLMs process
non-English input through: (1) understanding (language-specific neurons), (2) task-reasoning
(language-agnostic), and (3) generation (language-specific neurons). Provides the PLND
(Parallel Language-specific Neuron Detection) method to identify language-specific neurons
in attention and FFN layers, then validates via deactivation and enhancement experiments.

**Key directories:**
- `layers/` — Layer embedding decoding: decodes hidden states at each layer to vocabulary tokens
  - `test_layer.py` — Main script; requires setting `candidate_premature_layers` for the target model
  - `transformers/` — Modified transformers package files (must replace in installed package)
- `neuron_detection/` — PLND neuron detection implementation
  - `neuron_detection.py` — Main detection script; processes corpus text and extracts top-k activated neurons per layer
  - `corpus_all/` — Detection corpora (english.txt, chinese.txt, french.txt, russian.txt)
  - `transformers/` — Modified model files for Llama, Mistral, Gemma2
- `neuron_deactivate/` — Experiments deactivating detected neurons to measure impact
- `neuron_enhancement/` — Neuron-specific fine-tuning to improve multilingual performance
  - `train_neuron.py` — Training script for targeted neuron enhancement

**Supported models:** Llama-3-8B, Mistral, Gemma2 (modified transformers files provided for each)

**Dependencies:** `requirement.txt` — pandas, datasets, torch, vllm, cupy, openai, anthropic, google-generativeai

**Usage:**
```bash
pip install -r requirement.txt
# Layer decoding
cd layers && python test_layer.py
# Neuron detection (language, num_documents)
cd neuron_detection && python neuron_detection.py english 1000
# Neuron deactivation
cd neuron_deactivate && python test_mistral_gsm.py {language} {under_layer} {gen_layer} ...
# Neuron enhancement
cd neuron_enhancement && python train_neuron.py
```

**Important note:** This repo requires replacing files in the installed transformers package
with modified versions from the `transformers/` subdirectories. Each component (layers,
detection, deactivation, enhancement) has its own modified transformers files.

---

## 3. `self-translate/` — Self-Translate Evaluation Method

**Paper:** "Do Multilingual Language Models Think Better in English?"
(Etxaniz et al., 2023, UPV/EHU) — [arXiv:2308.01223](https://arxiv.org/abs/2308.01223)

**What it does:** Introduces a "self-translate" method where the multilingual model itself
translates non-English inputs to English before solving the task. Compares three approaches:
direct (original language), self-translate (model's own translation), and MT (external NLLB
translation). Evaluates on XCOPA, XStoryCloze, PAWS-X, XNLI, and MGSM across models
of different scales (XGLM, BLOOM, LLaMA, RedPajama, Open LLaMA).

**Key directories:**
- `translate/` — Translation pipeline
  - `scripts/` — Per-model translation scripts (bloom, llama, nllb, open_llama, redpajama, xglm)
  - `evaluation/` — Translation quality evaluation (BLEU, COMET, chrF++)
    - `evaluate_translations.py` — Computes metrics
    - `metrics/` — Pre-computed translation quality metrics per model/dataset
- `lm_eval/` — LM Evaluation Harness integration
  - `scripts/` — Per-model evaluation scripts for direct, self-translate, and MT methods
  - `results/` — Pre-computed evaluation results per model/dataset/method
  - Requires cloning the fork: `https://github.com/juletx/lm-evaluation-harness` (branch: translation)

**Key scripts:**
- `translate/scripts/xglm/translate_dataset_few_shot.py` — Few-shot self-translation
- `translate/scripts/nllb/translate_dataset_nllb.py` — NLLB-based MT translation
- `translate/evaluation/evaluate_translations.py` — Compute translation quality metrics

**Dependencies:** `requirements.txt` — accelerate, datasets, transformers 4.30.2, torch 2.0.0, evaluate, sacrebleu, comet, peft, scikit-learn

**Usage:**
```bash
pip install -r requirements.txt
# Install evaluation harness fork
cd lm_eval && git clone -b translation https://github.com/juletx/lm-evaluation-harness
cd lm-evaluation-harness && pip install -e .

# Self-translate a dataset
accelerate launch --mixed_precision fp16 translate_dataset_few_shot.py \
  --dataset xstory_cloze --target_lang "eng_Latn" \
  --model_name "facebook/xglm-7.5B" --starting_batch_size 128

# Evaluate with LM Evaluation Harness
python3 lm-evaluation-harness/main.py \
  --model hf-causal-experimental \
  --model_args pretrained=facebook/xglm-7.5B,use_accelerate=True \
  --tasks xcopa-mt_xglm-7.5B_* --device cuda --batch_size auto
```

**Pre-computed results:** Translation metrics and evaluation results are included in the repo.
Translated datasets available on [HuggingFace](https://huggingface.co/juletxara).

---

## Cross-Repo Connections

These three repos are complementary and address the same core question from different angles:

| Aspect | llm-latent-language | multilingual-analysis | self-translate |
|--------|--------------------|-----------------------|----------------|
| **Approach** | Logit lens (observational) | Neuron detection (mechanistic) | Task evaluation (behavioral) |
| **Key finding** | LLMs convert to English in middle layers | Language-specific neurons exist in early/late layers | Models perform better when self-translating to English |
| **Models** | Llama-2 (7B, 13B, 70B) | Llama-3-8B, Mistral, Gemma2 | XGLM, BLOOM, LLaMA, RedPajama, Open LLaMA |
| **GPU needed** | Yes (bitsandbytes quantization supported) | Yes (full precision) | Yes (accelerate + fp16) |

Together they provide evidence that multilingual LLMs have an English-centric internal
representation, with observable language-specific neurons, measurable latent language shifts,
and demonstrable performance benefits from translating to English before reasoning.
