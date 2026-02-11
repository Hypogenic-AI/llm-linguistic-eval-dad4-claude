# Resources: Evaluating Linguistic Performance in LLMs

Comprehensive catalog of papers, datasets, and code repositories gathered for this research project.

## Research Hypothesis

LLMs trained predominantly on English exhibit reduced performance in non-English languages, and evaluating performance across multiple languages reveals implicit internal translation mechanisms where models convert non-English input to English-like representations before processing.

---

## Papers (20 papers in `papers/`)

### Core Papers — Internal Translation Mechanisms

| # | File | Authors | Year | arXiv | Key Contribution |
|---|------|---------|------|-------|-----------------|
| 1 | `wendler2024_do_llamas_work_in_english.pdf` | Wendler, Veselovsky, Monea, West | 2024 | 2402.10588 | Logit lens reveals three-phase processing: input → English-biased concept space → output language |
| 2 | `zhang2024_how_llms_handle_multilingualism.pdf` | Zhao, Zhang, Chen, Kawaguchi, Bing | 2024 | 2402.18815 | MWork hypothesis + PLND neuron detection; deactivating 0.13% of neurons drops multilingual performance 99% |
| 3 | `etxaniz2023_multilingual_llm_prompts.pdf` | Etxaniz, Azkune, Soroa, Lopez de Lacalle, Artetxe | 2023 | 2308.01223 | Self-translate method shows LLMs perform better when translating to English first |

### Core Papers — Multilingual Evaluation & Prompting

| # | File | Authors | Year | arXiv | Key Contribution |
|---|------|---------|------|-------|-----------------|
| 4 | `lai2023_chatgpt_beyond_english.pdf` | Lai et al. | 2023 | 2304.04675 | Comprehensive multilingual evaluation of ChatGPT |
| 5 | `ahuja2023_mega.pdf` | Ahuja, Diddee, Hada et al. | 2023 | 2303.12528 | MEGA benchmark: 16 datasets, 70 languages, 4 LLMs; quantifies English vs non-English gaps |
| 6 | `huang2023_not_all_languages_equal.pdf` | Huang, Tang, Zhang et al. | 2023 | 2305.07004 | Cross-Lingual-Thought (XLT) prompting; >10pt gains; introduces democratization score |
| 7 | `shi2023_is_translation_all_you_need.pdf` | Etxaniz et al. | 2023 | 2308.01223 | Self-translate approach for multilingual tasks |
| 8 | `shi2022_multilingual_cot_reasoners.pdf` | Shi et al. | 2022 | 2210.03057 | Introduces MGSM benchmark; English CoT works cross-lingually |

### Benchmark & Infrastructure Papers

| # | File | Authors | Year | arXiv | Key Contribution |
|---|------|---------|------|-------|-----------------|
| 9 | `hu2020_xtreme.pdf` | Hu et al. | 2020 | 2003.11080 | XTREME: 9 tasks, 40 languages — foundational multilingual benchmark |
| 10 | `bandarkar2023_belebele.pdf` | Bandarkar et al. | 2023 | 2308.16884 | Belebele: reading comprehension in 122 language variants |
| 11 | `son2024_mmlu_prox.pdf` | Son et al. | 2024 | 2406.04264 | MMLU-ProX: multilingual knowledge evaluation |
| 12 | `conneau2020_xlm_roberta.pdf` | Conneau et al. | 2020 | 1911.02116 | XLM-RoBERTa: foundational cross-lingual transfer model |

### Cross-lingual Transfer & Analysis Papers

| # | File | Authors | Year | arXiv | Key Contribution |
|---|------|---------|------|-------|-----------------|
| 13 | `asai2023_buffet.pdf` | Asai et al. | 2023 | 2305.14857 | BUFFET: few-shot cross-lingual transfer benchmark |
| 14 | `wendler2024_turning_english_centric_polyglots.pdf` | Wendler et al. | 2024 | 2405.06089 | How much multilingual data is needed for English-centric LLMs |
| 15 | `ranjan2023_democratizing_llms.pdf` | Ranjan et al. | 2023 | 2306.11837 | Linguistically-diverse prompts for low-resource languages |
| 16 | `chen2023_plug_pivot_language.pdf` | Chen et al. | 2023 | 2311.08711 | PLUG: pivot language in cross-lingual instruction tuning |
| 17 | `min2024_beyond_english_prompting.pdf` | Min et al. | 2024 | 2401.07164 | Prompt translation strategies across languages |
| 18 | `li2024_crosslingual_capabilities_barriers.pdf` | Li et al. | 2024 | 2406.01581 | Knowledge barriers in multilingual LLMs |
| 19 | `keleg2023_cross_lingual_knowledge_eval.pdf` | Keleg et al. | 2023 | 2305.12679 | Methods for evaluating cross-lingual knowledge transfer |
| 20 | `lee2024_probing_crosslingual_alignment.pdf` | Lee et al. | 2024 | 2404.18397 | How cross-lingual alignment emerges during training |

---

## Datasets (in `datasets/`)

| Dataset | Task | Languages | Size | Format | Directory |
|---------|------|-----------|------|--------|-----------|
| **MGSM** | Math reasoning | 11 (bn, de, en, es, fr, ja, ru, sw, te, th, zh) | 250/lang | TSV | `mgsm/` |
| **XNLI** | Natural language inference | 15 | 392K train, 5K test | Arrow | `xnli/` |
| **XCOPA** | Commonsense reasoning | 1 (Turkish; 11 available) | 100 val, 500 test | Arrow | `xcopa_tr/` |
| **Belebele** | Reading comprehension | 10 (eng, zho, deu, fra, rus, hin, spa, jpn, arb, swh) | 900/lang | Arrow | `belebele_*/` |

### Language Coverage Matrix

| Language | MGSM | XNLI | Belebele | Script |
|----------|------|------|----------|--------|
| English | x | x | x | Latin |
| Chinese | x | x | x (Simplified) | CJK/Hanzi |
| German | x | - | x | Latin |
| French | x | x | x | Latin |
| Spanish | x | x | x | Latin |
| Russian | x | x | x | Cyrillic |
| Japanese | x | - | x | CJK/Kana |
| Hindi | - | x | x | Devanagari |
| Arabic | - | x | x | Arabic |
| Swahili | x | x | x | Latin |
| Bengali | x | - | - | Bengali |
| Telugu | x | - | - | Telugu |
| Thai | x | x | - | Thai |
| Turkish | - | x | - | Latin |

### Download Instructions

See `datasets/README.md` for detailed download/reproduction instructions for each dataset.

---

## Code Repositories (in `code/`)

| Repository | Paper | Approach | Models | Key Technique |
|-----------|-------|----------|--------|--------------|
| `llm-latent-language/` | Wendler et al. 2024 | Observational | Llama-2 (7B/13B/70B) | Logit lens decoding of intermediate layers |
| `multilingual-analysis/` | Zhao et al. 2024 | Mechanistic | Llama-3-8B, Mistral, Gemma2 | PLND neuron detection + deactivation |
| `self-translate/` | Etxaniz et al. 2023 | Behavioral | XGLM, BLOOM, LLaMA | Self-translate vs direct vs MT evaluation |

### Repository Details

**1. llm-latent-language** (Wendler et al.)
- Source: https://github.com/epfl-dlab/llm-latent-language
- Entry points: `Translation.ipynb`, `Cloze.ipynb`
- Requires: GPU with bitsandbytes quantization support

**2. multilingual-analysis** (Zhao/Zhang et al.)
- Source: https://github.com/DAMO-NLP-SG/multilingual_analysis
- Entry points: `neuron_detection/neuron_detection.py`, `layers/test_layer.py`
- Note: Requires replacing installed transformers package files with modified versions

**3. self-translate** (Etxaniz et al.)
- Source: https://github.com/juletx/self-translate
- Entry points: `translate/scripts/*/translate_dataset_few_shot.py`
- Includes pre-computed results for all model/dataset/method combinations

See `code/README.md` for detailed usage instructions and cross-repo comparisons.

---

## Recommended Experiment Design

Based on the gathered resources, the following experiment design is recommended:

### Models to Evaluate
- **Open-source (mechanistic analysis)**: Llama-2 or Llama-3 (logit lens + neuron analysis possible)
- **API models (performance benchmarking)**: GPT-4, Claude (strongest multilingual capabilities)

### Evaluation Pipeline
1. **Direct inference**: Prompt model in target language, measure accuracy
2. **Self-translate**: Use model to translate input to English, then solve
3. **External MT + solve**: Translate with NLLB-200, then solve in English
4. **XLT prompting**: Apply Cross-Lingual-Thought template
5. **English CoT**: Use English chain-of-thought on non-English inputs

### Metrics
- **Primary**: Accuracy per language per task
- **Cross-lingual equity**: Democratization score (avg/best language ratio)
- **Performance gap**: English accuracy minus per-language accuracy
- **Internal analysis** (open-source models only): Logit lens language distribution per layer

### Language Selection
- **High-resource**: English, Chinese, German, French, Spanish, Russian
- **Medium-resource**: Japanese, Hindi, Arabic
- **Low-resource**: Swahili, Bengali, Telugu

### Task Coverage
| Task Type | Dataset | Metric |
|-----------|---------|--------|
| Mathematical reasoning | MGSM | Accuracy |
| Natural language inference | XNLI | Accuracy |
| Reading comprehension | Belebele | Accuracy |
| Commonsense reasoning | XCOPA | Accuracy |

---

## Key Findings from Literature

1. **Internal English pivot confirmed**: Multiple independent studies show LLMs convert non-English input to English-biased internal representations in middle layers (Wendler 2024, Zhang 2024)
2. **Language-specific neurons exist**: Only 0.13% of neurons are language-specific, but their deactivation destroys multilingual capability (Zhang 2024)
3. **Self-translation helps**: LLMs consistently perform better when explicitly translating to English first, proving internal processing bottleneck (Etxaniz 2023)
4. **English CoT transfers**: English chain-of-thought prompting improves reasoning in all tested languages (Shi 2022)
5. **Low-resource gap persists**: Even GPT-4 shows significant performance drops for low-resource languages with non-Latin scripts (Ahuja 2023)
6. **XLT prompting democratizes**: Cross-Lingual-Thought prompting reduces the gap between best and worst language performance (Huang 2023)

---

## Gaps and Open Questions

1. Limited mechanistic studies beyond Llama-2 — need systematic comparison across model families
2. Probing tasks are mostly simple (translation, cloze) — complex reasoning and generation unexplored
3. Low-resource languages severely under-studied in mechanistic work
4. Causal evidence (activation patching) needed to determine if English pivoting is necessary vs. incidental
5. Potential Anglocentric bias propagation through English-biased internal representations unexplored
