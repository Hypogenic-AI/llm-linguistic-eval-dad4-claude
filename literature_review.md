# Literature Review: Evaluating Linguistic Performance in LLMs

## Research Area Overview

Large language models (LLMs) trained predominantly on English data have demonstrated surprising multilingual capabilities, yet exhibit significant performance disparities across languages. This review synthesizes research on three interconnected themes: (1) how LLMs internally process multilingual inputs, (2) the extent of performance gaps across languages, and (3) methods to evaluate and improve multilingual LLM capabilities. The central hypothesis under investigation is that English-dominant training induces implicit internal translation mechanisms that can be detected and measured, with implications for equitable multilingual deployment.

## Key Papers

### 1. Wendler et al. (2024) — "Do Llamas Work in English? On the Latent Language of Multilingual Transformers"
- **Source**: EPFL, arXiv:2402.10588 (233 citations)
- **Key Contribution**: First empirical investigation of whether LLMs use English as an internal pivot language. Using logit lens on Llama-2 (7B/13B/70B), they identify three processing phases.
- **Methodology**: Carefully constructed prompts (translation, repetition, cloze tasks) where the correct next token is unambiguous and language-attributable. Applied logit lens to decode intermediate layer representations. Tested on Chinese, German, French, Russian.
- **Key Results**:
  - **Phase 1** (early layers): High entropy, no language dominates — model builds feature representations.
  - **Phase 2** (middle layers): Low entropy, English dominates — model operates in "concept space" biased toward English.
  - **Phase 3** (final layers): Low entropy, target language dominates — model maps concepts to language-specific tokens.
- **Nuanced Finding**: The model's internal lingua franca is not strictly English but rather *concepts biased toward English*. The "concept space" is partially orthogonal to token space.
- **Code**: https://github.com/epfl-dlab/llm-latent-language
- **Relevance**: Directly confirms the hypothesis about implicit internal translation. The three-phase model provides a concrete framework for understanding multilingual processing.

### 2. Zhang/Zhao et al. (2024) — "How do Large Language Models Handle Multilingualism?" (NeurIPS 2024)
- **Source**: NUS & Alibaba DAMO, arXiv:2402.18815 (142 citations)
- **Key Contribution**: Proposes MWork (Multilingual Workflow) hypothesis and PLND (Parallel Language-specific Neuron Detection) method.
- **Methodology**: Decoded hidden embeddings layer-by-layer, classified into English/non-English. Developed PLND to identify language-specific neurons without labeled data. Verified MWork through selective neuron deactivation experiments on Vicuna and Mistral.
- **MWork Hypothesis**: LLMs process multilingual inputs in three stages:
  1. **Understand**: Convert multilingual input to unified representation
  2. **Task-solve**: Reason in English (self-attention) + extract multilingual knowledge (feed-forward)
  3. **Generate**: Produce output in original language
- **Key Results**: Deactivating just 0.13% of language-specific neurons drops multilingual performance by 99%. Self-attention neurons decrease in task-solving layers (English reasoning), while feed-forward neurons remain consistent (multilingual knowledge storage).
- **Code**: https://github.com/DAMO-NLP-SG/multilingual_analysis
- **Relevance**: Provides mechanistic verification that LLMs reason internally in English while maintaining language-specific neurons for multilingual I/O.

### 3. Etxaniz et al. (2023) — "Do Multilingual Language Models Think Better in English?"
- **Source**: UPV/EHU & Reka AI, arXiv:2308.01223
- **Key Contribution**: Introduces "self-translate" — using the LLM itself to translate input to English before solving tasks, proving LLMs can't fully leverage capabilities when prompted in non-English.
- **Methodology**: Compared direct inference vs. self-translate across XGLM (0.6B–7.5B) and LLaMA (7B–30B) on 5 tasks (XCOPA, XStoryCloze, XNLI, PAWS-X, MGSM).
- **Key Results**: Self-translate consistently outperforms direct inference (avg +2–3.5 points). Effect is more pronounced for larger models and high-resource languages. External MT still outperforms self-translate but the gap narrows at scale.
- **Code**: https://github.com/juletx/self-translate
- **Relevance**: Behavioral evidence that LLMs are more capable than they appear in non-English — internal processing bottleneck exists.

### 4. Huang et al. (2023) — "Not All Languages Are Created Equal in LLMs"
- **Source**: Microsoft Research Asia, arXiv:2305.07004 (230 citations)
- **Key Contribution**: Cross-Lingual-Thought (XLT) prompting template that systematically improves multilingual LLM capability by guiding cross-lingual reasoning.
- **Methodology**: XLT template includes: role assignment, task input, cross-lingual thinking (retell in English), task analysis, CoT solving, output formatting. Evaluated on 7 benchmarks, 27 languages.
- **Datasets Used**: MGSM, XCOPA, XNLI, PAWS-X, MKQA, XL-Sum, FLORES-200
- **Key Results**: XLT achieves >10 point improvement on MGSM and MKQA. Introduces "democratization score" — XLT reduces performance gap between best and average language performance.
- **Relevance**: Demonstrates that explicitly leveraging English as pivot in prompting improves multilingual performance, supporting the internal translation hypothesis.

### 5. Ahuja et al. (2023) — "MEGA: Multilingual Evaluation of Generative AI"
- **Source**: Microsoft, arXiv:2303.12528 (354 citations)
- **Key Contribution**: First comprehensive multilingual benchmarking of generative LLMs across 16 datasets, 70 languages, 4 LLMs.
- **Methodology**: Evaluated GPT-3.5 (text-davinci-003, gpt-3.5-turbo), GPT-4, BLOOMZ on classification, QA, sequence labeling, generation, and responsible AI tasks. Compared with fine-tuned SOTA models.
- **Key Results**: Significant English vs. non-English performance gap, especially for low-resource languages with non-Latin scripts. GPT-4 narrows but doesn't close the gap. Translate-test often outperforms direct multilingual prompting for low-resource languages.
- **Relevance**: Provides the evaluation methodology and baseline results against which multilingual improvements can be measured.

### 6. Shi et al. (2022) — "Language Models are Multilingual Chain-of-Thought Reasoners"
- **Source**: Google, arXiv:2210.03057 (521 citations)
- **Key Contribution**: Introduces MGSM benchmark (multilingual grade school math, 11 languages). Shows English CoT prompting works across languages.
- **Datasets**: MGSM — 250 problems per language × 11 languages
- **Key Results**: Using English CoT on non-English inputs significantly outperforms native-language CoT, providing early evidence for English as effective pivot language.
- **Relevance**: Key benchmark for multilingual reasoning evaluation; finding that English CoT works cross-lingually supports the internal pivot hypothesis.

### 7. Bandarkar et al. (2023) — "The Belebele Benchmark"
- **Source**: Meta, arXiv:2308.16884 (250 citations)
- **Key Contribution**: Parallel reading comprehension dataset in 122 language variants covering high, medium, and low-resource languages.
- **Methodology**: 900 questions based on FLORES-200 passages, professionally translated with quality control. Multiple-choice format.
- **Relevance**: Most linguistically diverse benchmark available; enables fine-grained comparison across language families and resource levels.

### 8. Hu et al. (2020) — "XTREME: A Massively Multilingual Multi-task Benchmark"
- **Source**: Google/CMU, arXiv:2003.11080 (1092 citations)
- **Key Contribution**: Foundational multilingual benchmark covering 9 tasks across 40 languages.
- **Tasks**: Sentence classification, structured prediction, QA, sentence retrieval
- **Relevance**: Standard benchmark for cross-lingual transfer evaluation; widely used as baseline for multilingual model comparison.

## Common Methodologies

### Evaluation Approaches
1. **Logit Lens / Probing**: Decoding intermediate layer representations to identify internal language (Wendler et al., 2024; Zhang et al., 2024)
2. **Translate-Test**: Translating non-English inputs to English before inference (Shi et al., 2022; Ahuja et al., 2023)
3. **Self-Translate**: Using the LLM itself to translate before solving (Etxaniz et al., 2023)
4. **Cross-Lingual Prompting**: English-language prompts for non-English inputs (Huang et al., 2023)
5. **Neuron Detection/Deactivation**: Identifying and ablating language-specific neurons (Zhang et al., 2024)
6. **Parallel Benchmarking**: Using semantically equivalent test sets across languages (Belebele, XNLI, MGSM)

### Models Commonly Evaluated
- **Llama-2 family** (7B/13B/70B): Most studied for internal mechanisms due to open weights
- **GPT-3.5 / GPT-4**: Strongest multilingual performance but closed-source
- **BLOOMZ**: Open-source, explicitly multilingual (46 languages)
- **Mistral / Vicuna**: Open-source models used for mechanistic analysis
- **XLM-RoBERTa / mBERT / mT5**: Encoder models for cross-lingual transfer baselines

## Standard Baselines
- **Zero-shot cross-lingual transfer**: Fine-tune on English, evaluate on other languages
- **Translate-test with external MT**: NLLB-200, Google Translate
- **Few-shot in-context learning**: Multilingual demonstrations
- **Chain-of-thought prompting**: English CoT for multilingual reasoning

## Evaluation Metrics
- **Accuracy**: Classification, QA, reasoning tasks (XNLI, XCOPA, MGSM, Belebele)
- **F1 Score**: Span prediction QA (XQuAD, MLQA)
- **ROUGE-L**: Summarization (XLSum)
- **BLEU/SacreBLEU**: Machine translation (FLORES)
- **Democratization Score**: Ratio of average to best language performance (Huang et al., 2023)

## Datasets in the Literature
| Dataset | Languages | Task | Used In |
|---------|-----------|------|---------|
| MGSM | 11 | Math reasoning | Shi 2022, Huang 2023, Etxaniz 2023, Zhang 2024 |
| XNLI | 15 | NLI | Huang 2023, Ahuja 2023, Etxaniz 2023 |
| XCOPA | 11 | Commonsense reasoning | Huang 2023, Ahuja 2023, Etxaniz 2023 |
| Belebele | 122 | Reading comprehension | Bandarkar 2023 |
| PAWS-X | 7 | Paraphrase identification | Huang 2023, Ahuja 2023 |
| XQuAD | 10 | Span QA | Zhang 2024, Ahuja 2023 |
| MLQA | 7 | Cross-lingual QA | Ahuja 2023 |
| XLSum | 44 | Summarization | Huang 2023, Zhang 2024 |
| FLORES-200 | 204 | Translation | Huang 2023 |
| MMLU-ProX | Multiple | Knowledge evaluation | Son 2024 |

## Gaps and Opportunities

1. **Mechanistic studies limited to Llama-2**: Wendler et al. show initial evidence that Mistral behaves similarly, but systematic study across model families (Qwen, Gemma, etc.) is needed.
2. **Tasks are mostly simple**: Current probing uses translation and cloze tasks. More complex reasoning, cultural knowledge, and generation tasks remain unexplored.
3. **Low-resource languages under-studied**: Most experiments focus on well-resourced languages (Chinese, German, French, Spanish). The behavior of the "English pivot" for truly low-resource languages (Swahili, Yoruba, etc.) is less understood.
4. **Causal vs. correlational evidence**: The three-phase model is descriptive. Causal interventions (e.g., activation patching) could provide stronger evidence about whether English pivoting is *necessary* or merely a *side effect*.
5. **Impact on downstream bias**: If LLMs reason through English-biased concepts, this could propagate Anglocentric biases into outputs across all languages — a largely unexplored area.

## Recommendations for Our Experiment

Based on this literature review:

- **Recommended datasets**: MGSM (reasoning across languages), XNLI (understanding), Belebele (reading comprehension with widest language coverage), XCOPA (commonsense)
- **Recommended baselines**: Direct inference, translate-test (external MT), self-translate, English CoT prompting, XLT prompting
- **Recommended metrics**: Accuracy (primary), democratization score (cross-language equity), per-language performance breakdown
- **Recommended models**: At minimum one open-source model (Llama family for mechanistic analysis) and one API model (GPT-4 or Claude for performance benchmarking)
- **Methodological considerations**:
  - Use parallel benchmarks where possible to ensure fair cross-language comparison
  - Include both high-resource (German, French, Chinese) and low-resource (Swahili, Bengali) languages
  - Consider logit lens analysis to detect internal translation if using open-weight models
  - Report per-language results, not just averages, to capture performance disparities
