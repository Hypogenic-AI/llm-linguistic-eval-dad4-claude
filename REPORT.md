# Evaluating Linguistic Performance in LLMs: Cross-Language Equity and Implicit Translation Mechanisms

## Executive Summary

We evaluated two frontier LLMs — GPT-4.1 (OpenAI) and Claude Sonnet 4 (Anthropic) — across 8 languages, 3 prompting strategies, and 2 benchmarks (MGSM math reasoning, Belebele reading comprehension) to test whether these models exhibit performance degradation on non-English tasks and whether explicit English-pivoting strategies provide evidence for implicit internal translation mechanisms. Our experiments comprised **4,800 real API calls** across all conditions.

**Key Findings:**

1. **Claude Sonnet 4 shows remarkable multilingual parity.** On MGSM direct inference, Claude achieves 0.92 accuracy on English and *higher* accuracy on several non-English languages (German: 0.98, Chinese: 0.96, Bengali: 0.94). The English-to-non-English performance gap is not statistically significant.

2. **GPT-4.1 shows a dramatic confound, not a language gap.** GPT-4.1's "direct" MGSM prompt (no chain-of-thought) yields low accuracy across all languages (English: 0.64), while self-translate and English CoT dramatically improve performance. This reflects a reasoning-strategy effect, not a translation effect.

3. **English-pivoting provides no benefit for Claude Sonnet 4** on either benchmark. Self-translate and English CoT do not improve non-English accuracy, suggesting Claude's internal multilingual processing is already efficient.

4. **Hindi/Belebele is the consistent weak spot.** Across both models and all strategies, Hindi consistently shows the lowest Belebele accuracy (0.80-0.90), suggesting a genuine low-resource penalty for this specific task-language combination.

5. **Democratization scores are high** (0.86-0.97), with Claude achieving near-perfect cross-language equity (0.97 on Belebele direct).

---

## 1. Introduction

### 1.1 Motivation

Large Language Models are increasingly deployed at national scale in non-English-speaking countries. Notable examples include xAI's partnership with Venezuela and OpenAI's collaboration with Estonia. Yet capability evaluations remain predominantly English-centric, raising the question: are these models equitable across languages?

Prior work has demonstrated that LLMs use English-biased internal representations (Wendler et al., 2024; Zhang et al., 2024), that self-translation to English improves performance (Etxaniz et al., 2023), and that English chain-of-thought reasoning transfers cross-lingually (Shi et al., 2022). However, these findings were established primarily on earlier models (GPT-3.5/4, Llama-2). Whether these patterns persist in 2025/2026 frontier models is an open empirical question.

### 1.2 Research Questions

**RQ1:** Do state-of-the-art LLMs exhibit significant performance degradation on non-English tasks compared to English?

**RQ2:** Does explicit English-pivoting (self-translate, English CoT) improve non-English performance, providing behavioral evidence for implicit internal translation?

**RQ3:** Does performance degradation correlate with language resource level?

**RQ4:** Do different models show different multilingual capability profiles?

### 1.3 Hypotheses

- **H1:** LLMs show statistically significant performance degradation on non-English vs. English tasks.
- **H2:** Explicit English-pivoting strategies improve non-English performance.
- **H3:** Performance degradation correlates with language resource level.
- **H4:** Different models show different multilingual capability profiles.

---

## 2. Methodology

### 2.1 Experimental Design

We use a **2 (models) x 3 (strategies) x 8 (languages) x 2 (tasks)** factorial design with 50 randomly sampled items per language per task (seed=42). Total: **4,800 API calls**.

### 2.2 Models

| Model | Provider | Access Method | Temperature |
|-------|----------|---------------|-------------|
| GPT-4.1 | OpenAI | Direct API | 0 |
| Claude Sonnet 4 | Anthropic | OpenRouter | 0 |

### 2.3 Benchmarks

**MGSM (Multilingual Grade School Math):** 250 math word problems per language. Tests mathematical reasoning. Metric: exact match on final integer answer.

**Belebele:** 900 reading comprehension questions per language with 4 multiple-choice options. Tests passage understanding. Metric: correct option selection (1-4).

### 2.4 Languages (8 languages across 3 resource tiers)

| Resource Level | MGSM Languages | Belebele Languages |
|---------------|----------------|-------------------|
| High | English, Chinese, German, French | English, Chinese, German, French |
| Medium | Russian, Japanese | Russian, Japanese |
| Low | Swahili, Bengali | Swahili, Hindi |

### 2.5 Prompting Strategies

1. **Direct:** Solve the problem in its original language. No chain-of-thought. Just return the answer.
2. **Self-Translate:** Translate the problem to English first, then solve step-by-step. Return the final answer.
3. **English CoT:** Reason step-by-step in English about the problem (presented in original language). Return the final answer.

**Important design note:** The "direct" prompt asks for only the final answer with no reasoning steps, while both self-translate and English CoT include step-by-step reasoning. This means any lift from these strategies includes both (a) a translation/language effect and (b) a chain-of-thought reasoning effect. Comparing self-translate to English CoT on non-English inputs isolates the translation component.

### 2.6 Statistical Analysis

- **H1:** One-sample t-test on performance gaps (English - target), with Cohen's d effect sizes
- **H2:** One-sample t-test on strategy lifts (strategy - direct), one-sided
- **H3:** Spearman rank correlation between resource level and accuracy
- **Significance level:** alpha = 0.05

---

## 3. Results

### 3.1 Overall Accuracy

#### MGSM (Math Reasoning)

| Model | Strategy | EN | ZH | DE | FR | RU | JA | SW | BN | **Avg** |
|-------|----------|---:|---:|---:|---:|---:|---:|---:|---:|--------:|
| GPT-4.1 | direct | .640 | .520 | .560 | .680 | .700 | .500 | .640 | .560 | **.600** |
| GPT-4.1 | self_translate | .600 | .940 | .960 | .860 | .960 | .860 | .820 | .900 | **.862** |
| GPT-4.1 | english_cot | .940 | .940 | .960 | .900 | .940 | .880 | .840 | .920 | **.915** |
| Claude Sonnet 4 | direct | .920 | .960 | .980 | .940 | .940 | .900 | .900 | .940 | **.935** |
| Claude Sonnet 4 | self_translate | .920 | .940 | .960 | .940 | .940 | .920 | .880 | .940 | **.930** |
| Claude Sonnet 4 | english_cot | 1.000 | .960 | .940 | .960 | .960 | .900 | .840 | .940 | **.938** |

#### Belebele (Reading Comprehension)

| Model | Strategy | EN | ZH | DE | FR | RU | JP | SW | HI | **Avg** |
|-------|----------|---:|---:|---:|---:|---:|---:|---:|---:|--------:|
| GPT-4.1 | direct | 1.000 | .920 | .920 | .940 | .980 | .940 | .980 | .820 | **.938** |
| GPT-4.1 | self_translate | 1.000 | .960 | .940 | .940 | .980 | .980 | .980 | .800 | **.948** |
| GPT-4.1 | english_cot | 1.000 | .940 | .960 | .980 | .980 | .940 | .980 | .820 | **.950** |
| Claude Sonnet 4 | direct | 1.000 | 1.000 | .960 | .940 | 1.000 | .960 | 1.000 | .900 | **.970** |
| Claude Sonnet 4 | self_translate | 1.000 | 1.000 | .960 | .940 | 1.000 | .960 | 1.000 | .880 | **.968** |
| Claude Sonnet 4 | english_cot | 1.000 | 1.000 | .960 | .960 | .980 | .980 | .960 | .880 | **.965** |

### 3.2 H1: Performance Gap Between English and Non-English

#### Statistical Tests (one-sample t-test, H0: gap = 0)

| Task | Model | Strategy | Mean Gap | t-stat | p-value | Sig | Cohen's d |
|------|-------|----------|---------|--------|---------|-----|-----------|
| MGSM | GPT-4.1 | direct | 0.046 | 1.53 | 0.176 | ns | 0.58 |
| MGSM | GPT-4.1 | self_translate | **-0.300** | -14.33 | <0.001 | *** | -5.42 |
| MGSM | GPT-4.1 | english_cot | 0.029 | 1.83 | 0.118 | ns | 0.69 |
| MGSM | Claude Sonnet 4 | direct | **-0.017** | -1.55 | 0.172 | ns | -0.59 |
| MGSM | Claude Sonnet 4 | self_translate | -0.011 | -1.19 | 0.280 | ns | -0.45 |
| MGSM | Claude Sonnet 4 | english_cot | **0.071** | 4.25 | 0.005 | ** | 1.60 |
| Belebele | GPT-4.1 | direct | **0.071** | 3.50 | 0.013 | * | 1.32 |
| Belebele | GPT-4.1 | self_translate | **0.060** | 2.47 | 0.049 | * | 0.93 |
| Belebele | GPT-4.1 | english_cot | **0.057** | 2.65 | 0.038 | * | 1.00 |
| Belebele | Claude Sonnet 4 | direct | 0.034 | 2.40 | 0.053 | ns | 0.91 |
| Belebele | Claude Sonnet 4 | self_translate | 0.037 | 2.24 | 0.066 | ns | 0.85 |
| Belebele | Claude Sonnet 4 | english_cot | **0.040** | 2.76 | 0.033 | * | 1.04 |

**Key findings for H1:**

- **GPT-4.1 MGSM self_translate shows a massive *negative* gap** (mean=-0.300, p<0.001): non-English languages dramatically outperform English. This is because the direct English prompt has no reasoning steps, yielding only 0.64 accuracy, while self-translate adds chain-of-thought reasoning that boosts all languages to ~0.86-0.96.
- **Claude Sonnet 4 on MGSM shows no significant English advantage** under direct or self-translate strategies. Non-English languages actually perform comparably or better than English.
- **GPT-4.1 on Belebele shows a small but significant English advantage** (mean gap 0.057-0.071, p<0.05), primarily driven by Hindi (0.82 vs English 1.00).
- **Claude Sonnet 4 on Belebele shows borderline significance** (p=0.033-0.066), with very small gaps.

**Verdict: H1 is partially supported.** Small English advantages exist on Belebele (primarily GPT-4.1), but MGSM shows no English advantage for Claude and an artifactual result for GPT-4.1. Modern frontier models show substantially reduced English bias compared to earlier generations.

### 3.3 H2: Effect of English-Pivoting Strategies

#### Non-English Language Lift (strategy - direct)

| Task | Model | Strategy | Mean Lift | t-stat | p (one-sided) | Sig |
|------|-------|----------|----------|--------|----------------|-----|
| MGSM | GPT-4.1 | self_translate | **+0.306** | 8.10 | <0.001 | *** |
| MGSM | GPT-4.1 | english_cot | **+0.317** | 8.98 | <0.001 | *** |
| MGSM | Claude Sonnet 4 | self_translate | -0.006 | -1.00 | 0.822 | ns |
| MGSM | Claude Sonnet 4 | english_cot | -0.009 | -0.75 | 0.759 | ns |
| Belebele | GPT-4.1 | self_translate | +0.011 | 1.33 | 0.115 | ns |
| Belebele | GPT-4.1 | english_cot | **+0.014** | 1.99 | 0.047 | * |
| Belebele | Claude Sonnet 4 | self_translate | -0.003 | -1.00 | 0.822 | ns |
| Belebele | Claude Sonnet 4 | english_cot | -0.006 | -0.68 | 0.739 | ns |

**Key findings for H2:**

- **GPT-4.1 MGSM shows enormous lift (+0.31)** from both strategies, but this is primarily a **chain-of-thought effect**, not a translation effect. Evidence: English itself improves from 0.64 (direct) to 0.94 (english_cot), a +0.30 lift. The self-translate lift for non-English languages (~0.31) is similar, suggesting the benefit comes from step-by-step reasoning, not from translation.
- **Comparing self-translate vs. english_cot for GPT-4.1 non-English MGSM** (isolating translation effect): Self-translate mean=0.90, English CoT mean=0.92. The difference is small (+0.02 for CoT), suggesting minimal additional benefit from explicit translation when reasoning is already in English.
- **Claude Sonnet 4 shows zero lift** from English-pivoting strategies on either benchmark. Direct inference already achieves 0.93-0.97 average accuracy, leaving little room for improvement.
- **Belebele shows minimal lift across both models**, likely because reading comprehension at this level is already near ceiling.

**Verdict: H2 is not supported as a translation effect.** The massive lift for GPT-4.1 MGSM is attributable to chain-of-thought reasoning, not to English translation. When controlling for reasoning (comparing self-translate vs. english_cot), the translation component is negligible. Claude Sonnet 4 shows no benefit from English pivoting at all.

### 3.4 H3: Resource Level Correlation

| Task | Model | Spearman rho | p-value | Sig |
|------|-------|-------------|---------|-----|
| MGSM | GPT-4.1 | -0.086 | 0.855 | ns |
| MGSM | Claude Sonnet 4 | 0.693 | 0.084 | ns |
| Belebele | GPT-4.1 | -0.496 | 0.258 | ns |
| Belebele | Claude Sonnet 4 | -0.222 | 0.632 | ns |

**Key findings for H3:**

- No significant correlation between language resource level and accuracy in any condition.
- For Belebele, low-resource languages (Swahili) actually perform *better* than some high-resource languages.
- The expected pattern (high > medium > low) is not consistently observed.

**Verdict: H3 is not supported.** Resource level does not predict performance in our data. This is notable — it suggests frontier models have made substantial progress in handling diverse languages.

### 3.5 H4: Model Differences

Claude Sonnet 4 substantially outperforms GPT-4.1 across most conditions:

**MGSM Direct (most revealing comparison):**
- Claude: avg=0.935 (range: 0.90-0.98)
- GPT-4.1: avg=0.600 (range: 0.50-0.70)

However, this comparison is misleading. GPT-4.1's low direct accuracy stems from its sensitivity to the "just give the number" prompt format. When given reasoning steps (English CoT), GPT-4.1 reaches avg=0.915 vs Claude's 0.938 — a much smaller gap.

**Belebele Direct (fairer comparison):**
- Claude: avg=0.970 (range: 0.90-1.00)
- GPT-4.1: avg=0.938 (range: 0.82-1.00)

Claude shows more consistent performance and higher accuracy, especially on Swahili (1.00 vs 0.98) and Hindi (0.90 vs 0.82).

**Verdict: H4 is supported.** The models have distinctly different multilingual profiles. Claude Sonnet 4 is more robust to language variation, maintains higher accuracy with minimal prompting, and shows virtually no English bias. GPT-4.1 is more sensitive to prompt format but performs well with appropriate scaffolding.

### 3.6 Democratization Scores

| Task | Model | Direct | Self-Translate | English CoT |
|------|-------|--------|---------------|-------------|
| MGSM | GPT-4.1 | 0.857 | 0.898 | 0.953 |
| MGSM | Claude Sonnet 4 | **0.954** | **0.969** | 0.938 |
| Belebele | GPT-4.1 | 0.938 | 0.948 | 0.950 |
| Belebele | Claude Sonnet 4 | **0.970** | **0.968** | **0.965** |

Claude Sonnet 4 achieves consistently higher democratization scores, meaning more equitable performance across languages. GPT-4.1's low MGSM direct score (0.857) reflects its format sensitivity rather than genuine language bias.

---

## 4. Discussion

### 4.1 The Disappearing English Advantage

Our most striking finding is that **frontier 2025-2026 models show minimal English advantage on standard benchmarks**. Claude Sonnet 4 actually performs comparably or better on several non-English languages (German, Chinese) than English for MGSM. This contrasts sharply with findings from Ahuja et al. (2023) showing 10-30% gaps for GPT-3.5/4.

This suggests that recent training improvements — likely including more balanced multilingual data and RLHF in multiple languages — have substantially narrowed the cross-language performance gap.

### 4.2 Chain-of-Thought, Not Translation

The massive lift from self-translate and English CoT for GPT-4.1 MGSM initially appears to support the implicit translation hypothesis. However, careful analysis reveals this is primarily a **chain-of-thought effect**: the same lift occurs for English itself (0.64 → 0.94). GPT-4.1 simply cannot reliably extract numerical answers without reasoning steps, regardless of language.

When we control for chain-of-thought (comparing self-translate vs. English CoT for non-English inputs), the translation component adds only ~2 percentage points. This is a much weaker signal than the 15-30% translation benefits reported for earlier models.

### 4.3 Hindi as the Persistent Challenge

Hindi consistently shows the lowest accuracy on Belebele across both models and all strategies (0.80-0.90). This is noteworthy because Hindi is classified as a "medium-resource" language. The difficulty may relate to the specific Belebele items, the Devanagari script processing, or the domain of the passages. Further investigation with larger samples would help clarify this pattern.

### 4.4 Implications for Deployment

1. **Claude Sonnet 4 is ready for multilingual deployment** without specialized prompting strategies. Direct inference achieves >0.90 accuracy across all tested languages.
2. **GPT-4.1 benefits from explicit reasoning prompts** but this is a general finding, not specific to non-English contexts.
3. **Neither model requires English-pivoting** to achieve strong multilingual performance, suggesting that the internal translation bottleneck documented in earlier work has been substantially mitigated.

### 4.5 Limitations

1. **Sample size:** 50 items per language limits statistical power. Performance differences of 2-4% may not achieve significance.
2. **Prompt confound:** The direct prompt omits chain-of-thought, creating an unfair comparison with self-translate and English CoT. A fairer design would include a "native-language CoT" condition.
3. **Task selection:** MGSM and Belebele test specific capabilities. Generative tasks (summarization, creative writing) may show different patterns.
4. **Temperature 0:** Deterministic decoding means we cannot estimate within-condition variance, limiting statistical analysis to across-language variation.
5. **API routing:** Claude accessed via OpenRouter may introduce latency/routing differences versus direct API access.
6. **Only 2 models:** A broader model comparison would strengthen conclusions.

### 4.6 Relation to Prior Work

Our findings update and partially contradict prior work:
- **Wendler et al. (2024):** Found English-biased internal representations in Llama-2. Our behavioral results suggest newer models may have mitigated this bias at the output level, even if internal representations remain English-influenced.
- **Etxaniz et al. (2023):** Found self-translate improved performance by 10-15%. We find <2% improvement for GPT-4.1 (when controlling for CoT) and 0% for Claude. The English pivot is no longer needed.
- **Ahuja et al. (2023):** Found 10-30% gaps between English and non-English. We find gaps of 0-7%, representing substantial progress.

---

## 5. Conclusions

1. **Frontier LLMs have largely closed the multilingual performance gap** on standard benchmarks. Claude Sonnet 4 shows near-perfect cross-language equity.
2. **English-pivoting strategies (self-translate, English CoT) no longer provide meaningful benefits** for frontier models. The implicit translation bottleneck has been substantially mitigated.
3. **GPT-4.1 is sensitive to prompt format** (requires reasoning scaffolding), but this is language-independent. Claude Sonnet 4 is more robust to prompt variation.
4. **Resource level does not predict performance** in our data, suggesting frontier models have made progress on lower-resource languages.
5. **Hindi remains a challenge** for Belebele reading comprehension across both models, warranting further investigation.

### Future Directions

- Add a **native-language CoT condition** to properly control for the reasoning effect
- Test **generative tasks** (translation quality, summarization) where language-specific capabilities matter more
- Evaluate **open-source models** (Llama-3, Mistral) alongside closed-source ones
- Increase sample sizes for better statistical power
- Test **truly low-resource languages** (e.g., Yoruba, Quechua) that may still show significant gaps

---

## 6. Experimental Details

### 6.1 Runtime Statistics

- **Total experiment time:** 16,147 seconds (4 hours 29 minutes)
- **Total API calls:** 4,800
- **API calls per condition:** 50
- **Average response time:** ~3.4 seconds per call
- **Models tested:** GPT-4.1 (OpenAI), Claude Sonnet 4 (OpenRouter)
- **Date:** February 11, 2026
- **Random seed:** 42

### 6.2 Reproducibility

All code, data, and results are available in this repository. To reproduce:

```bash
# Install dependencies
uv sync

# Run experiments (requires API keys)
OPENAI_API_KEY=... OPENROUTER_KEY=... python3 src/run_experiments.py

# Run analysis
python3 src/analysis.py
```

### 6.3 Figures

All figures are saved in the `figures/` directory:
- `mgsm_*_by_language.png` — Accuracy by language and strategy for each model
- `belebele_*_by_language.png` — Same for Belebele
- `*_gap_heatmap.png` — Performance gap heatmaps (English - target)
- `*_model_comparison.png` — Side-by-side model comparison
- `strategy_lift_by_resource.png` — Strategy lift by resource level
- `democratization_scores.png` — Democratization score comparison

---

## References

- Ahuja, K., et al. (2023). MEGA: Multilingual Evaluation of Generative AI.
- Etxaniz, J., et al. (2023). Do Multilingual Language Models Think Better in English?
- Shi, F., et al. (2022). Language Models are Multilingual Chain-of-Thought Reasoners.
- Wendler, C., et al. (2024). Do Llamas Work in English? On the Latent Language of Multilingual Transformers.
- Zhang, B., et al. (2024). Unveiling Linguistic Regions in Large Language Models.
