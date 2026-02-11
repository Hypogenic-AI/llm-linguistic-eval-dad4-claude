# Research Plan: Evaluating Linguistic Performance in LLMs

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly deployed at national scale in non-English-speaking countries (e.g., xAI partnership with Venezuela, OpenAI with Estonia), yet capability evaluations remain overwhelmingly English-centric. Understanding whether these models have implicit internal translation mechanisms — and how much performance degrades across languages — is critical for equitable global deployment. Poor multilingual performance could mean millions of users receive substandard AI services.

### Gap in Existing Work
While Wendler et al. (2024) and Zhang et al. (2024) have shown mechanistic evidence of English-biased internal representations in Llama-2, and Ahuja et al. (2023) benchmarked GPT-3.5/4 multilingually, **no study has systematically compared state-of-the-art 2025 models (GPT-4.1, Claude Sonnet 4.5) across multiple prompting strategies on standardized multilingual benchmarks**. The existing benchmarks are from 2022-2023, and newer models may have closed — or widened — the gap. Additionally, the interaction between prompting strategy (direct, self-translate, XLT, English CoT) and language resource level remains underexplored for current frontier models.

### Our Novel Contribution
1. **Cross-model comparison on 2025 frontier models**: We evaluate GPT-4.1 and Claude Sonnet 4.5 on multilingual benchmarks, providing an updated snapshot of the multilingual landscape.
2. **Prompting strategy × language interaction**: We systematically test whether explicit English-pivoting strategies (self-translate, XLT) still help with newer models, or if their improved multilingual training has reduced the need.
3. **Evidence for implicit translation**: By comparing direct multilingual inference with English-pivoted strategies, performance differences serve as behavioral evidence for internal translation mechanisms.
4. **Democratization analysis**: We measure cross-language equity using the democratization score, comparing high-resource vs. low-resource languages across models and strategies.

### Experiment Justification
- **Experiment 1 (Direct Multilingual Inference)**: Establishes baseline performance across languages — needed to quantify the performance gap.
- **Experiment 2 (Self-Translate Strategy)**: If self-translating to English before solving improves accuracy, this is behavioral evidence of an internal English-processing bottleneck.
- **Experiment 3 (English CoT Prompting)**: Tests whether English chain-of-thought reasoning helps non-English tasks, another signal of English-biased internal processing.
- **Experiment 4 (Cross-Model Comparison)**: Different models may have different multilingual capabilities; comparing them reveals whether the English pivot is model-dependent.

---

## Research Question
Do state-of-the-art LLMs (GPT-4.1, Claude Sonnet 4.5) exhibit significant performance degradation on non-English tasks compared to English, and does explicit English-pivoting (self-translate, English CoT) improve non-English performance — providing behavioral evidence for implicit internal translation mechanisms?

## Background and Motivation
LLMs trained predominantly on English data are deployed globally. Literature shows:
- LLMs use English-biased internal representations (Wendler 2024, Zhang 2024)
- Self-translation to English improves performance (Etxaniz 2023)
- English CoT works cross-lingually (Shi 2022)
- Low-resource languages suffer the most (Ahuja 2023)

We need updated evidence on whether these patterns persist in newer, more capable models.

## Hypothesis Decomposition

**H1**: LLMs show statistically significant performance degradation on non-English vs. English tasks.
- Metric: Accuracy difference (English - target language)
- Success: Significant difference (p < 0.05) for at least some languages

**H2**: Explicit English-pivoting strategies (self-translate, English CoT) improve non-English performance.
- Metric: Accuracy improvement from direct → self-translate/CoT
- Success: Significant improvement (p < 0.05), especially for non-Latin-script languages

**H3**: Performance degradation correlates with language resource level (low-resource languages suffer more).
- Metric: Correlation between language resource level and accuracy
- Success: Significant negative correlation

**H4**: Different models show different multilingual capability profiles.
- Metric: Model × language interaction
- Success: Significant interaction effect

## Proposed Methodology

### Approach
We conduct a multi-factor experiment: **2 models × 3 prompting strategies × 8 languages × 2 tasks (MGSM + Belebele)**. We use real API calls to GPT-4.1 (via OpenAI) and Claude Sonnet 4.5 (via OpenRouter) to test each condition.

### Task Selection
1. **MGSM** (Math reasoning): Well-structured, has clear correct answers, tests reasoning ability. 250 problems per language. We use a sample of 50 per language for feasibility.
2. **Belebele** (Reading comprehension): Multiple-choice with 4 options, tests understanding. 900 per language. We use a sample of 50 per language.

### Language Selection (8 languages across resource levels)
- **High-resource**: English (en), Chinese (zh), German (de), French (fr)
- **Medium-resource**: Russian (ru), Japanese (ja)
- **Low-resource**: Swahili (sw), Bengali (bn) [MGSM only; for Belebele: Hindi instead]

### Prompting Strategies
1. **Direct**: Prompt in the target language, solve in target language
2. **Self-Translate**: Ask the model to translate the problem to English, then solve
3. **English CoT**: Provide English chain-of-thought instructions, input in target language

### Models
1. **GPT-4.1** (via OpenAI API): State-of-the-art reasoning
2. **Claude Sonnet 4.5** (via OpenRouter): Strong multilingual capabilities

### Experimental Steps
1. Load and sample datasets (50 items per language per task)
2. Create prompt templates for each strategy
3. Run API calls for all conditions (2 models × 3 strategies × 8 languages × 2 tasks × 50 items)
4. Parse and evaluate responses
5. Compute accuracy, democratization scores, and statistical tests
6. Visualize results

### Baselines
- English performance serves as the ceiling baseline
- Direct inference serves as the default baseline
- Random performance: 25% for Belebele (4-choice), ~0% for MGSM

### Evaluation Metrics
- **Accuracy**: Primary metric (exact match for MGSM, correct choice for Belebele)
- **Performance Gap**: English accuracy minus target language accuracy
- **Democratization Score**: Average accuracy / best language accuracy (1.0 = perfect equity)
- **Self-Translate Lift**: Accuracy(self-translate) - Accuracy(direct)

### Statistical Analysis Plan
- **H1**: Paired t-test or Wilcoxon signed-rank test (English vs. each language)
- **H2**: Paired t-test (direct vs. self-translate/CoT per language)
- **H3**: Spearman correlation (resource level rank vs. accuracy)
- **H4**: Two-way ANOVA (model × language) or non-parametric equivalent
- Significance level: α = 0.05 with Bonferroni correction for multiple comparisons
- Effect sizes: Cohen's d for pairwise comparisons
- Confidence intervals: 95% bootstrap CIs for accuracy estimates

## Expected Outcomes
- **H1 supported**: Non-English performance < English, especially for low-resource languages
- **H2 partially supported**: Self-translate helps more for low-resource languages; English CoT provides moderate improvement
- **H3 supported**: Low-resource languages show larger performance gaps
- **H4 supported**: Models differ in their multilingual profiles (e.g., one may be stronger on non-Latin scripts)

If H2 is strongly supported (self-translate significantly helps), this provides behavioral evidence for implicit internal translation — the models "think better" when given English input explicitly, suggesting internal processing is English-biased.

## Timeline and Milestones
1. **Environment Setup & Data Prep** (10 min): Install packages, validate datasets
2. **Code Implementation** (30 min): Prompt templates, API calling code, evaluation
3. **Run Experiments** (60-90 min): ~4,800 API calls total
4. **Analysis & Visualization** (30 min): Statistical tests, plots
5. **Documentation** (20 min): REPORT.md, README.md

## Potential Challenges
- **API rate limits**: Mitigate with retry logic, exponential backoff
- **Cost**: ~4,800 calls × ~200 tokens avg = ~1M tokens ≈ $10-30. Manageable.
- **Response parsing**: MGSM needs numerical answer extraction; Belebele needs option extraction. Use robust regex patterns.
- **Model availability**: If one model's API is down, focus on the other.

## Success Criteria
1. Experiments run successfully for all conditions
2. Clear quantitative evidence for/against each hypothesis
3. Statistical tests with proper corrections
4. Informative visualizations showing cross-language patterns
5. Comprehensive REPORT.md with actionable findings
