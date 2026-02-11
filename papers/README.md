# Downloaded Papers

Papers related to "Evaluating Linguistic Performance in LLMs" - focusing on multilingual evaluation, internal translation mechanisms, and cross-lingual performance disparities.

## Core Papers (Internal Translation & Latent Language)

1. **Do Llamas Work in English? On the Latent Language of Multilingual Transformers** (wendler2024_do_llamas_work_in_english.pdf)
   - Authors: Wendler, Veselovsky, Monea, West (EPFL)
   - Year: 2024
   - arXiv: 2402.10588
   - Why relevant: Directly investigates whether LLMs use English as internal pivot language. Finds three-phase processing: input space → English-biased concept space → output language space.

2. **How do Large Language Models Handle Multilingualism?** (zhang2024_how_llms_handle_multilingualism.pdf)
   - Authors: Zhao, Zhang, Chen, Kawaguchi, Bing (NUS, Alibaba DAMO)
   - Year: 2024 (NeurIPS)
   - arXiv: 2402.18815
   - Why relevant: Proposes MWork hypothesis: LLMs understand in multilingual, reason in English, generate in original language. Introduces PLND for detecting language-specific neurons.

3. **Do Multilingual Language Models Think Better in English?** (etxaniz2023_multilingual_llm_prompts.pdf)
   - Authors: Etxaniz, Azkune, Soroa, Lopez de Lacalle, Artetxe
   - Year: 2023
   - arXiv: 2308.01223
   - Why relevant: Introduces self-translate approach showing LLMs perform better when self-translating to English first, proving models can't leverage full potential in non-English.

## Core Papers (Multilingual Evaluation & Benchmarks)

4. **ChatGPT Beyond English** (lai2023_chatgpt_beyond_english.pdf)
   - Authors: Lai et al.
   - Year: 2023
   - arXiv: 2304.04675
   - Why relevant: Comprehensive multilingual evaluation of ChatGPT across tasks and languages.

5. **MEGA: Multilingual Evaluation of Generative AI** (ahuja2023_mega.pdf)
   - Authors: Ahuja, Diddee, Hada et al. (Microsoft)
   - Year: 2023
   - arXiv: 2303.12528
   - Why relevant: First comprehensive benchmarking of generative LLMs across 16 datasets, 70 languages. Shows significant English vs non-English performance gaps.

6. **Not All Languages Are Created Equal in LLMs** (huang2023_not_all_languages_equal.pdf)
   - Authors: Huang, Tang, Zhang, Zhao, Song, Xia, Wei (Microsoft Research Asia)
   - Year: 2023
   - arXiv: 2305.07004
   - Why relevant: Introduces Cross-Lingual-Thought (XLT) prompting to improve multilingual performance. Shows >10 point gains on reasoning/QA tasks.

7. **Is Translation All You Need?** (shi2023_is_translation_all_you_need.pdf)
   - Authors: Etxaniz et al.
   - Year: 2023
   - arXiv: 2308.01223
   - Why relevant: Studies self-translate approach for multilingual tasks.

8. **Language Models are Multilingual Chain-of-Thought Reasoners** (shi2022_multilingual_cot_reasoners.pdf)
   - Authors: Shi et al.
   - Year: 2022
   - arXiv: 2210.03057
   - Why relevant: Introduces MGSM benchmark; shows English CoT improves multilingual reasoning.

## Benchmark & Infrastructure Papers

9. **XTREME: A Massively Multilingual Multi-task Benchmark** (hu2020_xtreme.pdf)
   - Authors: Hu et al.
   - Year: 2020
   - arXiv: 2003.11080
   - Why relevant: Foundational multilingual benchmark covering 9 tasks, 40 languages.

10. **The Belebele Benchmark** (bandarkar2023_belebele.pdf)
    - Authors: Bandarkar et al. (Meta)
    - Year: 2023
    - arXiv: 2308.16884
    - Why relevant: Parallel reading comprehension in 122 language variants.

11. **MMLU-ProX** (son2024_mmlu_prox.pdf)
    - Authors: Son et al.
    - Year: 2024
    - arXiv: 2406.04264
    - Why relevant: Multilingual version of MMLU for advanced LLM evaluation.

12. **Unsupervised Cross-lingual Representation Learning at Scale (XLM-RoBERTa)** (conneau2020_xlm_roberta.pdf)
    - Authors: Conneau et al.
    - Year: 2020
    - arXiv: 1911.02116
    - Why relevant: Foundational multilingual model and cross-lingual transfer analysis.

## Cross-lingual Transfer & Analysis Papers

13. **BUFFET: Benchmarking LLMs for Few-shot Cross-lingual Transfer** (asai2023_buffet.pdf)
    - Authors: Asai et al.
    - Year: 2023
    - arXiv: 2305.14857
    - Why relevant: Comprehensive few-shot cross-lingual evaluation benchmark.

14. **Turning English-centric LLMs Into Polyglots** (wendler2024_turning_english_centric_polyglots.pdf)
    - Authors: Wendler et al.
    - Year: 2024
    - arXiv: 2405.06089
    - Why relevant: Studies how much multilingual data is needed for English-centric LLMs.

15. **Democratizing LLMs for Low-Resource Languages** (ranjan2023_democratizing_llms.pdf)
    - Authors: Ranjan et al.
    - Year: 2023
    - arXiv: 2306.11837
    - Why relevant: Uses linguistically-diverse prompts to leverage English-dominant abilities.

16. **PLUG: Leveraging Pivot Language in Cross-Lingual Instruction Tuning** (chen2023_plug_pivot_language.pdf)
    - Authors: Chen et al.
    - Year: 2023
    - arXiv: 2311.08711
    - Why relevant: Studies pivot language (English) role in cross-lingual instruction tuning.

17. **Beyond English: Impact of Prompt Translation Strategies** (min2024_beyond_english_prompting.pdf)
    - Authors: Min et al.
    - Year: 2024
    - arXiv: 2401.07164
    - Why relevant: Evaluates different prompt translation strategies across languages.

18. **Crosslingual Capabilities and Knowledge Barriers** (li2024_crosslingual_capabilities_barriers.pdf)
    - Authors: Li et al.
    - Year: 2024
    - arXiv: 2406.01581
    - Why relevant: Identifies knowledge barriers in multilingual LLMs.

19. **Analyzing Cross-Lingual Knowledge Transfer Evaluation** (keleg2023_cross_lingual_knowledge_eval.pdf)
    - Authors: Keleg et al.
    - Year: 2023
    - arXiv: 2305.12679
    - Why relevant: Methods for evaluating cross-lingual knowledge transfer.

20. **Probing Cross-lingual Alignment during LLM Training** (lee2024_probing_crosslingual_alignment.pdf)
    - Authors: Lee et al.
    - Year: 2024
    - arXiv: 2404.18397
    - Why relevant: Studies how cross-lingual alignment emerges during LLM training.
