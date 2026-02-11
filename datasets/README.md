# Datasets for Multilingual LLM Evaluation

This directory contains multilingual benchmark datasets for evaluating linguistic performance in LLMs. The actual data files are excluded from git (see `.gitignore`) — use the instructions below to reproduce.

## Datasets

### 1. MGSM (Multilingual Grade School Math)
- **Source**: Shi et al. (2022), "Language Models are Multilingual Chain-of-Thought Reasoners"
- **Task**: Mathematical reasoning
- **Languages**: 11 (Bengali, German, English, Spanish, French, Japanese, Russian, Swahili, Telugu, Thai, Chinese)
- **Size**: 250 problems per language (2,750 total)
- **Format**: TSV (question\tanswer)
- **Location**: `mgsm/mgsm_{lang}.tsv`

**Download instructions:**
```bash
mkdir -p datasets/mgsm
for lang in bn de en es fr ja ru sw te th zh; do
  curl -L "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_${lang}.tsv" \
    -o "datasets/mgsm/mgsm_${lang}.tsv"
done
```

### 2. XNLI (Cross-lingual Natural Language Inference)
- **Source**: Conneau et al. (2018), "XNLI: Evaluating Cross-lingual Sentence Representations"
- **Task**: Natural language inference (entailment, contradiction, neutral)
- **Languages**: 15 (Arabic, Bulgarian, Chinese, English, French, German, Greek, Hindi, Russian, Spanish, Swahili, Thai, Turkish, Urdu, Vietnamese)
- **Size**: 392,702 train / 2,490 validation / 5,010 test
- **Format**: HuggingFace Arrow dataset
- **Location**: `xnli/`

**Download instructions:**
```python
from datasets import load_dataset
dataset = load_dataset("xnli", "all_languages")
dataset.save_to_disk("datasets/xnli")
```

### 3. XCOPA (Cross-lingual Choice of Plausible Alternatives)
- **Source**: Ponti et al. (2020), "XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning"
- **Task**: Causal commonsense reasoning
- **Languages**: Downloaded Turkish (tr); available in 11 languages total
- **Size**: 100 validation / 500 test
- **Format**: HuggingFace Arrow dataset
- **Location**: `xcopa_tr/`

**Download instructions:**
```python
from datasets import load_dataset
# Replace "tr" with any of: et, ht, id, it, qu, sw, ta, th, tr, vi, zh
dataset = load_dataset("xcopa", "tr")
dataset.save_to_disk("datasets/xcopa_tr")
```

### 4. Belebele (Multilingual Reading Comprehension)
- **Source**: Bandarkar et al. (2023), "The Belebele Benchmark"
- **Task**: Multiple-choice reading comprehension
- **Languages**: Downloaded 10 variants (English, Chinese Simplified, German, French, Russian, Hindi, Spanish, Japanese, Arabic, Swahili)
- **Size**: 900 questions per language (9,000 total)
- **Format**: HuggingFace Arrow dataset
- **Columns**: link, question_number, flores_passage, question, mc_answer1–4, correct_answer_num, dialect, ds
- **Location**: `belebele_{lang_code}/`

**Language codes:**
| Language | Code | Directory |
|----------|------|-----------|
| English | eng_Latn | `belebele_eng/` |
| Chinese (Simplified) | zho_Hans | `belebele_zho_Hans/` |
| German | deu_Latn | `belebele_deu_Latn/` |
| French | fra_Latn | `belebele_fra_Latn/` |
| Russian | rus_Cyrl | `belebele_rus_Cyrl/` |
| Hindi | hin_Deva | `belebele_hin_Deva/` |
| Spanish | spa_Latn | `belebele_spa_Latn/` |
| Japanese | jpn_Jpan | `belebele_jpn_Jpan/` |
| Arabic | arb_Arab | `belebele_arb_Arab/` |
| Swahili | swh_Latn | `belebele_swh_Latn/` |

**Download instructions:**
```python
from datasets import load_dataset

lang_codes = [
    "eng_Latn", "zho_Hans", "deu_Latn", "fra_Latn", "rus_Cyrl",
    "hin_Deva", "spa_Latn", "jpn_Jpan", "arb_Arab", "swh_Latn"
]
for code in lang_codes:
    dataset = load_dataset("facebook/belebele", code)
    # Belebele uses non-standard config names; save with readable directory names
    short = code.split("_")[0] if code != "zho_Hans" else "zho_Hans"
    dataset.save_to_disk(f"datasets/belebele_{code}")
```

## Usage Notes

- **MGSM**: Load TSV files directly with `pandas.read_csv(path, sep='\t', header=None, names=['question', 'answer'])`
- **XNLI, XCOPA, Belebele**: Load with `datasets.load_from_disk("datasets/<name>")`
- All datasets provide parallel content across languages, enabling fair cross-lingual comparison
- Belebele has the widest language coverage (122 variants available on HuggingFace)

## Recommended Additional Datasets

For broader evaluation, consider also downloading:
- **PAWS-X** (7 languages): Paraphrase identification — `load_dataset("paws-x", "en")`
- **XQuAD** (10 languages): Span-based QA — `load_dataset("xquad", "xquad.en")`
- **XLSum** (44 languages): Summarization — `load_dataset("csebuetnlp/xlsum", "english")`
- **FLORES-200** (204 languages): Translation — `load_dataset("facebook/flores", "eng_Latn-fra_Latn")`
