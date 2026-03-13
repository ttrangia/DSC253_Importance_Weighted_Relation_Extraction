# DSC253_Importance_Weighted_Relation_Extraction
Final project for DSC 253 - Advanced Data-Driven Text Mining

## ATLANTIS for Relation Extraction
Adaptation of the ATLANTIS importance-weighted weak supervision framework (ACL 2025) to relation extraction using LLM-generated weak labels.

## Notebooks

### `atlantis_flan_t5.ipynb`
Seq2seq implementation using Flan-T5 (small → base).

### `atlantis_qwen.ipynb`
Decoder-only implementation using Qwen2 (0.5B / 1.5B / 7B).

Key architectural difference from Flan-T5: input and label are concatenated as a single sequence `[prompt tokens][label tokens][EOS]`, with the prompt masked via `-100` in the labels tensor so loss is computed over label tokens only.

---

## Datasets

| Dataset | HuggingFace ID | Train | Test | Relations |
|---------|---------------|-------|------|-----------|
| SemEval 2010 Task 8 | `sem_eval_2010_task_8` | 8,000 | 2,717 | 10 |
| CoNLL04 | `DFKI-SLT/conll04` | ~1,150* | ~340* | 5 |

\* After expanding multi-relation sentences to one example per relation pair.

Set `DATASET = "semeval"` or `DATASET = "conll2004"` in the config cell.

---

## Experimental Paradigms

**Paradigm A — Clean Labels**  
Train and evaluate on gold-labeled data. Validates the pipeline and provides a baseline. ATLANTIS is expected to match SFT since there is no noise to correct.

**Paradigm B — Weak Labels**  
Train on LLM-generated (GPT) weak labels, evaluate on gold test set. This is the primary setting ATLANTIS is designed for. Three conditions are compared:
- **Gold**: upper bound, trained on gold labels
- **Uniform SFT**: trained on weak labels with uniform weights
- **ATLANTIS**: trained on weak labels with importance weights

---

## Dependencies

```
transformers
datasets
scikit-learn
accelerate
sentencepiece
bitsandbytes
scipy
tqdm
matplotlib
pandas
```

Install via: `pip install -q transformers datasets scikit-learn accelerate sentencepiece bitsandbytes scipy tqdm matplotlib pandas`

Hardware: Experiments were run on Google Colab with A100 GPU High-RAM configuration
Runtime: Flan evaluations take 20-30 minutes. Qwen experiments may take 40 minutes to 2 hours, depending on parameters and model size.



