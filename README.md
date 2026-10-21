# Twitter Named Entity Recognition (NER) using BERT + CRF

## Overview
An end-to-end NLP project that identifies named entities in Twitter text using a fine-tuned BERT model combined with a Conditional Random Field (CRF) layer.

The model classifies entities such as persons, companies, geographical locations, products, movies, and music artists using BIO sequence labeling.

## Tech Stack
- Python
- PyTorch
- Hugging Face Transformers
- BERT (`bert-base-uncased`)
- Conditional Random Fields (CRF)
- Google Colab
- Seqeval

## Approach
1. Dataset exploration and entity-distribution analysis.
2. Data preprocessing and BIO label alignment with BERT subword tokenization.
3. BERT token-classification baseline.
4. BERT + CRF model training and evaluation.
5. Per-entity error analysis and oversampling experiment.
6. Reusable inference and entity-extraction functions.

## Results

| Metric | Score |
|---|---:|
| Validation F1 | 64.71% |
| Test F1 | 47.31% |
| Oversampled model validation F1 | 61.44% |

The original BERT + CRF model was retained because oversampling did not improve validation performance.

## Key Learnings
- Transformer-based sequence labeling.
- BIO tagging and subword alignment.
- CRF-based sequence decoding.
- Entity-level evaluation and error analysis.
- Building reusable inference functions.

## Limitations and Future Work
The gap between validation and test performance indicates limited generalization. Future work includes investigating dataset differences, evaluating Twitter-specific pretrained models such as BERTweet, and improving recognition of underrepresented entity categories.

## Notebook
The complete implementation, experiments, evaluation, and inference pipeline are available in the Jupyter notebook in this repository.
