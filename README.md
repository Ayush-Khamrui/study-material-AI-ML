# Learning Path: From Zero to Production AI/ML Engineer

> A complete, hands-on curriculum that takes you from "I know Python" to "I can fine-tune and deploy LLMs in production."

---

## Prerequisites
- Basic Python knowledge (variables, loops, functions)
- High school math (algebra, basic calculus concepts)
- Curiosity

## How to Use This Guide

1. **Go in order** — each notebook builds on the previous one
2. **Run every cell** — don't just read, execute and experiment
3. **Do the exercises** — they're designed to solidify understanding
4. **Break things** — change parameters, see what happens

---

## The Curriculum

### 🟢 BEGINNER (Weeks 1-2)

| # | Notebook | What You'll Learn | Time |
|---|----------|-------------------|------|
| 01 | [ML Foundations](01_ml_foundations.ipynb) | NumPy, Pandas, Matplotlib, statistics, data thinking | 3-4 hrs |
| 02 | [Classical ML](02_classical_ml.ipynb) | Linear/logistic regression, decision trees, metrics, train/test/val splits, overfitting | 4-5 hrs |

### 🟡 INTERMEDIATE (Weeks 3-4)

| # | Notebook | What You'll Learn | Time |
|---|----------|-------------------|------|
| 03 | [Neural Networks from Scratch](03_neural_networks_from_scratch.ipynb) | Perceptrons, backpropagation by hand, activation functions, gradient descent | 4-5 hrs |
| 04 | [PyTorch & Deep Learning](04_pytorch_deep_learning.ipynb) | Tensors, autograd, building NNs in PyTorch, CNNs, training loops | 4-5 hrs |

### 🟠 ADVANCED (Weeks 5-7)

| # | Notebook | What You'll Learn | Time |
|---|----------|-------------------|------|
| 05 | [NLP & Text Processing](05_nlp_text_processing.ipynb) | Tokenization, BoW, TF-IDF, Word2Vec, text classification | 3-4 hrs |
| 06 | [RNNs & Sequence Models](06_rnns_sequence_models.ipynb) | RNNs, LSTMs, GRUs, seq2seq, why they fail at scale | 4-5 hrs |
| 07 | [Attention & Transformers](07_attention_and_transformers.ipynb) | Self-attention from scratch, multi-head attention, positional encoding, full transformer | 5-6 hrs |

### 🔴 PROFESSIONAL (Weeks 8-10)

| # | Notebook | What You'll Learn | Time |
|---|----------|-------------------|------|
| 08 | [HuggingFace Ecosystem](08_huggingface_ecosystem.ipynb) | Pretrained models, tokenizers, pipelines, datasets library | 3-4 hrs |
| 09 | [Fine-Tuning Approaches](09_fine_tuning_approaches.ipynb) | Full fine-tuning, LoRA, QLoRA, adapters, prefix tuning, RLHF concepts | 5-6 hrs |
| 10 | [Production Fine-Tuning](10_production_fine_tuning.ipynb) | End-to-end: data pipeline → QLoRA training → evaluation → FastAPI serving → Docker | 5-6 hrs |

---

## Total Estimated Time: 40-50 hours

After completing this curriculum, you will be able to:
- ✅ Understand ML/DL fundamentals deeply (not just API calls)
- ✅ Build neural networks from scratch in PyTorch
- ✅ Understand transformer architecture at the mathematical level
- ✅ Fine-tune LLMs using modern techniques (LoRA, QLoRA)
- ✅ Evaluate models rigorously (classification + generation metrics)
- ✅ Deploy models as production APIs
- ✅ Discuss trade-offs confidently in interviews

---

## Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch transformers datasets peft bitsandbytes accelerate rouge-score
```
