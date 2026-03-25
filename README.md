# Learning Path: From Zero to Production AI/ML Engineer

> A complete, hands-on curriculum that takes you from "I know Python" to "I can design, build, and deploy production AI systems."

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

### 🟡 INTERMEDIATE (Weeks 3-5)

| # | Notebook | What You'll Learn | Time |
|---|----------|-------------------|------|
| 03 | [Neural Networks from Scratch](03_neural_networks_from_scratch.ipynb) | Perceptrons, backpropagation by hand, activation functions, gradient descent | 4-5 hrs |
| 04 | [PyTorch & Deep Learning](04_pytorch_deep_learning.ipynb) | Tensors, autograd, building NNs in PyTorch, CNNs, training loops | 4-5 hrs |
| 11 | [Mathematics for ML](11_mathematics_for_ml.ipynb) | Eigenvalues, SVD, optimization theory, information theory, Bayesian thinking | 4-5 hrs |
| 12 | [Advanced Classical ML](12_advanced_classical_ml.ipynb) | XGBoost, LightGBM, SHAP, Optuna, imbalanced data, sklearn pipelines | 4-5 hrs |
| 13 | [CNNs & Training Techniques](13_cnns_and_training_techniques.ipynb) | CNN architectures, ResNet, transfer learning, LR schedulers, mixed precision | 5-6 hrs |

### 🟠 ADVANCED (Weeks 6-9)

| # | Notebook | What You'll Learn | Time |
|---|----------|-------------------|------|
| 05 | [NLP & Text Processing](05_nlp_text_processing.ipynb) | Tokenization, BoW, TF-IDF, Word2Vec, text classification | 3-4 hrs |
| 06 | [RNNs & Sequence Models](06_rnns_sequence_models.ipynb) | RNNs, LSTMs, GRUs, seq2seq, why they fail at scale | 4-5 hrs |
| 07 | [Attention & Transformers](07_attention_and_transformers.ipynb) | Self-attention from scratch, multi-head attention, positional encoding, full transformer | 5-6 hrs |
| 14 | [GenAI: Embeddings & RAG](14_genai_embeddings_and_rag.ipynb) | Text embeddings, vector databases, semantic search, RAG pipelines | 5-6 hrs |
| 15 | [Prompt Engineering & LLM Patterns](15_prompt_engineering_and_llm_patterns.ipynb) | Chain-of-thought, structured output, function calling, agents, ReAct | 4-5 hrs |
| 16 | [Evaluation & Debugging](16_evaluation_and_debugging.ipynb) | ROUGE, BERTScore, LLM-as-judge, data leakage, debugging models | 4-5 hrs |

### 🔴 PROFESSIONAL (Weeks 10-14)

| # | Notebook | What You'll Learn | Time |
|---|----------|-------------------|------|
| 08 | [HuggingFace Ecosystem](08_huggingface_ecosystem.ipynb) | Pretrained models, tokenizers, pipelines, datasets library | 3-4 hrs |
| 09 | [Fine-Tuning Approaches](09_fine_tuning_approaches.ipynb) | Full fine-tuning, LoRA, QLoRA, adapters, prefix tuning, RLHF concepts | 5-6 hrs |
| 10 | [Production Fine-Tuning](10_production_fine_tuning.ipynb) | End-to-end: data pipeline → QLoRA training → evaluation → FastAPI serving → Docker | 5-6 hrs |
| 17 | [MLOps & Experiment Tracking](17_mlops_and_experiment_tracking.ipynb) | MLflow, Weights & Biases, model versioning, data drift, CI/CD for ML | 4-5 hrs |
| 18 | [Deployment & Serving](18_deployment_and_serving.ipynb) | FastAPI, Docker, Kubernetes, vLLM, quantization, load testing | 5-6 hrs |
| 19 | [System Design for AI](19_system_design_for_ai.ipynb) | ML system design, distributed training, scaling, performance optimization | 5-6 hrs |
| 20 | [Capstone Project](20_capstone_project.ipynb) | End-to-end GenAI app: RAG + fine-tuning + API + deployment + security | 6-8 hrs |

---

## Total Estimated Time: 85-105 hours

After completing this curriculum, you will be able to:
- ✅ Understand ML/DL fundamentals deeply (not just API calls)
- ✅ Build neural networks from scratch in PyTorch
- ✅ Understand transformer architecture at the mathematical level
- ✅ Fine-tune LLMs using modern techniques (LoRA, QLoRA)
- ✅ Build RAG pipelines and GenAI applications
- ✅ Engineer prompts, structured output, and LLM agents
- ✅ Evaluate models rigorously (classification + generation + LLM-as-judge)
- ✅ Track experiments and manage model lifecycle (MLflow, W&B)
- ✅ Deploy models as production APIs with Docker and Kubernetes
- ✅ Design end-to-end ML systems that scale
- ✅ Handle security, ethics, and bias in AI systems
- ✅ Discuss trade-offs confidently in system design interviews

---

## Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy torch torchvision transformers datasets peft bitsandbytes accelerate rouge-score bert-score nltk sentence-transformers chromadb faiss-cpu xgboost lightgbm optuna shap mlflow wandb fastapi uvicorn pydantic locust onnx onnxruntime
```
