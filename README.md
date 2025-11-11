# Fine-Tuning and Reinforcement Learning with Unsloth

This repository contains five Google Colab notebooks demonstrating modern AI fine-tuning and reinforcement learning workflows using the **[Unsloth.ai](https://unsloth.ai)** framework.  

---

## Overview of Notebooks

| Notebook | Description | Model | Dataset | Objective |
|-----------|--------------|--------|----------|------------|
| **Colab 1 – Full Fine-tuning** | Full fine-tuning of a small model using the **Alpaca** dataset. | `unsloth/smollm2-135m` | [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | Learn general instruction following. |
| **Colab 2 – LoRA Fine-tuning** | Parameter-efficient fine-tuning using **LoRA adapters** on the same dataset. | `unsloth/smollm2-135m` | Alpaca | Reduce GPU memory usage and training time. |
| **Colab 3 – Preference-based Reinforcement Learning** | Train the model to prefer “chosen” over “rejected” responses (RLHF-style). | `unsloth/smollm2-135m` | [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) | Reinforcement learning from preferences. |
| **Colab 4 – GRPO Reasoning Fine-tuning** | Apply **Guided Reward Policy Optimization (GRPO)** on reasoning problems. | `unsloth/smollm2-135m` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | Reinforcement learning for reasoning improvement. |
| **Colab 5 – Continued Pretraining** | Continue pretraining to teach the model a **new language (Hindi)**. | `unsloth/smollm2-135m` | Custom Hindi text file (from Kaggle) | Make the model multilingual. |

---

## ⚙️ Requirements

All notebooks run on **Google Colab** with a GPU (T4 or better).

Install dependencies at the start of each notebook:
```bash
pip install unsloth transformers datasets accelerate bitsandbytes peft trl
