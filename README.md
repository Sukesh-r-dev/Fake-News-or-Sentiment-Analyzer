
# 🎭 AI Sentiment & Fake News Analyzer

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-DistilBERT-yellow?style=for-the-badge)](https://huggingface.co/docs/transformers/index)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An end-to-end Machine Learning pipeline that fine-tunes a pre-trained Transformer model (`distilbert-base-uncased`) to perform sequence classification on text data. Built with efficiency and scalability in mind using Hugging Face's `Trainer` API.

---

## 🌟 Key Features

* **⚡ Transformer-Powered:** Fine-tunes `distilbert-base-uncased` for high-accuracy text classification with reduced computational overhead.
* **⚙️ Custom Preprocessing Pipeline:** Tokenizes and structures text inputs seamlessly for model training.
* **🏋️ Hugging Face Trainer API:** Employs optimized training, evaluation, and hyperparameter management.
* **💾 Local Inference & Deployment:** Saves trained model checkpoints locally for quick testing and inference.

---

## 📁 Project Structure

```text
.
├── train_sentiment.py    # Core training, tokenization, and evaluation script
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
