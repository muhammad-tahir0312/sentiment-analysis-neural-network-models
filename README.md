# Sentiment Analysis with Neural Network Models 🎬🧠

This project evaluates the performance of various deep learning models for sentiment analysis on movie reviews. Using the **Large Movie Review Dataset** from Stanford, we compare four approaches: **CNN**, **LSTM**, **CNN-LSTM**, and **LDA**, aiming to identify the most effective method for binary sentiment classification.

## 📌 Project Description

Sentiment analysis is a key task in Natural Language Processing (NLP) that involves classifying text into sentiments such as positive or negative. This project explores and compares deep learning models capable of understanding the emotions conveyed in textual data.

By applying supervised and unsupervised techniques to a balanced movie review dataset, we assess each model's effectiveness using performance metrics such as **accuracy, precision, recall, and F1-score**.

---

## 👥 Team Members

- **Muhammad Tahir** (21K-4503)  
- **Insha Javed** (21K-3279)  
- **Muhammad Samama** (21K-3205)

Supervised by **Sir Nouman Durrani**

---

## 📂 Dataset

We use the **Large Movie Review Dataset v1.0** by Stanford:

- 📦 50,000 labeled reviews:
  - 25,000 positive
  - 25,000 negative
- 📁 50,000 unlabeled reviews for unsupervised learning
- Balanced data (no more than 30 reviews per movie)

**Dataset source**: [IMDB Dataset – Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/)

---

## 🎯 Objectives

- Train and evaluate **CNN**, **LSTM**, **CNN-LSTM**, and **LDA** models.
- Compare their performance on sentiment classification.
- Explore unsupervised learning with LDA.
- Identify the best-performing model for the task.

---

## 🧠 Models Implemented

| Model        | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| **CNN**      | Captures spatial features and local patterns in text                        |
| **LSTM**     | Excels at learning long-term dependencies in sequences                      |
| **CNN-LSTM** | Combines CNN’s feature extraction with LSTM’s sequential understanding      |
| **LDA**      | Topic modeling technique used for unsupervised text analysis                |

---

## 🛠️ Methodology

1. **Preprocessing**:  
   - Tokenization, stop-word removal, stemming

2. **Training**:  
   - Supervised learning using labeled data for CNN, LSTM, CNN-LSTM

3. **Unsupervised Learning**:  
   - Use LDA on unlabeled data

4. **Evaluation**:  
   - Use accuracy, precision, recall, and F1-score for testing

---

## 📊 Results

| Model       | Accuracy |
|-------------|----------|
| CNN         | 0.86     |
| LSTM        | 0.84     |
| CNN-LSTM    | **0.87** |
| LDA         | 0.51     |

> ✅ **CNN-LSTM** outperformed other models, showcasing the effectiveness of hybrid architectures in sentiment analysis.

## 🚀 Getting Started

### 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/sentiment-analysis-nn-models.git
cd sentiment-analysis-nn-models

# Create virtual environment
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🧾 Requirements
Python 3.8+

TensorFlow or PyTorch

scikit-learn

NLTK

Pandas

NumPy

Matplotlib / Seaborn

---

## 🤝 Acknowledgments
Stanford AI Lab for the dataset

Sir Nouman Durrani for supervision and support
