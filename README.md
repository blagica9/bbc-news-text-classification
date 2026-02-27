# 📊 BBC News Text Classification

## Overview

This project implements an end-to-end Natural Language Processing (NLP) pipeline for multi-class classification of BBC news articles into five categories:

- Business  
- Entertainment  
- Politics  
- Sport  
- Tech  

The focus of the project is to explore data quality, feature engineering, unsupervised topic modeling, and supervised classification, followed by a comparative analysis of different modeling approaches.

---

## Dataset

The dataset consists of two CSV files:

- `BBC News Train.csv` – labeled BBC news articles  
- `BBC News Test.csv` – unlabeled articles for inference  

Each record contains:

- `ArticleId` – unique identifier  
- `Text` – full article content  
- `Category` – target label (training set only)

The training dataset contains 1,490 labeled articles and is relatively balanced across categories, with a minor imbalance ratio of approximately 1.32.

---

## Exploratory Data Analysis

### Category Distribution

The dataset shows a fairly balanced distribution across categories:

- Sport: 346  
- Business: 336  
- Politics: 274  
- Entertainment: 273  
- Tech: 261  

The small imbalance did not require resampling techniques.

### Text Length Analysis

Word count analysis revealed that most articles contain fewer than 500 words.  
Text length distributions overlap significantly across categories, indicating that article length alone is not a strong predictive feature.

---

## Data Quality

- No missing values were detected.
- Duplicate articles (based on text content) were identified and removed.
- Ensuring clean data improves model reliability and prevents biased training caused by redundant samples.

---

## Text Preprocessing

The following preprocessing steps were applied:

- Lowercasing
- Removal of punctuation and special characters (regex-based)
- Stop word removal using `ENGLISH_STOP_WORDS`

These steps reduce noise, normalize vocabulary, and improve feature quality for downstream modeling.

---

## Feature Engineering – TF-IDF

Text data was transformed into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

- Vocabulary limited to the top 5,000 features
- Vectorizer fitted on training data only
- Applied consistently to validation and test sets

TF-IDF highlights words that are important within a document but less frequent across the dataset, providing a strong baseline representation for classical machine learning models.

---

## Unsupervised Learning – NMF

Non-Negative Matrix Factorization (NMF) was applied to:

- Reduce the TF-IDF feature space
- Identify latent topic structures
- Analyze how articles group naturally based on shared vocabulary

Reconstruction error was used to evaluate how well the reduced representation approximates the original feature space.

Although NMF does not use class labels, it provided meaningful insights into topic structure within the dataset.

---

## Supervised Learning

### Logistic Regression

Logistic Regression was trained using TF-IDF features:

- Train/Validation split (80/20)
- Training accuracy: ~99%
- Validation accuracy: ~96–97%

Additional experiments using random subsets of the training data (10%, 30%, 50%, 70%) showed steady performance improvements as more data became available, indicating good generalization behavior.

### Naive Bayes

Multinomial Naive Bayes was evaluated as a probabilistic baseline.  
While slightly less accurate than Logistic Regression, it provided competitive results with simpler modeling assumptions.

---

## Model Comparison

| Model | Validation Accuracy (approx.) |
|------|-------------------------------|
| Logistic Regression | 96–97% |
| NMF-based approach | ~95% |
| Naive Bayes | Slightly below LogReg |

**Key insight:**  
Supervised learning clearly outperformed unsupervised topic modeling for this task. Logistic Regression combined with TF-IDF achieved the best balance between simplicity, interpretability, and performance.

---

## Limitations

- Word-based TF-IDF representations may struggle with semantic similarity between related categories (e.g., Business vs Politics).
- No contextual embeddings were used.
- Hyperparameter tuning was limited.


---

## Technologies Used

- Python 3.x  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  

---


1. Install dependencies:
```bash
pip install -r requirements.txt
