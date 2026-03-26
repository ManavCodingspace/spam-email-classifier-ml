# 📩 Email Spam Classifier using Machine Learning

## 🚀 Overview

This project is a Machine Learning-based Email Spam Classifier built using Python and Scikit-learn.
It classifies messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) techniques.

---

## 🎯 Objective

The goal of this project is to build an efficient model that can accurately detect spam messages by analyzing text data and improving classification performance using feature engineering.

---

## 🧠 Approach

### 🔹 Data Preprocessing

* Converted text to lowercase
* Removed punctuation
* Cleaned and normalized textual data

### 🔹 Feature Extraction

* Used **TF-IDF Vectorization**
* Removed stopwords
* Limited features using `max_features=3000`

### 🔹 Model Used

* **Multinomial Naive Bayes**
* Suitable for text classification problems

### 🔹 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 📊 Results

| Metric           | Value |
| ---------------- | ----- |
| Accuracy         | 98%   |
| Precision (Spam) | 1.00  |
| Recall (Spam)    | 0.85  |
| F1-score         | 0.92  |

### 🔍 Key Insights

* High overall accuracy with strong performance on non-spam messages
* Improved spam detection recall significantly
* Reduced false negatives compared to baseline model

---

## 📦 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorizer

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/spam-email-classifier-ml.git
cd spam-email-classifier-ml
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the main script:

```bash
python src/main.py
```

---

## 📁 Project Structure

```
spam-email-classifier-ml/
│
├── data/
│   └── spam.csv
│
├── src/
│   └── main.py
│
├── notebook/
│   └── spam_classifier.ipynb
│
├── requirements.txt
├── README.md
├── .gitignore
```

---

## 📊 Dataset

* SMS Spam Collection Dataset
* Contains labeled messages: **spam** and **ham**

---

## 🚀 Future Improvements

* Improve recall using advanced models (Logistic Regression, SVM)
* Use n-grams for better feature extraction
* Deploy as a web application (Streamlit / Flask)
* Handle class imbalance more effectively

---

## 🙋‍♂️ Author

**Manav**
Aspiring Machine Learning Engineer

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
