# 📰 News Classification using NLP & Machine Learning

A Natural Language Processing (NLP) project that automatically classifies news articles into four categories — **World, Sports, Business, and Sci/Tech** — using **TF-IDF** for text feature extraction and **Logistic Regression** for classification.

The trained model is integrated with a **Streamlit web application** that allows users to enter a news headline or article description and receive a predicted category with confidence scores.

---

## 📌 Project Overview

News websites publish thousands of articles every day, making automatic organization and categorization useful for information retrieval and content management.

This project uses machine learning and NLP techniques to automatically classify news text into one of four categories:

- 🌍 **World**
- ⚽ **Sports**
- 💼 **Business**
- 🔬 **Sci/Tech**

The project covers the complete machine learning workflow:

**Data → Preprocessing → TF-IDF → Model Training → Evaluation → Model Saving → Streamlit Deployment**

---

## 🎯 Objectives

- Perform text preprocessing on news articles.
- Convert text into numerical features using TF-IDF.
- Train a machine learning classification model.
- Evaluate the model using standard classification metrics.
- Save the trained model and vectorizer for reuse.
- Build an interactive Streamlit application.
- Predict the category of new/unseen news articles.

---

## 📂 Dataset

The project uses a news classification dataset containing separate training and testing files.

### Training Dataset
- **120,000 news articles**

### Testing Dataset
- **7,600 news articles**

### Features

| Column | Description |
|---|---|
| `Class Index` | Numerical category label |
| `Title` | News headline |
| `Description` | News article description |

### News Categories

| Class Index | Category |
|---|---|
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Sci/Tech |

---

## 🔄 Machine Learning Workflow


News Dataset
     ↓
Data Loading
     ↓
Data Cleaning & Preprocessing
     ↓
Combine Title + Description
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Model Evaluation
     ↓
Save Model & Vectorizer
     ↓
Streamlit Application
     ↓
News Category Prediction

## 🧹 Data Preprocessing

The text data is prepared before training the machine learning model.

The preprocessing workflow includes:

* Handling missing values
* Combining the `Title` and `Description` fields
* Converting text into a suitable format for NLP
* Preparing training and testing features
* Transforming text into numerical representations

---

## 🔤 TF-IDF Vectorization

Since machine learning models cannot directly process raw text, **TF-IDF (Term Frequency–Inverse Document Frequency)** is used to convert news articles into numerical feature vectors.

TF-IDF assigns higher importance to words that are useful for distinguishing between different news categories.

The trained vectorizer is saved and reused during prediction so that new user input is transformed using the same feature representation as the training data.

---

## 🤖 Machine Learning Model

### Logistic Regression

**Logistic Regression** is used as the classification algorithm.

It is well suited for text classification because it performs effectively with high-dimensional sparse feature representations such as TF-IDF.

The model learns patterns from the training news articles and predicts one of the four categories:

World
Sports
Business
Sci/Tech


## 📊 Model Performance

The Logistic Regression model achieved approximately:

| Metric         | Score |
| -------------- | ----: |
| Accuracy       |  ~92% |
| Macro F1-Score | ~0.92 |

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

These metrics were used to understand the overall classification performance as well as performance across individual categories.

---

## 💾 Saved Model Files

The trained components are stored using `joblib`.

```text
model/
├── logistic_model.pkl
└── tfidf_vectorizer.pkl
```

### `logistic_model.pkl`

Contains the trained Logistic Regression classifier.

### `tfidf_vectorizer.pkl`

Contains the fitted TF-IDF vectorizer used to transform text before prediction.

---

## 🌐 Streamlit Application

A Streamlit web application was developed to make the trained model interactive.

Users can:

1. Enter a news headline or article description.
2. Click the prediction button.
3. View the predicted news category.
4. View the model confidence/probability.
5. Compare probabilities across the four supported categories.

### Supported Categories

```text
🌍 World
⚽ Sports
💼 Business
🔬 Sci/Tech
```

---

## 🖥️ Running the Application Locally

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the project

```bash
cd News-Classification-NLP
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Logistic Regression

### Natural Language Processing

* TF-IDF

### Model Persistence

* Joblib

### Web Application

* Streamlit

### Development Environment

* Jupyter Notebook
* Git
* GitHub

---

## 📁 Project Structure

```text
News_Classification/
│
├── model/
│   ├── logistic_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── News_Classification.ipynb
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

> **Note:** The dataset files are kept locally and are excluded from the GitHub repository because of their large size. The trained model and TF-IDF vectorizer are sufficient to run the Streamlit prediction application.

---

## 🔐 GitHub Dataset Handling

The dataset is excluded from version control using `.gitignore`.

```gitignore
data/
```

This keeps the repository lightweight while allowing the trained model and application code to be shared.

---

## 🚀 Future Improvements

Possible improvements include:

* Experimenting with Naive Bayes and Linear SVM.
* Hyperparameter tuning.
* Advanced NLP preprocessing.
* Word embeddings such as Word2Vec or GloVe.
* Transformer-based models such as BERT.
* Adding multilingual news classification.
* Deploying the Streamlit application publicly.
* Adding batch prediction for multiple news articles.
* Adding model explainability.

---

## 💡 Key Learning Outcomes

Through this project, the following concepts were implemented:

* Natural Language Processing
* Text preprocessing
* TF-IDF feature engineering
* Supervised machine learning
* Multi-class classification
* Model evaluation
* Model serialization
* Streamlit application development
* Git and GitHub project management

---

## 👩‍💻 Author

**Bhumika G S**

Machine Learning | NLP | Data Science | AI

---

## ⭐ Project Summary

This project demonstrates an end-to-end **NLP-based news classification system**, starting from raw news data and progressing through preprocessing, feature engineering, machine learning, evaluation, model persistence, and deployment using Streamlit.

The project showcases how machine learning can be used to automatically organize and classify large volumes of textual news data.

````

`git rm --cached` removes the dataset **from Git tracking but does not delete it from your computer**.
