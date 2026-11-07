# Credit Card Fraud Detection Using Classification Techniques

## 1. Project Overview
Credit Card Fraud Detection is a Machine Learning project that identifies fraudulent credit card transactions and distinguishes them from normal transactions.

This project uses classification algorithms to analyze transaction data and predict whether a transaction is legitimate or fraudulent.

## 2. Objective
The main objective is to detect suspicious credit card transactions and help reduce financial losses caused by fraudulent activities.

## 3. Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

## 4. Machine Learning Algorithms
The following classification algorithms are used:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## 5. Dataset Description
The project uses a credit card transaction dataset containing:
- 284,807 transactions
- 30 input features
- 1 target column (Class)
- Class 0 – Normal Transaction
- Class 1 – Fraudulent Transaction

**Note:** The original dataset is not included in this repository because of its large file size.

## 6. Key Features
- Data preprocessing and cleaning
- Machine Learning model training
- Normal and fraudulent transaction prediction
- Model performance evaluation
- Accuracy, Precision, Recall and F1-score analysis
- Interactive Streamlit dashboard
- Transaction amount analysis
- Fraud vs Normal transaction visualization

## 7. Model Evaluation
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score

Random Forest achieved the best overall performance in the current model evaluation.

## 8. Project Structure

```text
Credit_Card_Fraud_Detection/
│
├── app.py
├── train_model.py
├── data/
├── features/
├── fraud-model/
├── requirements.txt
├── README.md
└── .gitignore
```

## 9. How to Run the Project

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Install the required libraries

```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit application

```bash
streamlit run app.py
```

## 10. Project Information

- **Project Title:** Credit Card Fraud Detection Using Classification Techniques
- **Domain:** Machine Learning
- **Application:** Financial Fraud Detection

## 11. Developed By

**Akhila Atelli**

B.Tech – Artificial Intelligence and Machine Learning