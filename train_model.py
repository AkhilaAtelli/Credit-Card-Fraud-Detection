
import pandas as pd

# Load the credit card dataset
data = pd.read_csv("data/creditcard.csv")

# Display first 5 records
print("First 5 Records:")
print(data.head())

# Display dataset shape
print("\nDataset Shape:")
print(data.shape)

# Display column names
print("\nColumn Names:")
print(data.columns)

# Count normal and fraud transactions
print("\nTransaction Distribution:")
print(data["Class"].value_counts())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum().sum())

# Check duplicate records
print("\nDuplicate Records:")
print(data.duplicated().sum())

# Display statistical information
print("\nStatistical Summary:")
print(data.describe())

# Remove duplicate records
data = data.drop_duplicates()

# Check updated dataset
print("\nDataset After Removing Duplicates:")
print("New Dataset Shape:", data.shape)

# Check updated transaction distribution
print("\nUpdated Transaction Distribution:")
print(data["Class"].value_counts())

# Save cleaned dataset
data.to_csv("data/creditcard_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")

import matplotlib.pyplot as plt
import seaborn as sns

# Graph 1: Normal vs Fraud transactions
plt.figure(figsize=(7, 5))

sns.countplot(x="Class", data=data)

plt.title("Normal vs Fraud Transactions")
plt.xlabel("Transaction Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.show()


# Graph 2: Transaction Amount Distribution
plt.figure(figsize=(8, 5))

sns.histplot(data["Amount"], bins=50)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.show()

from sklearn.model_selection import train_test_split

# Separate input features and target variable
X = data.drop("Class", axis=1)
y = data["Class"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining and Testing Data:")
print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)
print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Scale the data for Logistic Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create classification models
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

# Train and evaluate models
for name, model in models.items():

    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    print("\n", name)
    print("-------------------------")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions))
    print("Recall:", recall_score(y_test, predictions))
    print("F1 Score:", f1_score(y_test, predictions))

import joblib

# Save the best model
best_model = models["Random Forest"]

joblib.dump(best_model, "fraud_model.pkl")

# Save feature names
joblib.dump(list(X.columns), "features.pkl")

print("\nRandom Forest model saved successfully!")
print("Feature names saved successfully!")
