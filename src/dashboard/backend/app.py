 #Week 1: Project Setup & Data Acquisition

# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import warnings

warnings.filterwarnings("ignore")

# Load the Dataset
file_path = "creditcard.csv"  
df = pd.read_csv(file_path)

# Display Dataset Information
print("Dataset Overview:")
print(df.info())  # Check column names, data types
print(df.describe())  # Summary statistics

# Check for Missing Values
print("\nMissing Values Per Column:")
print(df.isnull().sum())


# Week 2: Exploratory Data Analysis (EDA)

# Countplot for Class Distribution (Imbalance Check)
sns.countplot(x="Class", data=df)
plt.title("Distribution of Legitimate vs Fraudulent Transactions")
plt.show()

# Pairplot to check relationships between key variables
sns.pairplot(df[['Time', 'Amount', 'Class']], hue="Class", diag_kind='kde')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# Week 3: Data Preprocessing

# Drop Unnecessary Features
X = df.drop("Class", axis=1)  # Features
y = df["Class"]  # Target Variable

# Split the dataset into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize Numerical Features (Time, Amount)
scaler = StandardScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

# Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nClass Distribution After SMOTE:")
print(y_train_balanced.value_counts())

# Week 4: Feature Engineering & Model Selection

# Feature Importance using RandomForest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)

# Get Feature Importance
feature_importance = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot Feature Importance
plt.figure(figsize=(12, 6))
feature_importance[:10].plot(kind='bar', color='blue')
plt.title("Top 10 Most Important Features")
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.show()

# Select Top Features
top_features = feature_importance[:10].index.tolist()
X_train_selected = X_train_balanced[top_features]
X_test_selected = X_test[top_features]

# Train & Evaluate Logistic Regression as a Baseline Model
log_reg = LogisticRegression()
log_reg.fit(X_train_selected, y_train_balanced)
y_pred_log_reg = log_reg.predict(X_test_selected)

print("\n📊 Logistic Regression Model Performance:")
print(classification_report(y_test, y_pred_log_reg))
print("AUC-ROC:", roc_auc_score(y_test, log_reg.predict_proba(X_test_selected)[:, 1]))

# Train & Evaluate RandomForest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_selected, y_train_balanced)
y_pred_rf = rf.predict(X_test_selected)

print("\n🌲 Random Forest Model Performance:")
print(classification_report(y_test, y_pred_rf))
print("AUC-ROC:", roc_auc_score(y_test, rf.predict_proba(X_test_selected)[:, 1]))
