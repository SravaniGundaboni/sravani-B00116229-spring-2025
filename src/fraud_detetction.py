# Week 1: Project Setup & Data Acquisition

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
file_path = "/content/creditcard.csv"
df = pd.read_csv(file_path)

# Display Dataset Information
print("Dataset Overview:")
print(df.info())  # Check column names, data types
print(df.describe())  # Summary statistics

# Check for Missing Values
print("\nMissing Values Per Column:")
print(df.isnull().sum())


# Week 2: Exploratory Data Analysis (EDA)
#Understand the data and identify potential issues
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

# WEEK 5: Model Selection
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

# Initialize Models
log_reg = LogisticRegression(random_state=42, max_iter=1000)
rf_model = RandomForestClassifier(random_state=42, n_estimators=50)

# Cross-validation setup (Reduced folds from 5 to 3)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # CHANGED

# Function to evaluate model
def evaluate_model(model, X, y):
    scores = {
        'Precision': cross_val_score(model, X, y, cv=cv, scoring='precision').mean(),
        'Recall': cross_val_score(model, X, y, cv=cv, scoring='recall').mean(),
        'F1-Score': cross_val_score(model, X, y, cv=cv, scoring='f1').mean(),
        'Accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean(),
        'AUC-ROC': cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()
    }
    return scores

# Evaluate Models
log_reg_scores = evaluate_model(log_reg, X_train_balanced, y_train_balanced)
rf_scores = evaluate_model(rf_model, X_train_balanced, y_train_balanced)

print("Logistic Regression Cross-Validation Scores:", log_reg_scores)
print("Random Forest Cross-Validation Scores:", rf_scores)
# WEEK 6: Model Training
from sklearn.metrics import confusion_matrix, roc_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Retrain models on full balanced training set
log_reg.fit(X_train_balanced, y_train_balanced)
rf_model.fit(X_train_balanced, y_train_balanced)

# Evaluate on the test set
y_pred_log_reg = log_reg.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Function for Evaluation Metrics
def test_evaluation(y_true, y_pred, model_name):
    print(f"\n{model_name} Test Evaluation:")
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1-Score:", f1_score(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("AUC-ROC:", roc_auc_score(y_true, y_pred))

# Evaluate both models
test_evaluation(y_test, y_pred_log_reg, "Logistic Regression")
test_evaluation(y_test, y_pred_rf, "Random Forest")

# Visualizing Confusion Matrix
for model_name, y_pred in zip(["Logistic Regression", "Random Forest"], [y_pred_log_reg, y_pred_rf]):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix: {model_name}")
    plt.show()

# ROC Curve Plot
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr_log_reg, tpr_log_reg, label='Logistic Regression')
plt.plot(fpr_rf, tpr_rf, label='Random Forest')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
# WEEK 7: Model Optimization
# Optimize hyperparameters and finalize the model
from sklearn.model_selection import RandomizedSearchCV  # CHANGED
import shap
from sklearn.utils import resample

# Subsample the data for faster hyperparameter tuning (20% of the data)
X_train_small, y_train_small = resample(X_train_balanced, y_train_balanced, n_samples=int(0.2 * len(X_train_balanced)), random_state=42)  # CHANGED

# Hyperparameter Tuning for Logistic Regression (Reduced parameter space)
log_reg_params = {
    'C': [0.1, 1, 10],  # Reduced from 4 to 3 values
    'solver': ['liblinear']  # Reduced from 2 to 1 value
}
log_reg_random = RandomizedSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_distributions=log_reg_params,
    n_iter=5,  # Evaluate only 5 random combinations
    cv=cv,
    scoring='f1',
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)
log_reg_random.fit(X_train_small, y_train_small)  # CHANGED: Use subsampled data

# Hyperparameter Tuning for Random Forest (Reduced parameter space)
rf_params = {
    'n_estimators': [100, 200],  # Reduced from 3 to 2 values
    'max_depth': [10, 20],       # Reduced from 3 to 2 values
    'min_samples_split': [2, 5]  # Reduced from 3 to 2 values
}
rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=rf_params,
    n_iter=5,  # Evaluate only 5 random combinations
    cv=cv,
    scoring='f1',
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)
rf_random.fit(X_train_small, y_train_small)  # CHANGED: Use subsampled data

# Best parameters
print("Best Logistic Regression Parameters:", log_reg_random.best_params_)
print("Best Random Forest Parameters:", rf_random.best_params_)

# Analyze Feature Importance for Random Forest
best_rf = rf_random.best_estimator_
feature_importance = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(12, 6))
feature_importance[:10].plot(kind='bar', color='green')
plt.title("Top 10 Important Features - Random Forest")
plt.show()

# SHAP Values for Logistic Regression (Using a subset of test data)
best_log_reg = log_reg_random.best_estimator_
explainer = shap.Explainer(best_log_reg, X_train_small)  # CHANGED: Use subsampled data
shap_values = explainer(X_test[:100])  # CHANGED: Use only 100 samples
shap.summary_plot(shap_values, X_test[:100], plot_type='bar')  # CHANGED: Use only 100 samples

# Final Model Selection Based on F1-Score
final_model = best_rf if rf_random.best_score_ > log_reg_random.best_score_ else best_log_reg
print("Selected Final Model:", final_model)


# Model Comparison
import pandas as pd
import matplotlib.pyplot as plt

# Evaluate both models on the test set
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    scores = {
        'Model': model_name,
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }
    return scores

# Evaluate Logistic Regression
log_reg_scores = evaluate_model(log_reg, X_test, y_test, "Logistic Regression")

# Evaluate Random Forest
rf_scores = evaluate_model(rf_model, X_test, y_test, "Random Forest")

# Create a DataFrame for comparison
comparison_df = pd.DataFrame([log_reg_scores, rf_scores])
print("Model Comparison:")
print(comparison_df)

# Visualize the comparison
metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'AUC-ROC']
comparison_df.set_index('Model')[metrics].plot(kind='bar', figsize=(12, 6))
plt.title('Model Comparison')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.show()

# Select the best model based on F1-Score
best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
print(f"\nSelected Best Model: {best_model}")

#Week8 Save predictions to CSV
import pandas as pd
predictions_df = pd.DataFrame({
    'Transaction_ID': range(len(X_test)),  # Unique ID for each transaction
    'Time': X_test['Time'],  # Assuming 'Time' is a column in X_test
    'Amount': X_test['Amount'],  # Assuming 'Amount' is a column in X_test
    'Prediction': final_model.predict(X_test),  # Predictions from the final model
    'Actual_Class': y_test  # Actual labels from the test set
})

# Save to CSV
predictions_df.to_csv('fraud_predictions.csv', index=False)
print("Predictions saved to 'fraud_predictions.csv'")
