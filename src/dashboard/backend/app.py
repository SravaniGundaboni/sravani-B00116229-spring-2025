# WEEK 5: Model Selection
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

# Initialize Models
log_reg = LogisticRegression(random_state=42, max_iter=1000)
rf_model = RandomForestClassifier(random_state=42, n_estimators=50)

# Cross-validation setup
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

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
from sklearn.model_selection import GridSearchCV
import shap

# Hyperparameter Tuning for Logistic Regression
log_reg_params = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}
log_reg_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), log_reg_params, cv=cv, scoring='f1')
log_reg_grid.fit(X_train_balanced, y_train_balanced)

# Hyperparameter Tuning for Random Forest
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=cv, scoring='f1')
rf_grid.fit(X_train_balanced, y_train_balanced)

# Best parameters
print("Best Logistic Regression Parameters:", log_reg_grid.best_params_)
print("Best Random Forest Parameters:", rf_grid.best_params_)

# Analyze Feature Importance for Random Forest
best_rf = rf_grid.best_estimator_
feature_importance = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(12, 6))
feature_importance[:10].plot(kind='bar', color='green')
plt.title("Top 10 Important Features - Random Forest")
plt.show()

# SHAP Values for Logistic Regression
best_log_reg = log_reg_grid.best_estimator_
explainer = shap.Explainer(best_log_reg, X_train_balanced)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, plot_type='bar')

# Final Model Selection Based on F1-Score
final_model = best_rf if rf_grid.best_score_ > log_reg_grid.best_score_ else best_log_reg
print("Selected Final Model:", final_model)
