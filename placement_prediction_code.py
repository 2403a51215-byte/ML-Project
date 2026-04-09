"""
College Placement Prediction System
====================================
A machine learning project to predict whether a student will be placed or not
based on academic performance and other features.

Requirements: pip install pandas numpy scikit-learn matplotlib seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# ─── 1. Load Dataset ───
df = pd.read_csv("placement_dataset.csv")
print("=" * 60)
print("COLLEGE PLACEMENT PREDICTION SYSTEM")
print("=" * 60)
print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nDataset Info:")
print(df.describe())
print(f"\nPlacement Distribution:\n{df['status'].value_counts()}")

# ─── 2. Data Preprocessing ───
# Drop serial number and salary (target leakage)
df_model = df.drop(['sl_no', 'salary'], axis=1)

# Encode categorical variables
le_dict = {}
categorical_cols = ['gender', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    le_dict[col] = le

# Features and target
X = df_model.drop('status', axis=1)
y = df_model['status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ─── 3. Model Training & Evaluation ───
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
print("\n" + "=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)

for name, model in models.items():
    use_scaled = name in ["Logistic Regression", "SVM", "KNN"]
    Xtr = X_train_scaled if use_scaled else X_train
    Xte = X_test_scaled if use_scaled else X_test
    
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {"accuracy": acc, "auc": auc, "y_pred": y_pred, "y_prob": y_prob}
    
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC-ROC:  {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Not Placed', 'Placed'])}")

# ─── 4. Visualizations ───
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('College Placement Prediction - Model Analysis', fontsize=16, fontweight='bold')

# 4a. Model Accuracy Comparison
ax = axes[0, 0]
names = list(results.keys())
accs = [results[n]["accuracy"] for n in names]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
bars = ax.barh(names, accs, color=colors)
ax.set_xlim(0.7, 1.0)
ax.set_xlabel('Accuracy')
ax.set_title('Model Accuracy Comparison')
for bar, acc in zip(bars, accs):
    ax.text(acc + 0.005, bar.get_y() + bar.get_height()/2, f'{acc:.2%}', va='center', fontsize=10)

# 4b. ROC Curves
ax = axes[0, 1]
for i, name in enumerate(names):
    fpr, tpr, _ = roc_curve(y_test, results[name]["y_prob"])
    ax.plot(fpr, tpr, color=colors[i], label=f'{name} (AUC={results[name]["auc"]:.3f})')
ax.plot([0,1],[0,1],'k--',alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend(fontsize=7)

# 4c. Confusion Matrix (best model)
best_model_name = max(results, key=lambda k: results[k]["accuracy"])
ax = axes[0, 2]
cm = confusion_matrix(y_test, results[best_model_name]["y_pred"])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Not Placed','Placed'], yticklabels=['Not Placed','Placed'])
ax.set_title(f'Confusion Matrix\n({best_model_name})')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')

# 4d. Feature Importance (Random Forest)
ax = axes[1, 0]
rf = models["Random Forest"]
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values()
feat_imp.plot(kind='barh', ax=ax, color='#2196F3')
ax.set_title('Feature Importance (Random Forest)')
ax.set_xlabel('Importance')

# 4e. Placement by Gender
ax = axes[1, 1]
ct = pd.crosstab(df['gender'], df['status'])
ct.plot(kind='bar', ax=ax, color=['#F44336','#4CAF50'])
ax.set_title('Placement by Gender')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Status')

# 4f. Score Distribution
ax = axes[1, 2]
df['avg_score'] = (df['ssc_p'] + df['hsc_p'] + df['degree_p'] + df['mba_p']) / 4
for status, color in [('Placed','#4CAF50'), ('Not Placed','#F44336')]:
    subset = df[df['status']==status]['avg_score']
    ax.hist(subset, bins=20, alpha=0.6, color=color, label=status)
ax.set_title('Average Score Distribution')
ax.set_xlabel('Average Score')
ax.legend()

plt.tight_layout()
plt.savefig('/mnt/documents/placement_analysis_charts.png', dpi=150, bbox_inches='tight')
print(f"\nCharts saved to placement_analysis_charts.png")
print(f"\nBest Model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")
