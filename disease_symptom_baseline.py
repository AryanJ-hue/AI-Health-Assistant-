import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
DATA_PATH = "C:/PROJECT/Final_Augmented_dataset_Diseases_and_Symptoms.csv" 
LABEL_COL = "diseases"
MIN_SAMPLES_PER_CLASS = 50
MAX_CLASSES = None  
df = pd.read_csv(DATA_PATH)
print(f"Raw shape: {df.shape}")
print(f"Raw class count: {df[LABEL_COL].nunique()}")
print("\nMissing values per column (top 10):")
print(df.isnull().sum().sort_values(ascending=False).head(10))

n_dupes = df.duplicated().sum()
print(f"\nDuplicate rows: {n_dupes}")
df = df.drop_duplicates()
print(f"Shape after dropping duplicates: {df.shape}")
symptom_cols = [c for c in df.columns if c != LABEL_COL]
non_binary = [c for c in symptom_cols if not set(df[c].unique()).issubset({0, 1})]
if non_binary:
    print(f"\nWARNING: {len(non_binary)} symptom columns are not strictly binary: {non_binary[:10]}")
else:
    print("\nAll symptom columns confirmed binary (0/1).")
all_zero_mask = (df[symptom_cols].sum(axis=1) == 0)
print(f"Rows with all symptoms == 0: {all_zero_mask.sum()}")
df = df[~all_zero_mask]
class_counts = df[LABEL_COL].value_counts()
print(f"\nClass count distribution:\n{class_counts.describe()}")

kept_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index
if MAX_CLASSES is not None:
    kept_classes = class_counts.loc[kept_classes].nlargest(MAX_CLASSES).index

dropped_classes = set(class_counts.index) - set(kept_classes)
print(f"\nDropping {len(dropped_classes)} classes below {MIN_SAMPLES_PER_CLASS} samples "
      f"(and beyond MAX_CLASSES cap if set).")

df = df[df[LABEL_COL].isin(kept_classes)].copy()
print(f"Final shape: {df.shape}")
print(f"Final class count: {df[LABEL_COL].nunique()}")
dataset_card = pd.DataFrame({
    "raw_rows": [len(pd.read_csv(DATA_PATH))],
    "raw_classes": [pd.read_csv(DATA_PATH)[LABEL_COL].nunique()],
    "final_rows": [len(df)],
    "final_classes": [df[LABEL_COL].nunique()],
    "min_samples_per_class_threshold": [MIN_SAMPLES_PER_CLASS],
})
print("\nDataset card:")
print(dataset_card.to_string(index=False))
dataset_card.to_csv("dataset_card.csv", index=False)

X = df[symptom_cols]
y = df[LABEL_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "DecisionTree": DecisionTreeClassifier(
        class_weight="balanced", random_state=RANDOM_STATE
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=100, max_depth=30, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=2
    ),
}

import os
if os.environ.get("SKIP_RF") == "1":
    models.pop("RandomForest", None)

results = []
fitted_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    fitted_models[name] = model

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    f1_macro_test = f1_score(y_test, y_pred_test, average="macro")
    f1_weighted_test = f1_score(y_test, y_pred_test, average="weighted")

    print(f"  Train accuracy: {acc_train:.4f}")
    print(f"  Test accuracy:  {acc_test:.4f}")
    print(f"  Test macro-F1:  {f1_macro_test:.4f}   (this matters more than accuracy here)")
    print(f"  Test weighted-F1: {f1_weighted_test:.4f}")

    results.append({
        "model": name,
        "train_accuracy": acc_train,
        "test_accuracy": acc_test,
        "test_macro_f1": f1_macro_test,
        "test_weighted_f1": f1_weighted_test,
    })

results_df = pd.DataFrame(results).sort_values("test_macro_f1", ascending=False)
print("\n=== Model comparison (Table 3 material) ===")
print(results_df.to_string(index=False))
results_df.to_csv("model_comparison.csv", index=False)

best_model_name = results_df.iloc[0]["model"]
best_model = models[best_model_name]

if os.environ.get("SKIP_CV") == "1":
    print(f"\nSkipping CV (SKIP_CV=1). Best model: {best_model_name}")
else:
    print(f"\nRunning 5-fold stratified CV on best model: {best_model_name}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
    print(f"CV macro-F1 scores: {cv_scores}")
    print(f"CV macro-F1 mean +/- std: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

best_fitted = fitted_models[best_model_name]
y_pred_best = best_fitted.predict(X_test)

print(f"\n=== Classification report: {best_model_name} ===")
report = classification_report(y_test, y_pred_best, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).T
print(report_df.to_string())
report_df.to_csv("classification_report.csv")

per_class = report_df.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
worst_classes = per_class.sort_values("f1-score").head(15)
print("\n=== Worst 15 classes by F1 (start here for failure analysis) ===")
print(worst_classes[["precision", "recall", "f1-score", "support"]].to_string())
worst_classes.to_csv("worst_classes.csv")

top_n_labels = class_counts.loc[kept_classes].nlargest(20).index.tolist()
mask = y_test.isin(top_n_labels)
cm = confusion_matrix(y_test[mask], y_pred_best[mask], labels=top_n_labels)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=top_n_labels, yticklabels=top_n_labels)
plt.title(f"Confusion Matrix — {best_model_name} (top 20 classes by support)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=90, fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig("confusion_matrix_top20.png", dpi=150)

import joblib
joblib.dump(best_fitted, "best_model.joblib")
joblib.dump(symptom_cols, "symptom_columns.joblib")
print(f"\nSaved best model ({best_model_name}) to best_model.joblib")
print("Saved symptom_columns.joblib (needed to build the input vector at inference time)")

print("\nSaved: dataset_card.csv, model_comparison.csv, classification_report.csv, "
      "worst_classes.csv, confusion_matrix_top20.png, best_model.joblib, symptom_columns.joblib")
