import argparse

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# 1. ARGUMENT PARSER
# ============================================================

parser = argparse.ArgumentParser(
    description="Training Sentiment Analysis Gojek"
)

parser.add_argument(
    "--data_path",
    type=str,
    default="hasil_Preprocessing_gojek.csv",
    help="Path dataset preprocessing"
)

args = parser.parse_args()

DATA_PATH = args.data_path


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("LOAD DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

# Pastikan kolom teks tidak memiliki NaN
df["text_String"] = df["text_String"].fillna("")

X = df["text_String"]
y = df["label_num"]

print(f"Jumlah data: {len(df)}")

print("\nDistribusi label:")
print(y.value_counts())

print("\n")


# ============================================================
# 3. TF-IDF
# ============================================================

print("=" * 60)
print("TF-IDF")
print("=" * 60)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_tfidf = tfidf.fit_transform(X)

print(f"Ukuran fitur TF-IDF: {X_tfidf.shape}")
print("\n")


# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

print("=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Data training : {X_train.shape[0]}")
print(f"Data testing  : {X_test.shape[0]}")
print("\n")


# ============================================================
# 5. MLFLOW AUTOLOG
# ============================================================

# Jangan menggunakan:
# mlflow.start_run()
#
# Karena mlflow run . sudah membuat active run.

mlflow.sklearn.autolog(
    log_input_examples=False,
    log_model_signatures=True,
    log_models=True
)


# ============================================================
# 6. RANDOM FOREST
# ============================================================

print("=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

# Training
model.fit(
    X_train,
    y_train
)


# ============================================================
# 7. PREDICTION
# ============================================================

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# ============================================================
# 8. EVALUATION
# ============================================================

train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

print("\n===== HASIL BASIC =====")
print(f"Training Accuracy : {train_accuracy:.4f}")
print(f"Testing Accuracy  : {test_accuracy:.4f}")


# ============================================================
# 9. CUSTOM MLFLOW METRICS
# ============================================================

mlflow.log_metric(
    "custom_train_accuracy",
    train_accuracy
)

mlflow.log_metric(
    "custom_test_accuracy",
    test_accuracy
)


# ============================================================
# 10. INFORMATION MLFLOW RUN
# ============================================================

active_run = mlflow.active_run()

if active_run is not None:
    print("\n" + "=" * 60)
    print("MLFLOW INFORMATION")
    print("=" * 60)

    print(f"Run ID       : {active_run.info.run_id}")
    print(f"Experiment ID: {active_run.info.experiment_id}")

else:
    print("\nWarning: MLflow active run tidak ditemukan.")


print("\nTraining selesai.")