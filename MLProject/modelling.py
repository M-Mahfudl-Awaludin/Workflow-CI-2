import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# ARGUMENT
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    type=str,
    default="hasil_Preprocessing_gojek.csv"
)

args = parser.parse_args()

DATA_PATH = args.data_path


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df["text_String"] = df["text_String"].fillna("")

X = df["text_String"]
y = df["label_num"]

print("Jumlah data:", len(df))
print("Distribusi label:")
print(y.value_counts())


# ============================================================
# 2. TF-IDF
# ============================================================

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_tfidf = tfidf.fit_transform(X)

print("\nUkuran fitur TF-IDF:", X_tfidf.shape)


# ============================================================
# 3. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. MLFLOW
# ============================================================

mlflow.set_tracking_uri("file:./mlruns")

mlflow.set_experiment("Analisis Sentimen Gojek")

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_Basic"):

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

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

    mlflow.log_metric(
        "custom_train_accuracy",
        train_accuracy
    )

    mlflow.log_metric(
        "custom_test_accuracy",
        test_accuracy
    )

    print(
        "\nMLflow Run ID:",
        mlflow.active_run().info.run_id
    )