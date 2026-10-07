from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data import load_data
from preprocess import clean_text

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = load_data().dropna().drop_duplicates()
df["clean_comment"] = df["clean_comment"].apply(clean_text)
df = df[df["clean_comment"] != ""]

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_comment"], df["category"], test_size=0.2, stratify=df["category"], random_state=42
)

params = {"max_features": 20000, "ngram_range": (1, 2), "C": 1.0}

mlflow.set_experiment("vibechecker")

with mlflow.start_run():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=params["max_features"], ngram_range=params["ngram_range"])),
        ("clf", LogisticRegression(C=params["C"], max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    mlflow.log_params({k: str(v) for k, v in params.items()})
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("macro_f1", f1)

    report = classification_report(y_test, preds, output_dict=True)
    names = {"-1": "negative", "0": "neutral", "1": "positive"}
    for key, name in names.items():
        mlflow.log_metric("f1_" + name, report[key]["f1-score"])
        mlflow.log_metric("recall_" + name, report[key]["recall"])

    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))

    mlflow.sklearn.log_model(pipe, name="model")

    joblib.dump(pipe, MODEL_DIR / "model.joblib")
    print("accuracy:", round(acc, 4), "macro_f1:", round(f1, 4))