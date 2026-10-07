import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from data import load_data
from preprocess import clean_text

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
N_TRIALS = 25
FIXED_PARAMS = {
    "model": "lr",
    "C": 10.0,
    "word_ngram": (1, 1),
    "word_max_features": 60000,
    "min_df": 3,
    "use_char": False,
    "char_ngram": (2, 4),
}

df = load_data().dropna().drop_duplicates()
df["clean_comment"] = df["clean_comment"].apply(clean_text)
df = df[df["clean_comment"] != ""]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    df["clean_comment"], df["category"], test_size=0.2, stratify=df["category"], random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.125, stratify=y_train_full, random_state=42
)

SPACE = {
    "model": ["lr", "svc"],
    "C": [0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
    "word_ngram": [(1, 1), (1, 2), (1, 3)],
    "word_max_features": [30000, 60000, 100000],
    "min_df": [1, 2, 3],
    "use_char": [True, False],
    "char_ngram": [(2, 4), (2, 5), (3, 6)],
}


def build(p):
    word = TfidfVectorizer(
        max_features=p["word_max_features"],
        ngram_range=p["word_ngram"],
        min_df=p["min_df"],
        sublinear_tf=True,
    )
    if p["use_char"]:
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=p["char_ngram"],
            max_features=120000,
            min_df=p["min_df"],
            sublinear_tf=True,
        )
        features = FeatureUnion([("word", word), ("char", char)])
    else:
        features = word

    if p["model"] == "lr":
        clf = LogisticRegression(C=p["C"], max_iter=3000, class_weight="balanced")
    else:
        clf = LinearSVC(C=p["C"], class_weight="balanced")

    return Pipeline([("features", features), ("clf", clf)])


def log(msg):
    print(msg)
    sys.stdout.flush()


mlflow.set_experiment("vibechecker")

if FIXED_PARAMS is not None:
    trials = [FIXED_PARAMS]
else:
    trials = list(ParameterSampler(SPACE, n_iter=N_TRIALS, random_state=42))

best_params, best_score = None, -1.0

for i, p in enumerate(trials):
    with mlflow.start_run(run_name="trial_" + str(i)):
        pipe = build(p)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        f1 = f1_score(y_val, preds, average="macro")
        mlflow.log_params({k: str(v) for k, v in p.items()})
        mlflow.log_metric("val_macro_f1", f1)
        mlflow.log_metric("val_accuracy", accuracy_score(y_val, preds))
        log(str(i) + " " + str(round(f1, 4)) + " " + str(p))
        if f1 > best_score:
            best_params, best_score = p, f1

log("refitting best on train + val")
final = build(best_params)
final.fit(X_train_full, y_train_full)
test_preds = final.predict(X_test)
test_f1 = f1_score(y_test, test_preds, average="macro")
test_acc = accuracy_score(y_test, test_preds)

joblib.dump(final, MODEL_DIR / "model.joblib")
log("saved models/model.joblib")
log("best params: " + str(best_params))
log("val macro_f1: " + str(round(best_score, 4)))
log("test macro_f1: " + str(round(test_f1, 4)))
log("test accuracy: " + str(round(test_acc, 4)))

log("logging final run to mlflow")
with mlflow.start_run(run_name="final_best"):
    mlflow.log_params({k: str(v) for k, v in best_params.items()})
    mlflow.log_metric("val_macro_f1", best_score)
    mlflow.log_metric("test_macro_f1", test_f1)
    mlflow.log_metric("test_accuracy", test_acc)
    log("logging model artifact")
    mlflow.sklearn.log_model(
        final,
        name="model",
        pip_requirements=["scikit-learn==" + sklearn.__version__, "pandas"],
    )
log("done")