
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from catboost import CatBoostClassifier
from pathlib import Path

# ==========================================
# 1. SETUP
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "EPL_features.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["H", "D", "A"]
LABEL_TO_INT = {label: i for i, label in enumerate(LABELS)}
INT_TO_LABEL = {i: label for label, i in LABEL_TO_INT.items()}

# ==========================================
# 2. METRICS USING NUMPY (NO SCIKIT-LEARN)
# ==========================================
def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    confusion = np.zeros((3, 3), dtype=int)

    for actual, predicted in zip(y_true, y_pred):
        confusion[actual, predicted] += 1

    recalls, f1_scores = [], []
    for class_id in range(3):
        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "Accuracy": float(np.mean(y_true == y_pred)),
        "Balanced_Accuracy": float(np.mean(recalls)),
        "Macro_F1": float(np.mean(f1_scores)),
        "Confusion_Matrix": confusion,
    }


# ==========================================
# 3. NUMPY MULTICLASS LOGISTIC REGRESSION
# ==========================================
class NumpyLogisticRegression:
    def __init__(self, learning_rate=0.05, iterations=3000, l2=0.01):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2 = l2

    @staticmethod
    def softmax(scores):
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        X = (X - self.mean_) / self.scale_

        n, p = X.shape
        k = len(LABELS)
        counts = np.bincount(y, minlength=k)
        class_weights = np.zeros(k)
        present = counts > 0
        class_weights[present] = n / (k * counts[present])
        sample_weights = class_weights[y]
        weight_sum = sample_weights.sum()
        y_onehot = np.eye(k)[y]

        self.weights_ = np.zeros((p, k))
        self.bias_ = np.zeros(k)

        for _ in range(self.iterations):
            probabilities = self.softmax(X @ self.weights_ + self.bias_)
            errors = (probabilities - y_onehot) * sample_weights[:, None]
            grad_w = X.T @ errors / weight_sum + self.l2 * self.weights_
            grad_b = errors.sum(axis=0) / weight_sum
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        X = (X - self.mean_) / self.scale_
        return self.softmax(X @ self.weights_ + self.bias_)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def to_state(self):
        return {
            "mean": self.mean_,
            "scale": self.scale_,
            "weights": self.weights_,
            "bias": self.bias_,
        }


# ==========================================
# 4. LOAD AND PREPARE DATA
# ==========================================
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}. Run feature_engineering.py first."
    )

data = pd.read_csv(DATA_PATH)
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

print("Dataset loaded:", len(data), "matches")
print("\nSeason distribution:")
print(data["Season"].value_counts())

FEATURES = [
    "Home_Form_5", "Home_Form_10", "Home_Combined_Form",
    "Away_Form_5", "Away_Form_10", "Away_Combined_Form",
    "Home_Attack_Form", "Home_Defense_Form",
    "Away_Attack_Form", "Away_Defense_Form",
    "Form_Difference", "Attack_Difference", "Defense_Difference",
]
TARGET = "FTR"

required = FEATURES + [TARGET, "Season"]
missing = [col for col in required if col not in data.columns]
if missing:
    raise ValueError("Missing columns: " + ", ".join(missing))

data = data.dropna(subset=required).copy()
data = data[data[TARGET].isin(LABELS)].copy()

train_data = data[data["Season"].isin(["2024/25", "2025/26"])].copy()
test_data = data[data["Season"] == "2026/27"].copy()

if train_data.empty or test_data.empty:
    raise ValueError(
        "Training or testing dataset is empty. Check the Season values."
    )

X_train = train_data[FEATURES].astype(float)
X_test = test_data[FEATURES].astype(float)
y_train = train_data[TARGET].map(LABEL_TO_INT).to_numpy(dtype=int)
y_test = test_data[TARGET].map(LABEL_TO_INT).to_numpy(dtype=int)

print("\nTRAINING MATCHES:", len(X_train))
print("TESTING MATCHES:", len(X_test))
print("\nTraining result distribution:")
print(train_data[TARGET].value_counts())
print("\nTesting result distribution:")
print(test_data[TARGET].value_counts())

counts = np.bincount(y_train, minlength=3)
class_weights = np.zeros(3, dtype=float)
present = counts > 0
class_weights[present] = len(y_train) / (3 * counts[present])
sample_weights = class_weights[y_train]

# ==========================================
# 5. BASELINE
# ==========================================
results = []
majority_class = int(np.bincount(y_train, minlength=3).argmax())
dummy_predictions = np.full(len(X_test), majority_class, dtype=int)
dummy_metrics = calculate_metrics(y_test, dummy_predictions)

results.append({
    "Model": "Dummy Baseline",
    "Accuracy": dummy_metrics["Accuracy"],
    "Balanced_Accuracy": dummy_metrics["Balanced_Accuracy"],
    "Macro_F1": dummy_metrics["Macro_F1"],
})
print("\nDummy Baseline completed.")

# ==========================================
# 6. LOGISTIC REGRESSION (NUMPY)
# ==========================================
print("\n" + "=" * 55)
print("MODEL: Logistic Regression (NumPy)")
print("=" * 55)

logistic_model = NumpyLogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_prob = logistic_model.predict_proba(X_test)
logistic_pred = np.argmax(logistic_prob, axis=1)
logistic_metrics = calculate_metrics(y_test, logistic_pred)

print("Accuracy:", round(logistic_metrics["Accuracy"], 4))
print("Balanced Accuracy:", round(logistic_metrics["Balanced_Accuracy"], 4))
print("Macro F1:", round(logistic_metrics["Macro_F1"], 4))
print("Confusion Matrix:\n", logistic_metrics["Confusion_Matrix"])

results.append({
    "Model": "Logistic Regression",
    "Accuracy": logistic_metrics["Accuracy"],
    "Balanced_Accuracy": logistic_metrics["Balanced_Accuracy"],
    "Macro_F1": logistic_metrics["Macro_F1"],
})

# ==========================================
# 7. XGBOOST (NATIVE API, NOT SKLEARN WRAPPER)
# ==========================================
print("\n" + "=" * 55)
print("MODEL: XGBoost")
print("=" * 55)

dtrain = xgb.DMatrix(
    X_train, label=y_train, weight=sample_weights, feature_names=FEATURES
)
dtest = xgb.DMatrix(X_test, feature_names=FEATURES)

xgb_params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "tree_method": "hist",
    "seed": 42,
    "nthread": -1,
}
xgb_model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=300,
    verbose_eval=False,
)
xgb_prob = np.asarray(xgb_model.predict(dtest)).reshape(-1, 3)
xgb_pred = np.argmax(xgb_prob, axis=1)
xgb_metrics = calculate_metrics(y_test, xgb_pred)

print("Accuracy:", round(xgb_metrics["Accuracy"], 4))
print("Balanced Accuracy:", round(xgb_metrics["Balanced_Accuracy"], 4))
print("Macro F1:", round(xgb_metrics["Macro_F1"], 4))
print("Confusion Matrix:\n", xgb_metrics["Confusion_Matrix"])

results.append({
    "Model": "XGBoost",
    "Accuracy": xgb_metrics["Accuracy"],
    "Balanced_Accuracy": xgb_metrics["Balanced_Accuracy"],
    "Macro_F1": xgb_metrics["Macro_F1"],
})

# ==========================================
# 8. CATBOOST
# ==========================================
print("\n" + "=" * 55)
print("MODEL: CatBoost")
print("=" * 55)

catboost_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.03,
    loss_function="MultiClass",
    eval_metric="MultiClass",
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
    thread_count=-1,
)
catboost_model.fit(X_train, y_train, sample_weight=sample_weights)
catboost_prob = np.asarray(catboost_model.predict_proba(X_test))
catboost_pred = np.argmax(catboost_prob, axis=1)
catboost_metrics = calculate_metrics(y_test, catboost_pred)

print("Accuracy:", round(catboost_metrics["Accuracy"], 4))
print("Balanced Accuracy:", round(catboost_metrics["Balanced_Accuracy"], 4))
print("Macro F1:", round(catboost_metrics["Macro_F1"], 4))
print("Confusion Matrix:\n", catboost_metrics["Confusion_Matrix"])

results.append({
    "Model": "CatBoost",
    "Accuracy": catboost_metrics["Accuracy"],
    "Balanced_Accuracy": catboost_metrics["Balanced_Accuracy"],
    "Macro_F1": catboost_metrics["Macro_F1"],
})

# ==========================================
# 9. COMPARE MODELS
# ==========================================
results_df = pd.DataFrame(results).sort_values(
    "Macro_F1", ascending=False
).reset_index(drop=True)

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)
print(results_df.to_string(index=False))

results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

# ==========================================
# 10. SAVE BEST MODEL
# ==========================================
best_model_name = results_df.iloc[0]["Model"]

if best_model_name == "Dummy Baseline":
    best_model_type = "dummy"
    best_model_state = {"majority_class": majority_class}
elif best_model_name == "Logistic Regression":
    best_model_type = "numpy_logistic"
    best_model_state = logistic_model.to_state()
elif best_model_name == "XGBoost":
    best_model_type = "xgboost_native"
    best_model_state = xgb_model
else:
    best_model_type = "catboost"
    best_model_state = catboost_model

model_bundle = {
    "model_name": best_model_name,
    "model_type": best_model_type,
    "model_state": best_model_state,
    "features": FEATURES,
    "labels": LABELS,
    "label_to_int": LABEL_TO_INT,
}

joblib.dump(model_bundle, MODEL_DIR / "best_model.pkl")
print("\nBest model:", best_model_name)
print("Best model saved successfully!")

# ==========================================
# 11. SAVE TEST PREDICTIONS
# ==========================================
if best_model_type == "dummy":
    best_prob = np.zeros((len(X_test), 3), dtype=float)
    best_prob[:, majority_class] = 1.0
elif best_model_type == "numpy_logistic":
    best_prob = logistic_prob
elif best_model_type == "xgboost_native":
    best_prob = xgb_prob
else:
    best_prob = catboost_prob

best_pred = np.argmax(best_prob, axis=1)

prediction_results = test_data[
    ["Date", "HomeTeam", "AwayTeam", "FTR"]
].copy()
prediction_results["Predicted_FTR"] = [
    INT_TO_LABEL[int(value)] for value in best_pred
]
prediction_results["Probability_Home_Win"] = best_prob[:, 0]
prediction_results["Probability_Draw"] = best_prob[:, 1]
prediction_results["Probability_Away_Win"] = best_prob[:, 2]

prediction_results.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False,
)

print("\nTest predictions saved!")
print("\nMODEL TRAINING COMPLETED!")
