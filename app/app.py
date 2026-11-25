import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ==========================================
# 1. MODEL WRAPPER AND PROJECT PATHS
# ==========================================

class NumpyLogisticRegression:
    def __init__(self, state):
        self.mean_ = np.asarray(state["mean"], dtype=float)
        self.scale_ = np.asarray(state["scale"], dtype=float)
        self.weights_ = np.asarray(state["weights"], dtype=float)
        self.bias_ = np.asarray(state["bias"], dtype=float)

    @staticmethod
    def softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        X = (X - self.mean_) / self.scale_
        return self.softmax(X @ self.weights_ + self.bias_)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed"

st.set_page_config(
    page_title="Premier League Predictor",
    page_icon="⚽",
    layout="wide"
)

LABEL_NAMES = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win",
}

# ==========================================
# 2. LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run model_training.py first."
        )

    bundle = joblib.load(MODEL_PATH)
    model_type = bundle["model_type"]
    state = bundle["model_state"]

    if model_type == "numpy_logistic":
        model = NumpyLogisticRegression(state)
    elif model_type in ("xgboost_native", "catboost"):
        # These model objects are stored directly in best_model.pkl.
        model = state
    elif model_type == "dummy":
        model = state
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return bundle, model


try:
    model_bundle, model = load_model()
except Exception as exc:
    st.error(f"Could not load the prediction model: {exc}")
    st.stop()

FEATURES = model_bundle["features"]
LABELS = model_bundle["labels"]
LABEL_TO_INT = model_bundle["label_to_int"]
MODEL_TYPE = model_bundle["model_type"]

# The model was trained with these 13 features.
EXPECTED_FEATURES = [
    "Home_Form_5",
    "Home_Form_10",
    "Home_Combined_Form",
    "Away_Form_5",
    "Away_Form_10",
    "Away_Combined_Form",
    "Home_Attack_Form",
    "Home_Defense_Form",
    "Away_Attack_Form",
    "Away_Defense_Form",
    "Form_Difference",
    "Attack_Difference",
    "Defense_Difference",
]

if list(FEATURES) != EXPECTED_FEATURES:
    st.error(
        "The feature list in best_model.pkl does not match the feature "
        "definition expected by this app. Please check model_training.py."
    )
    st.stop()

# ==========================================
# 3. LOAD CURRENT TEAM PERFORMANCE
# ==========================================

@st.cache_data
def load_team_data():
    file_path = DATA_PATH / "latest_team_performance.csv"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}. "
            "Run time_analysis.py first."
        )

    df = pd.read_csv(file_path)

    required_columns = [
        "Team", "Date", "Form_5", "Form_10", "Combined_Form",
        "Attack_Form", "Defense_Form"
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "latest_team_performance.csv is missing columns: "
            + ", ".join(missing)
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Team", "Date"]).copy()
    df = df.sort_values("Date").groupby("Team").tail(1)
    return df.reset_index(drop=True)


@st.cache_data
def load_training_medians():
    """
    Calculate imputation values from the model's historical training seasons
    only. This is used only when a current team has insufficient match history.
    """
    file_path = DATA_PATH / "EPL_features.csv"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Training feature file not found: {file_path}. "
            "The app needs EPL_features.csv to handle teams with missing "
            "Form_10 / attack / defense values."
        )

    df = pd.read_csv(file_path)

    needed = EXPECTED_FEATURES[:10] + ["Season"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(
            "EPL_features.csv is missing columns needed for imputation: "
            + ", ".join(missing)
        )

    train_rows = df[df["Season"].isin(["2024/25", "2025/26"])].copy()
    if train_rows.empty:
        raise ValueError(
            "No historical training rows found in EPL_features.csv "
            "for seasons 2024/25 and 2025/26."
        )

    medians = train_rows[EXPECTED_FEATURES[:10]].apply(
        pd.to_numeric, errors="coerce"
    ).median()

    if medians.isna().any():
        bad = medians[medians.isna()].index.tolist()
        raise ValueError(
            "Could not calculate historical medians for: " + ", ".join(bad)
        )

    return medians.to_dict()


try:
    team_data = load_team_data()
    training_medians = load_training_medians()
except Exception as exc:
    st.error(f"Could not load team data: {exc}")
    st.stop()

teams = sorted(team_data["Team"].dropna().unique().tolist())

if len(teams) < 2:
    st.error("At least two teams are required to make a prediction.")
    st.stop()

# ==========================================
# 4. PAGE HEADER AND TEAM SELECTION
# ==========================================

st.title("⚽ Premier League Match Predictor")
st.subheader("Football Analytics & Prediction System")

info_col1, info_col2 = st.columns(2)
with info_col1:
    st.success(f"Model loaded: {model_bundle.get('model_name', MODEL_TYPE)}")
with info_col2:
    st.info(f"Teams available: {len(teams)}")

st.markdown("---")
st.header("⚽ Match Prediction")
st.write(
    "Select the home and away teams. The app uses each team's latest "
    "available performance record."
)

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox(
        "🏠 Home Team",
        teams,
        index=teams.index("Arsenal") if "Arsenal" in teams else 0
    )

with col2:
    away_options = [team for team in teams if team != home_team]
    default_away_index = (
        away_options.index("Manchester City")
        if "Manchester City" in away_options else 0
    )
    away_team = st.selectbox(
        "✈️ Away Team",
        away_options,
        index=default_away_index
    )

# ==========================================
# 5. BUILD MODEL INPUT
# ==========================================

def get_team_record(team_name):
    matches = team_data[team_data["Team"] == team_name]
    if matches.empty:
        raise ValueError(f"No performance record found for {team_name}.")
    return matches.iloc[0]


def numeric_value(record, column):
    return pd.to_numeric(pd.Series([record[column]]), errors="coerce").iloc[0]


def build_prediction_features(home_name, away_name):
    home = get_team_record(home_name)
    away = get_team_record(away_name)

    # Map the latest-team CSV fields to the exact training feature names.
    feature_values = {
        "Home_Form_5": numeric_value(home, "Form_5"),
        "Home_Form_10": numeric_value(home, "Form_10"),
        "Home_Combined_Form": numeric_value(home, "Combined_Form"),
        "Away_Form_5": numeric_value(away, "Form_5"),
        "Away_Form_10": numeric_value(away, "Form_10"),
        "Away_Combined_Form": numeric_value(away, "Combined_Form"),
        "Home_Attack_Form": numeric_value(home, "Attack_Form"),
        "Home_Defense_Form": numeric_value(home, "Defense_Form"),
        "Away_Attack_Form": numeric_value(away, "Attack_Form"),
        "Away_Defense_Form": numeric_value(away, "Defense_Form"),
    }

    # Keep track of missing inputs so the user can see when imputation occurs.
    missing_features = [
        name for name, value in feature_values.items()
        if pd.isna(value)
    ]

    # Impute missing current-season statistics with the median of the
    # corresponding feature from the historical training seasons.
    for feature_name in list(feature_values):
        if pd.isna(feature_values[feature_name]):
            feature_values[feature_name] = float(training_medians[feature_name])

    # Recreate difference features in the same way as the training dataset.
    feature_values["Form_Difference"] = (
        feature_values["Home_Combined_Form"]
        - feature_values["Away_Combined_Form"]
    )
    feature_values["Attack_Difference"] = (
        feature_values["Home_Attack_Form"]
        - feature_values["Away_Attack_Form"]
    )
    feature_values["Defense_Difference"] = (
        feature_values["Home_Defense_Form"]
        - feature_values["Away_Defense_Form"]
    )

    input_df = pd.DataFrame(
        [[feature_values[name] for name in FEATURES]],
        columns=FEATURES
    ).astype(float)

    if input_df.isna().any().any():
        raise ValueError("The model input still contains missing values.")

    return input_df, missing_features, home, away


# ==========================================
# 6. PREDICT PROBABILITIES
# ==========================================

def predict_probabilities(input_df):
    if MODEL_TYPE == "numpy_logistic":
        probabilities = model.predict_proba(input_df)

    elif MODEL_TYPE == "xgboost_native":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "Install xgboost in your virtual environment: "
                "pip install xgboost"
            ) from exc

        dmatrix = xgb.DMatrix(
            input_df,
            feature_names=list(FEATURES)
        )
        probabilities = np.asarray(model.predict(dmatrix)).reshape(1, -1)

    elif MODEL_TYPE == "catboost":
        probabilities = np.asarray(model.predict_proba(input_df))

    elif MODEL_TYPE == "dummy":
        probabilities = np.zeros((1, len(LABELS)), dtype=float)
        majority_class = int(model["majority_class"])
        probabilities[0, majority_class] = 1.0

    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(1, -1)

    if probabilities.shape[1] != len(LABELS):
        raise ValueError(
            f"The model returned {probabilities.shape[1]} probabilities, "
            f"but {len(LABELS)} classes are expected."
        )

    # Normalize defensively against small floating-point rounding errors.
    row_sum = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("The model returned invalid probability values.")

    probabilities = probabilities / row_sum
    return probabilities[0]


# ==========================================
# 7. SHOW TEAM FORM AND RUN PREDICTION
# ==========================================

def show_team_summary(team_name, record):
    st.markdown(f"**{team_name}**")
    st.caption(f"Latest record date: {record['Date'].strftime('%d %b %Y')}")

    summary = pd.DataFrame({
        "Metric": [
            "Form (last 5)",
            "Form (last 10)",
            "Combined Form",
            "Attack Form",
            "Defense Form",
        ],
        "Value": [
            record["Form_5"],
            record["Form_10"],
            record["Combined_Form"],
            record["Attack_Form"],
            record["Defense_Form"],
        ],
    })
    summary["Value"] = pd.to_numeric(summary["Value"], errors="coerce")
    st.dataframe(
        summary.style.format({"Value": "{:.3f}"}, na_rep="Not available"),
        use_container_width=True,
        hide_index=True
    )


home_record = get_team_record(home_team)
away_record = get_team_record(away_team)

st.markdown("---")
st.subheader("📊 Latest Team Performance")
summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    show_team_summary(home_team, home_record)

with summary_col2:
    show_team_summary(away_team, away_record)

st.markdown("---")

if st.button("⚽ Predict Match Result", type="primary", use_container_width=True):
    try:
        input_df, missing_features, home_record, away_record = (
            build_prediction_features(home_team, away_team)
        )
        probabilities = predict_probabilities(input_df)

        predicted_index = int(np.argmax(probabilities))
        predicted_label = LABELS[predicted_index]
        predicted_name = LABEL_NAMES.get(predicted_label, predicted_label)

        st.subheader("🏆 Prediction Result")
        st.success(
            f"Predicted result: {predicted_name}"
        )

        result_cols = st.columns(3)
        probability_by_label = {
            label: float(probabilities[index])
            for index, label in enumerate(LABELS)
        }

        display_order = [
            ("H", "🏠 Home Win"),
            ("D", "🤝 Draw"),
            ("A", "✈️ Away Win"),
        ]

        for col, (label, title) in zip(result_cols, display_order):
            with col:
                st.metric(
                    title,
                    f"{probability_by_label[label] * 100:.1f}%"
                )

        chart_data = pd.DataFrame({
            "Outcome": ["Home Win", "Draw", "Away Win"],
            "Probability (%)": [
                probability_by_label["H"] * 100,
                probability_by_label["D"] * 100,
                probability_by_label["A"] * 100,
            ],
        }).set_index("Outcome")

        st.markdown("#### Probability comparison")
        st.bar_chart(chart_data, y="Probability (%)", use_container_width=True)

        if missing_features:
            st.warning(
                "Some current-season statistics were unavailable and were "
                "filled using historical training-season medians. "
                "The prediction is therefore less certain for teams with "
                "limited Premier League history."
            )
            st.caption(
                "Imputed fields: " + ", ".join(missing_features)
            )
        else:
            st.info(
                "All 10 team-performance inputs were available; no imputation "
                "was needed."
            )

        st.caption(
            "Probabilities are model estimates, not guarantees. "
            "They are based on the available dataset and selected features."
        )

    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")

# ==========================================
# 8. MODEL INFORMATION
# ==========================================

with st.expander("ℹ️ About this prediction model"):
    st.write(f"**Model:** {model_bundle.get('model_name', MODEL_TYPE)}")
    st.write(f"**Model type:** {MODEL_TYPE}")
    st.write("**Prediction classes:** Home Win (H), Draw (D), Away Win (A)")
    st.write("**Number of model features:**", len(FEATURES))
    st.write("**Feature order:**")
    st.code("\n".join(FEATURES))
