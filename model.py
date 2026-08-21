import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample


def train_model(file_path):

    # Load dataset
    df = pd.read_csv(r"customer_churn.csv")

    # -----------------------------
    # HANDLE MISSING VALUES
    # -----------------------------

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # Fill missing categorical values
    categorical_cols = (
        df.select_dtypes(include=["object"])
        .columns
        .drop("customerID")
    )

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    # -----------------------------
    # ENCODE TARGET
    # -----------------------------

    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    # Remove customer ID
    df = df.drop("customerID", axis=1)

    # One-hot encode categorical features
    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

    # -----------------------------
    # FEATURES AND TARGET
    # -----------------------------

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # -----------------------------
    # TRAIN / TEST SPLIT
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # BALANCE TRAINING DATA
    # -----------------------------

    train_data = X_train.copy()
    train_data["target"] = y_train.values

    majority = train_data[
        train_data["target"] == 0
    ]

    minority = train_data[
        train_data["target"] == 1
    ]

    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=42
    )

    balanced_train = pd.concat(
        [majority, minority_upsampled]
    )

    X_train_balanced = balanced_train.drop(
        "target",
        axis=1
    )

    y_train_balanced = balanced_train["target"]

    # -----------------------------
    # RANDOM FOREST
    # -----------------------------

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train_balanced,
        y_train_balanced
    )

    # -----------------------------
    # EVALUATION
    # -----------------------------

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return model, X.columns.tolist(), accuracy
