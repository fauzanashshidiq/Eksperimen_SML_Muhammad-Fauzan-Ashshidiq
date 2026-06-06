from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "heart_failure_detection_raw" / "heart_failure_detection.csv"
OUTPUT_DIR = ROOT_DIR / "preprocessing" / "heart_failure_prediction_preprocessing"
TARGET_COLUMN = "HeartDisease"
RANDOM_STATE = 42


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    data = pd.read_csv(path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Kolom target '{TARGET_COLUMN}' tidak ditemukan.")

    return data


def preprocess_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    data = data.copy()
    data = data.drop_duplicates()

    numeric_columns = data.drop(columns=[TARGET_COLUMN]).select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = data.drop(columns=[TARGET_COLUMN]).select_dtypes(include=["object", "category"]).columns.tolist()

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", make_one_hot_encoder(), categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_preprocessed, columns=feature_names)

    return X_train_df, X_test_df, y_train.reset_index(drop=True), y_test.reset_index(drop=True), preprocessor


def save_outputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = X_train.copy()
    train_data[TARGET_COLUMN] = y_train

    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test

    full_data = pd.concat([train_data, test_data], ignore_index=True)

    train_data.to_csv(output_dir / "train_preprocessed.csv", index=False)
    test_data.to_csv(output_dir / "test_preprocessed.csv", index=False)
    full_data.to_csv(output_dir / "heart_failure_detection_preprocessed.csv", index=False)
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")


def main() -> None:
    data = load_data()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data)
    save_outputs(X_train, X_test, y_train, y_test, preprocessor)

    print("Preprocessing selesai.")
    print(f"Output tersimpan di: {OUTPUT_DIR}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")


if __name__ == "__main__":
    main()
