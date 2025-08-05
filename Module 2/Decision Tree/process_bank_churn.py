from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into training and validation sets.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        test_size: Proportion of data to use for validation.
        random_state: Seed for reproducibility.

    Returns:
        Tuple containing:
            - train_inputs: Training features
            - val_inputs: Validation features
            - train_targets: Training target
            - val_targets: Validation target
    """
    input_cols = [col for col in df.columns if col not in ["id", "CustomerId", "Surname", target_col]]

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state
    )

    train_inputs = train_df[input_cols].copy()
    val_inputs = val_df[input_cols].copy()
    train_targets = train_df[target_col].copy()
    val_targets = val_df[target_col].copy()

    return train_inputs, val_inputs, train_targets, val_targets


def fit_scaler(train_inputs: pd.DataFrame, numeric_cols: List[str]) -> StandardScaler:
    """
    Fit a StandardScaler on numeric columns.

    Args:
        train_inputs: Training features.
        numeric_cols: List of numeric column names.

    Returns:
        Fitted StandardScaler object.
    """
    scaler = StandardScaler()
    scaler.fit(train_inputs[numeric_cols])
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    numeric_cols: List[str],
    scaler: StandardScaler
) -> pd.DataFrame:
    """
    Apply a fitted scaler to a dataset.

    Args:
        df: Input DataFrame.
        numeric_cols: List of numeric column names.
        scaler: Fitted StandardScaler object.

    Returns:
        DataFrame with scaled numeric columns.
    """
    df = df.copy()
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df


def fit_encoder(train_inputs: pd.DataFrame, categorical_cols: List[str]) -> OneHotEncoder:
    """
    Fit a OneHotEncoder on categorical columns.

    Args:
        train_inputs: Training features.
        categorical_cols: List of categorical column names.

    Returns:
        Fitted OneHotEncoder object.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    encoder.fit(train_inputs[categorical_cols])
    return encoder


def apply_encoder(
    df: pd.DataFrame,
    categorical_cols: List[str],
    encoder: OneHotEncoder
) -> pd.DataFrame:
    """
    Apply a fitted encoder to a dataset.

    Args:
        df: Input DataFrame.
        categorical_cols: List of categorical column names.
        encoder: Fitted OneHotEncoder object.

    Returns:
        DataFrame with encoded categorical columns appended.
    """
    df = df.copy()
    encoded = encoder.transform(df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
    df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
    return df


def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "Exited",
    scaler_numeric: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[StandardScaler], OneHotEncoder]:
    """
    Preprocess training and validation data.

    Args:
        df: Input DataFrame.
        target_col: Name of target column.
        scaler_numeric: Whether to scale numeric columns.

    Returns:
        Tuple containing:
            - train_inputs: Preprocessed training features
            - val_inputs: Preprocessed validation features
            - train_targets: Training target
            - val_targets: Validation target
            - scaler: Fitted scaler (or None if not used)
            - encoder: Fitted encoder
    """
    train_inputs, val_inputs, train_targets, val_targets = split_data(df, target_col)

    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

    scaler = None
    if scaler_numeric and numeric_cols:
        scaler = fit_scaler(train_inputs, numeric_cols)
        train_inputs = apply_scaler(train_inputs, numeric_cols, scaler)
        val_inputs = apply_scaler(val_inputs, numeric_cols, scaler)

    encoder = fit_encoder(train_inputs, categorical_cols)
    train_inputs = apply_encoder(train_inputs, categorical_cols, encoder)
    val_inputs = apply_encoder(val_inputs, categorical_cols, encoder)

    return train_inputs, val_inputs, train_targets, val_targets, scaler, encoder


def preprocess_new_data(
    new_data: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaler: Optional[StandardScaler],
    encoder: OneHotEncoder
) -> pd.DataFrame:
    """
    Preprocess new/unseen data using fitted scaler and encoder.

    Args:
        new_data: New data to preprocess.
        numeric_cols: List of numeric columns.
        categorical_cols: List of categorical columns.
        scaler: Previously fitted scaler (or None if scaling not used).
        encoder: Previously fitted encoder.

    Returns:
        Preprocessed DataFrame ready for model inference.
    """
    df = new_data.copy()

    if scaler is not None and numeric_cols:
        df = apply_scaler(df, numeric_cols, scaler)

    df = apply_encoder(df, categorical_cols, encoder)
    return df
