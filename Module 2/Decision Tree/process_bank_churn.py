from typing import Tuple, Optional
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
    """Split dataset into training and validation sets."""
    input_cols = [col for col in df.columns if col not in ["id", "CustomerId", "Surname", target_col]]
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state
    )
    return (
        train_df[input_cols].copy(),
        val_df[input_cols].copy(),
        train_df[target_col].copy(),
        val_df[target_col].copy()
    )


def fit_scaler(train_inputs: pd.DataFrame, numeric_cols: list[str]) -> StandardScaler:
    """Fit a StandardScaler and store column names."""
    scaler = StandardScaler()
    scaler.fit(train_inputs[numeric_cols])
    scaler.feature_names_in_ = numeric_cols  # store numeric columns
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply a fitted scaler to its stored numeric columns."""
    df = df.copy()
    df[scaler.feature_names_in_] = scaler.transform(df[scaler.feature_names_in_])
    return df


def fit_encoder(train_inputs: pd.DataFrame, categorical_cols: list[str]) -> OneHotEncoder:
    """Fit a OneHotEncoder and store column names."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    encoder.fit(train_inputs[categorical_cols])
    encoder.feature_names_in_ = categorical_cols  # store categorical columns
    return encoder


def apply_encoder(df: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    """Apply a fitted encoder to its stored categorical columns."""
    df = df.copy()
    encoded = encoder.transform(df[encoder.feature_names_in_])
    encoded_cols = list(encoder.get_feature_names_out(encoder.feature_names_in_))
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
    return pd.concat([df.drop(columns=encoder.feature_names_in_), encoded_df], axis=1)


def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "Exited",
    scaler_numeric: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[StandardScaler], OneHotEncoder]:
    """
    Preprocess training and validation data.

    Returns:
        train_inputs, val_inputs, train_targets, val_targets,
        fitted scaler (or None),
        fitted encoder
    """
    train_inputs, val_inputs, train_targets, val_targets = split_data(df, target_col)

    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

    scaler = None
    if scaler_numeric and numeric_cols:
        scaler = fit_scaler(train_inputs, numeric_cols)
        train_inputs = apply_scaler(train_inputs, scaler)
        val_inputs = apply_scaler(val_inputs, scaler)

    encoder = fit_encoder(train_inputs, categorical_cols)
    train_inputs = apply_encoder(train_inputs, encoder)
    val_inputs = apply_encoder(val_inputs, encoder)

    return train_inputs, val_inputs, train_targets, val_targets, scaler, encoder


def preprocess_new_data(
    new_data: pd.DataFrame,
    scaler: Optional[StandardScaler],
    encoder: OneHotEncoder
) -> pd.DataFrame:
    """
    Preprocess new/unseen data using fitted scaler and encoder.
    Automatically detects numeric/categorical columns based on stored training columns.

    Args:
        new_data: New data to preprocess.
        scaler: Previously fitted scaler (or None if scaling not used).
        encoder: Previously fitted encoder.

    Returns:
        Preprocessed DataFrame ready for inference.
    """
    df = new_data.copy()

    # Apply scaling using stored numeric columns
    if scaler is not None:
        df = apply_scaler(df, scaler)

    # Apply encoding using stored categorical columns
    df = apply_encoder(df, encoder)

    return df
