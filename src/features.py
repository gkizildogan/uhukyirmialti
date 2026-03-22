"""
Simetrik (taraf-bağımsız) özellik mühendisliği modülü.

Sol ve sağ IDG frekanslarından, hangi tarafın arızalı olduğuna
bağlı olmayan (side-invariant) özellikler türetir.
"""

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "abs_diff",
    "max_freq",
    "min_freq",
    "mean_freq",
    "ratio",
    "range_norm",
    "std_pair",
    "max_dev_from_400",
    "sum_dev_from_400",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham IDG frekans verisinden simetrik özellikler türetir.

    Parametreler
    ----------
    df : pd.DataFrame
        freq_left, freq_right, aircraft_id, label sütunları içermelidir.

    Dönüş
    ------
    pd.DataFrame
        Orijinal sütunlar + türetilmiş özellikler.
    """
    out = df.copy()
    fl = df["freq_left"].values
    fr = df["freq_right"].values

    out["abs_diff"] = np.abs(fl - fr)
    out["max_freq"] = np.maximum(fl, fr)
    out["min_freq"] = np.minimum(fl, fr)
    out["mean_freq"] = (fl + fr) / 2.0
    out["ratio"] = out["max_freq"] / out["min_freq"]
    out["range_norm"] = out["abs_diff"] / out["mean_freq"]
    out["std_pair"] = np.sqrt(((fl - out["mean_freq"].values) ** 2 +
                                (fr - out["mean_freq"].values) ** 2) / 2.0)
    out["max_dev_from_400"] = np.maximum(np.abs(fl - 400.0), np.abs(fr - 400.0))
    out["sum_dev_from_400"] = np.abs(fl - 400.0) + np.abs(fr - 400.0)

    return out


def get_feature_names():
    """Türetilmiş özellik isimlerini döndürür."""
    return FEATURE_NAMES.copy()
