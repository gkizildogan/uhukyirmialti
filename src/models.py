"""
Modelleme, Leave-One-Out CV, metrik hesaplama ve görselleştirme modülü.

6 sınıflandırma modeli (XGBoost, LightGBM, Lojistik Regresyon,
Random Forest, SVM, Extra Trees) ile filo düzeyinde LOOCV.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Metrikler
# ---------------------------------------------------------------------------

def calculate_metrics(y_true, y_pred, y_prob=None):
    """Kapsamlı sınıflandırma metrikleri hesaplar."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        metrics["ROC AUC"] = roc_auc_score(y_true, y_prob)
        metrics["PR AUC"] = average_precision_score(y_true, y_prob)
    return metrics


# ---------------------------------------------------------------------------
# Model tanımları
# ---------------------------------------------------------------------------

def _get_pos_weight(y):
    """Pozitif/negatif sınıf ağırlığını hesaplar."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return n_neg / max(n_pos, 1)


def get_models(pos_weight=3.0):
    """Tüm temel modelleri sözlük olarak döndürür."""
    return {
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            scale_pos_weight=pos_weight, eval_metric="logloss",
            use_label_encoder=False, verbosity=0, random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            scale_pos_weight=pos_weight, verbose=-1, random_state=42,
            n_jobs=-1,

        ),
        "Lojistik Regresyon": LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=42,
            n_jobs=-1,
        ),
        "Linear SVM": LinearSVC(
            class_weight="balanced", max_iter=1000, random_state=42,
            
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=42,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Leave-One-Out CV (uçak bazlı)
# ---------------------------------------------------------------------------

def leave_one_out_cv(df, feature_cols, label_col, model, only_failing=True):
    """
    Filo düzeyinde Leave-One-Out Cross Validation.

    Her fold'da bir uçak test, geri kalanı eğitim. Her fold'da yeni RobustScaler.

    Parametreler
    ----------
    only_failing : bool
        True ise yalnızca arızalı uçaklar (pozitif örnek içeren) test edilir.
    """
    aircraft_ids = df["aircraft_id"].unique()
    per_aircraft = {}
    all_y_true, all_y_prob, all_y_pred = [], [], []

    for ac_id in aircraft_ids:
        test_mask = df["aircraft_id"] == ac_id
        test_df = df[test_mask]
        train_df = df[~test_mask]

        # Sadece arızalı uçakları değerlendir
        if only_failing and test_df[label_col].sum() == 0:
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df[label_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[label_col].values

        # Her fold'da yeni scaler
        scaler = RobustScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        # Model klonla ve eğit
        from sklearn.base import clone
        m = clone(model)

        # pos_weight güncelle
        pw = _get_pos_weight(y_train)
        if hasattr(m, "scale_pos_weight"):
            m.set_params(scale_pos_weight=pw)

        m.fit(X_train_sc, y_train)
        y_pred = m.predict(X_test_sc)
        y_prob = m.predict_proba(X_test_sc)[:, 1] if hasattr(m, "predict_proba") else None

        metrics = calculate_metrics(y_test, y_pred, y_prob)
        per_aircraft[ac_id] = metrics

        all_y_true.extend(y_test)
        if y_prob is not None:
            all_y_prob.extend(y_prob)
        all_y_pred.extend(y_pred)

    # Toplam metrikler
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob) if len(all_y_prob) > 0 else None

    agg_metrics = calculate_metrics(all_y_true, all_y_pred, all_y_prob)

    return {
        "per_aircraft": per_aircraft,
        "aggregate": agg_metrics,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "y_prob": all_y_prob,
    }


def compare_models(df, feature_cols, label_col="label"):
    """Tüm modelleri LOOCV ile karşılaştırır."""
    pw = _get_pos_weight(df[label_col].values)
    models = get_models(pos_weight=pw)
    results = {}
    comparison_rows = []

    for name, model in models.items():
        print(f"  Model: {name}...")
        res = leave_one_out_cv(df, feature_cols, label_col, model)
        results[name] = res
        row = {"Model": name}
        row.update(res["aggregate"])
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows).set_index("Model")
    return results, comparison_df


# ---------------------------------------------------------------------------
# Hiperparametre optimizasyonu ve eşik ayarlama
# ---------------------------------------------------------------------------

def tune_best_model(df, feature_cols, label_col="label"):
    """
    En iyi model (XGBoost) için grid search + eşik optimizasyonu.
    """
    pw = _get_pos_weight(df[label_col].values)

    param_grid = [
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.08},
    ]

    best_f1 = -1
    best_params = None
    best_result = None

    for params in param_grid:
        model = XGBClassifier(
            scale_pos_weight=pw, eval_metric="logloss",
            use_label_encoder=False, verbosity=0, random_state=42, n_jobs=-1,
            **params,
        )
        res = leave_one_out_cv(df, feature_cols, label_col, model)
        f1 = res["aggregate"]["F1"]
        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_result = res

    # Eşik optimizasyonu
    y_true = best_result["y_true"]
    y_prob = best_result["y_prob"]
    thresholds = np.arange(0.10, 0.91, 0.05)
    best_thresh = 0.5
    best_thresh_f1 = best_f1

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f1_t = f1_score(y_true, y_pred_t, zero_division=0)
        if f1_t > best_thresh_f1:
            best_thresh_f1 = f1_t
            best_thresh = t

    # Optimum eşik ile nihai tahminler
    y_pred_final = (y_prob >= best_thresh).astype(int)
    final_metrics = calculate_metrics(y_true, y_pred_final, y_prob)

    return {
        "best_params": best_params,
        "base_metrics": best_result["aggregate"],
        "best_threshold": best_thresh,
        "metrics": final_metrics,
        "y_true": y_true,
        "y_pred": y_pred_final,
        "y_prob": y_prob,
        "result": best_result,
    }


# ---------------------------------------------------------------------------
# Görselleştirme
# ---------------------------------------------------------------------------

def plot_roc_curves(results_dict, save_path):
    """Tüm modellerin ROC eğrilerini tek grafikte çizer."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    for (name, res), color in zip(results_dict.items(), colors):
        y_true = res["y_true"]
        y_prob = res["y_prob"]
        if y_prob is not None and len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Yanlış Pozitif Oranı (FPR)", fontsize=12)
    ax.set_ylabel("Doğru Pozitif Oranı (TPR)", fontsize=12)
    ax.set_title("ROC Eğrileri - Model Karşılaştırması", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Kaydedildi: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path, title="Karışıklık Matrisi (En İyi Model)"):
    """Karışıklık matrisini ısı haritası olarak çizer."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Sağlıklı", "Arızalı"],
        yticklabels=["Sağlıklı", "Arızalı"],
        ax=ax, linewidths=1,
    )
    ax.set_xlabel("Tahmin", fontsize=12)
    ax.set_ylabel("Gerçek", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Kaydedildi: {save_path}")


def plot_loocv_performance(per_aircraft_metrics, save_path):
    """Uçak bazlı LOOCV performansını çubuk grafik olarak gösterir."""
    rows = []
    for ac_id, metrics in per_aircraft_metrics.items():
        rows.append({
            "Uçak": ac_id,
            "F1": metrics["F1"],
            "Recall": metrics["Recall"],
            "Precision": metrics["Precision"],
        })
    perf_df = pd.DataFrame(rows).set_index("Uçak")

    fig, ax = plt.subplots(figsize=(10, 5))
    perf_df.plot(kind="bar", ax=ax, color=["#e74c3c", "#3498db", "#2ecc71"])
    ax.set_title("Uçak Bazlı LOOCV Performansı (Arızalı Uçaklar)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Kaydedildi: {save_path}")


def plot_feature_importance(model, feature_names, df, feature_cols, label_col, save_path):
    """En iyi modelin özellik önemlerini çubuk grafik olarak çizer."""
    from sklearn.base import clone

    # Tüm veri üzerinde eğit (görselleştirme için)
    scaler = RobustScaler()
    X = scaler.fit_transform(df[feature_cols].values)
    y = df[label_col].values
    m = clone(model)
    pw = _get_pos_weight(y)
    if hasattr(m, "scale_pos_weight"):
        m.set_params(scale_pos_weight=pw)
    m.fit(X, y)

    importances = m.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(feature_names)))
    ax.barh(
        [feature_names[i] for i in idx[::-1]],
        importances[idx[::-1]],
        color=colors,
    )
    ax.set_xlabel("Önem Skoru", fontsize=12)
    ax.set_title("XGBoost Özellik Önemleri", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Kaydedildi: {save_path}")
