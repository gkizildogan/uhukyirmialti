"""
Özellik değerlendirme ve keşifsel veri analizi (EDA) modülü.

Her özellik için bileşik bir skor hesaplar:
korelasyon, Mann-Whitney U, Kolmogorov-Smirnov, ROC AUC, Lojistik Regresyon AUC.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def score_single_feature(X_feat, y):
    """Tek bir özellik için 5 farklı skor hesaplar."""
    pos = X_feat[y == 1]
    neg = X_feat[y == 0]

    # 1. Point-biserial korelasyon
    corr, _ = stats.pointbiserialr(y, X_feat)
    corr_score = abs(corr)

    # 2. Mann-Whitney U testi
    try:
        _, mw_pval = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        mw_score = 1.0 - mw_pval
    except ValueError:
        mw_score = 0.0

    # 3. Kolmogorov-Smirnov testi
    ks_stat, _ = stats.ks_2samp(pos, neg)

    # 4. ROC AUC (özellik tek başına sınıflandırıcı olarak)
    try:
        raw_auc = roc_auc_score(y, X_feat)
        roc_score = max(raw_auc, 1.0 - raw_auc)  # Yön bağımsız
    except ValueError:
        roc_score = 0.5

    # 5. RobustScaler + Lojistik Regresyon AUC
    try:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_feat.reshape(-1, 1))
        lr = LogisticRegression(max_iter=1000, solver="lbfgs")
        lr.fit(X_scaled, y)
        y_prob = lr.predict_proba(X_scaled)[:, 1]
        lr_auc = roc_auc_score(y, y_prob)
    except Exception:
        lr_auc = 0.5

    return {
        "Korelasyon": corr_score,
        "Mann-Whitney": mw_score,
        "KS İstatistiği": ks_stat,
        "ROC AUC": roc_score,
        "LR AUC": lr_auc,
    }


def score_features(df, feature_cols, label_col="label"):
    """
    Tüm özellikler için bileşik skor tablosu oluşturur.

    Dönüş
    ------
    pd.DataFrame
        Satırlar: özellikler, Sütunlar: skorlar + Bileşik Skor
    """
    y = df[label_col].values
    results = []
    for col in feature_cols:
        X_feat = df[col].values
        scores = score_single_feature(X_feat, y)
        scores["Özellik"] = col
        results.append(scores)

    score_df = pd.DataFrame(results).set_index("Özellik")

    # Bileşik skor: tüm skorların ortalaması
    score_df["Bileşik Skor"] = score_df.mean(axis=1)
    score_df = score_df.sort_values("Bileşik Skor", ascending=False)

    return score_df


def plot_feature_scoring(score_df, save_path):
    """Özellik skorlarını ısı haritası olarak görselleştirir."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        score_df,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Özellik Değerlendirme Skorları", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Kaydedildi: {save_path}")


def plot_freq_boxplot(df, save_path):
    """Normal vs arıza öncesi IDG frekanslarını kutu grafiği ile gösterir."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    label_map = {0: "Normal", 1: "Arıza Öncesi"}
    df_plot = df.copy()
    df_plot["Durum"] = df_plot["label"].map(label_map)

    for i, (col, title) in enumerate([
        ("freq_left", "Sol IDG Frekansı (Hz)"),
        ("freq_right", "Sağ IDG Frekansı (Hz)"),
    ]):
        sns.boxplot(data=df_plot, x="Durum", y=col, ax=axes[i],
                    palette=["#2ecc71", "#e74c3c"])
        axes[i].set_title(title, fontsize=13, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Frekans (Hz)")
        axes[i].axhline(y=400.0, color="gray", linestyle="--", alpha=0.5, label="400 Hz")
        axes[i].legend(loc="upper right", fontsize=9)

    plt.suptitle("IDG Frekans Dağılımları: Normal ve Arıza Öncesi Dönem",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Kaydedildi: {save_path}")
