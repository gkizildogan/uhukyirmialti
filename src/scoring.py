"""
Özellik değerlendirme ve keşifsel veri analizi (EDA) modülü.

Her özellik için bileşik bir skor hesaplar:
korelasyon, Kolmogorov-Smirnov, ROC AUC ve uçak bazlı LOOCV kararlılık skoru.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score


def score_single_feature(X_feat, y, aircraft_ids):
    """
    Tek bir özellik için sızıntısız 4 farklı skor hesaplar.

    Kararlılık skoru (Stability), purge-split yerine uçak bazlı LOOCV ile
    hesaplanır: her uçak ayrı bir fold olarak değerlendirilir.
    """
    X_feat = np.asarray(X_feat)
    y = np.asarray(y)
    aircraft_ids = np.asarray(aircraft_ids)

    pos = X_feat[y == 1]
    neg = X_feat[y == 0]

    # 1. Point-biserial korelasyon
    try:
        corr, _ = stats.pointbiserialr(y, X_feat)
        corr_score = abs(corr)
    except Exception:
        corr_score = 0.0

    # 2. Kolmogorov-Smirnov testi
    try:
        ks_stat, _ = stats.ks_2samp(pos, neg)
    except Exception:
        ks_stat = 0.0

    # 3. ROC AUC (global, yön bağımsız)
    try:
        raw_auc = roc_auc_score(y, X_feat)
        roc_score = max(raw_auc, 1.0 - raw_auc)
    except Exception:
        roc_score = 0.5

    # 4. Kararlılık: uçak bazlı LOOCV AUC (her uçak ayrı fold)
    aucs = []
    for ac_id in np.unique(aircraft_ids):
        mask = aircraft_ids == ac_id
        y_fold = y[mask]
        X_fold = X_feat[mask]
        try:
            auc = roc_auc_score(y_fold, X_fold)
            aucs.append(max(auc, 1.0 - auc))
        except Exception:
            continue  # Pozitif örnek içermeyen uçaklar atlanır

    if len(aucs) > 0:
        stability_mean = np.mean(aucs)
        stability_std = np.std(aucs)
        stability_score = stability_mean * (1 - stability_std)
    else:
        stability_score = 0.5

    return {
        "ROC_AUC": roc_score,
        "KS": ks_stat,
        "Korelasyon": corr_score,
        "Stabilite (AUC)": stability_score,
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
    aircraft_ids = df["aircraft_id"].values
    results = []
    for col in feature_cols:
        X_feat = df[col].values
        scores = score_single_feature(X_feat, y, aircraft_ids)
        scores["Özellik"] = col
        results.append(scores)

    score_df = pd.DataFrame(results).set_index("Özellik")

    # Bileşik skor: tüm skorların ortalaması
    score_df["Bileşik Skor"] = score_df.mean(axis=1)
    score_df = score_df.sort_values("Bileşik Skor", ascending=False)

    return score_df


def plot_feature_correlation(df, feature_cols, save_path):
    """Özellikler arası Pearson korelasyon matrisini ısı haritası olarak çizer."""
    corr = df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Özellik Korelasyon Matrisi", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Kaydedildi: {save_path}")


def eliminate_correlated_features(df, feature_cols, score_df, threshold=0.90):
    """
    Yüksek korelasyonlu özellik çiftlerini eler.

    Mutlak Pearson korelasyonu `threshold` değerini aşan her çift için,
    hedef değişken ile daha düşük nokta-biserial korelasyona sahip olan
    özellik elenmiştir. Kalan özellikler ve elenen özellikler döndürülür.

    Parametreler
    ----------
    df : pd.DataFrame
        Özellik sütunlarını içeren veri çerçevesi.
    feature_cols : list
        Değerlendirilecek özellik sütun adları.
    score_df : pd.DataFrame
        score_features() çıktısı; "Korelasyon" sütununu içermelidir.
    threshold : float
        Eleme için korelasyon eşiği (varsayılan: 0.90).

    Dönüş
    ------
    retained : list
        Korunan özellik adları listesi.
    dropped : list
        Elenen özellik adları listesi (sıralı).
    """
    corr_matrix = df[feature_cols].corr().abs()
    target_corr = score_df["Korelasyon"]

    to_drop = set()
    features = list(feature_cols)

    for i in range(len(features)):
        if features[i] in to_drop:
            continue
        for j in range(i + 1, len(features)):
            if features[j] in to_drop:
                continue
            corr_val = corr_matrix.loc[features[i], features[j]]
            if corr_val > threshold:
                corr_i = target_corr.get(features[i], 0.0)
                corr_j = target_corr.get(features[j], 0.0)
                if corr_i < corr_j:
                    print(
                        f"  Eleniyor: {features[i]} "
                        f"(hedef korr={corr_i:.3f}) <- "
                        f"{features[j]} (hedef korr={corr_j:.3f}), "
                        f"özellik korr={corr_val:.3f})"
                    )
                    to_drop.add(features[i])
                    break
                else:
                    print(
                        f"  Eleniyor: {features[j]} "
                        f"(hedef korr={corr_j:.3f}) <- "
                        f"{features[i]} (hedef korr={corr_i:.3f}), "
                        f"özellik korr={corr_val:.3f})"
                    )
                    to_drop.add(features[j])

    retained = [f for f in features if f not in to_drop]
    return retained, sorted(to_drop)


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

    label_map = {0: "Sağlıklı", 1: "Arızalı"}
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

    plt.suptitle("IDG Frekans Dağılımları: Sağlıklı ve Arızalı Dönem",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Kaydedildi: {save_path}")
