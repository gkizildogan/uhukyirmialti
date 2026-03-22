"""
IDG Arıza Tahmini - Ana Orkestratör

Tüm pipeline'ı çalıştırır ve grafikleri images/ klasörüne kaydeder.
Kullanım: python src/main.py
"""

import os
import sys

# Proje kök dizinini path'e ekle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_generator import generate_fleet_data
from src.features import create_features, get_feature_names
from src.scoring import (
    score_features, plot_feature_scoring, plot_freq_boxplot,
    plot_feature_correlation, eliminate_correlated_features,
)
from src.models import (
    compare_models, tune_best_model,
    plot_roc_curves, plot_confusion_matrix,
    plot_loocv_performance, plot_feature_importance,
)
from xgboost import XGBClassifier


def main():
    images_dir = os.path.join(PROJECT_ROOT, "images")
    os.makedirs(images_dir, exist_ok=True)

    print("=" * 60)
    print("  IDG FREKANS ARIZA TAHMİNİ - ANALİZ PIPELINE")
    print("=" * 60)

    # 1. Veri üretimi
    print("\n[1/6] Sentetik filo verisi üretiliyor...")
    data = generate_fleet_data(n_aircraft=10, n_failing=3, seed=42)
    print(f"  Toplam kayıt: {len(data)}")
    print(f"  Uçak sayısı: {data['aircraft_id'].nunique()}")
    print(f"  Pozitif oran: {data['label'].mean():.2%}")

    # 2. Özellik mühendisliği
    print("\n[2/6] Simetrik özellikler türetiliyor...")
    df = create_features(data)
    feature_cols = get_feature_names()
    print(f"  Özellik sayısı: {len(feature_cols)}")

    # 3. EDA ve özellik skorlama
    print("\n[3/6] Özellik değerlendirme ve EDA...")
    plot_freq_boxplot(data, os.path.join(images_dir, "freq_boxplot.png"))

    score_df = score_features(df, feature_cols)
    print("\n  Özellik Skorları:")
    print(score_df.to_string())
    plot_feature_scoring(score_df, os.path.join(images_dir, "feature_scoring.png"))

    plot_feature_correlation(df, feature_cols, os.path.join(images_dir, "feature_correlation.png"))
    feature_cols, dropped_features = eliminate_correlated_features(df, feature_cols, score_df)
    print(f"\n  Elenen özellikler (|r| > 0.90): {dropped_features}")
    print(f"  Kalan özellikler ({len(feature_cols)}): {feature_cols}")

    # 4. Model karşılaştırma
    print("\n[4/6] Model karşılaştırması (LOOCV)...")
    results, comparison_df = compare_models(df, feature_cols)
    print("\n  Model Karşılaştırma Tablosu:")
    display_cols = ["F1", "Precision", "Recall", "Accuracy", "ROC AUC", "PR AUC"]
    available_cols = [c for c in display_cols if c in comparison_df.columns]
    print(comparison_df[available_cols].to_string(float_format="{:.3f}".format))
    plot_roc_curves(results, os.path.join(images_dir, "roc_curves.png"))

    # 5. Hiperparametre optimizasyonu
    print("\n[5/6] Hiperparametre optimizasyonu ve eşik ayarlama...")
    tuned = tune_best_model(df, feature_cols)
    print(f"  En iyi parametreler: {tuned['best_params']}")
    print(f"  Optimum eşik: {tuned['best_threshold']:.2f}")
    print(f"  F1 (eşik sonrası): {tuned['metrics']['F1']:.3f}")
    print(f"  Recall (eşik sonrası): {tuned['metrics']['Recall']:.3f}")
    print(f"  Precision (eşik sonrası): {tuned['metrics']['Precision']:.3f}")

    plot_confusion_matrix(
        tuned["y_true"], tuned["y_pred"],
        os.path.join(images_dir, "confusion_matrix.png"),
    )

    # Özellik önemleri
    best_model = XGBClassifier(
        **tuned["best_params"],
        eval_metric="logloss", use_label_encoder=False,
        verbosity=0, random_state=42,
    )
    plot_feature_importance(
        best_model, feature_cols, df, feature_cols, "label",
        os.path.join(images_dir, "feature_importance.png"),
    )

    # 6. LOOCV uçak bazlı performans
    print("\n[6/6] Uçak bazlı LOOCV sonuçları...")
    # En iyi modelin per-aircraft sonuçlarını kullan
    xgb_per_aircraft = results["XGBoost"]["per_aircraft"]
    plot_loocv_performance(
        xgb_per_aircraft,
        os.path.join(images_dir, "loocv_performance.png"),
    )

    print("\n" + "=" * 60)
    print("  TAMAMLANDI! Tüm grafikler images/ klasörüne kaydedildi.")
    print("=" * 60)
    print(f"\n  Üretilen dosyalar:")
    for f in sorted(os.listdir(images_dir)):
        print(f"    - images/{f}")


if __name__ == "__main__":
    main()
