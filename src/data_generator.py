"""
Sentetik IDG (Integrated Drive Generator) frekans verisi üretici.

IDG nominal frekansı 400 Hz. Normal operasyonda küçük gürültü,
arıza öncesi dönemde varyans artışı ve ortalama kayması simüle edilir.
Veri dakika bazında (1 ölçüm/dakika) üretilir.
"""

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 24 * 60
DEGRADATION_DAYS = 30  # Bozulma başlangıcı: arızadan kaç gün önce
LABEL_DAYS = 14        # Pozitif etiket penceresi: arızadan kaç gün önce


def _generate_single_aircraft(aircraft_id, is_failing, n_minutes, fail_side, seed):
    """Tek bir uçak için dakika bazlı IDG frekans verisi üretir."""
    rng = np.random.RandomState(seed)

    dates = pd.date_range("2025-01-01", periods=n_minutes, freq="min")

    # Normal operasyon: 400 Hz civarı, küçük gürültü
    base_noise_std = 0.30
    freq_left = 400.0 + rng.normal(0, base_noise_std, n_minutes)
    freq_right = 400.0 + rng.normal(0, base_noise_std, n_minutes)

    labels = np.zeros(n_minutes, dtype=int)

    if is_failing:
        degradation_start = max(0, n_minutes - DEGRADATION_DAYS * MINUTES_PER_DAY)
        label_start = max(0, n_minutes - LABEL_DAYS * MINUTES_PER_DAY)

        # Vektörize bozulma: son ~30 günde kademeli varyans artışı ve ortalama kayması
        deg_length = n_minutes - degradation_start
        progress = np.arange(deg_length) / deg_length  # [0, 1)

        extra_std = 1.2 * progress          # 0.0 -> 1.2 Hz varyans artışı
        mean_drift = 1.8 * progress ** 1.5  # 0.0 -> 1.8 Hz ortalama kayması

        degradation = mean_drift + extra_std * rng.standard_normal(deg_length)

        if fail_side == "left":
            freq_left[degradation_start:] += degradation
        else:
            freq_right[degradation_start:] += degradation

        labels[label_start:] = 1

    df = pd.DataFrame({
        "aircraft_id": aircraft_id,
        "date": dates,
        "freq_left": freq_left,
        "freq_right": freq_right,
        "label": labels,
    })
    return df


def generate_fleet_data(n_aircraft=10, n_failing=3, seed=42):
    """
    Filo düzeyinde sentetik IDG frekans verisi üretir (dakika bazlı).

    Parametreler
    ----------
    n_aircraft : int
        Toplam uçak sayısı.
    n_failing : int
        Arıza geliştirecek uçak sayısı.
    seed : int
        Rastgelelik tohumu.

    Dönüş
    ------
    pd.DataFrame
        aircraft_id, date, freq_left, freq_right, label sütunları.
        Her satır bir dakikalık IDG frekans ölçümünü temsil eder.
    """
    rng = np.random.RandomState(seed)

    failing_indices = set(rng.choice(n_aircraft, n_failing, replace=False))

    dfs = []
    for i in range(n_aircraft):
        is_failing = i in failing_indices
        n_days = rng.randint(90, 181)  # 3-6 ay arası izleme süresi
        n_minutes = n_days * MINUTES_PER_DAY
        fail_side = rng.choice(["left", "right"]) if is_failing else None
        aircraft_seed = seed + i * 100

        df = _generate_single_aircraft(
            aircraft_id=f"AC-{i+1:03d}",
            is_failing=is_failing,
            n_minutes=n_minutes,
            fail_side=fail_side,
            seed=aircraft_seed,
        )
        dfs.append(df)

    fleet_df = pd.concat(dfs, ignore_index=True)
    return fleet_df


if __name__ == "__main__":
    data = generate_fleet_data()
    print(f"Toplam kayıt: {len(data):,}")
    print(f"Uçak sayısı: {data['aircraft_id'].nunique()}")
    print(f"Pozitif oran: {data['label'].mean():.2%}")
    print(data.groupby("aircraft_id")["label"].mean())
