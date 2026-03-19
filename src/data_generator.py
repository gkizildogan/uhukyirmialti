"""
Sentetik IDG (Integrated Drive Generator) frekans verisi üretici.

IDG nominal frekansı 400 Hz. Normal operasyonda küçük gürültü,
arıza öncesi dönemde varyans artışı ve ortalama kayması simüle edilir.
"""

import numpy as np
import pandas as pd


def _generate_single_aircraft(aircraft_id, is_failing, n_days, fail_side, seed):
    """Tek bir uçak için IDG frekans verisi üretir."""
    rng = np.random.RandomState(seed)

    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")

    # Normal operasyon: 400 Hz civarı, küçük gürültü
    base_noise_std = 0.30
    freq_left = 400.0 + rng.normal(0, base_noise_std, n_days)
    freq_right = 400.0 + rng.normal(0, base_noise_std, n_days)

    labels = np.zeros(n_days, dtype=int)

    if is_failing:
        # Arıza öncesi dönem: son ~30 günde kademeli bozulma
        degradation_start = max(0, n_days - 30)
        label_start = max(0, n_days - 14)  # Son 14 gün label=1

        for i in range(degradation_start, n_days):
            progress = (i - degradation_start) / (n_days - degradation_start)
            # Varyans artışı (0.3 -> 1.5 Hz) ve ortalama kayması (0 -> 1.8 Hz)
            extra_std = 1.2 * progress
            mean_drift = 1.8 * progress ** 1.5

            if fail_side == "left":
                freq_left[i] += rng.normal(mean_drift, extra_std)
            else:
                freq_right[i] += rng.normal(mean_drift, extra_std)

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
    Filo düzeyinde sentetik IDG frekans verisi üretir.

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
    """
    rng = np.random.RandomState(seed)

    # Hangi uçaklar arızalı
    failing_indices = set(rng.choice(n_aircraft, n_failing, replace=False))

    dfs = []
    for i in range(n_aircraft):
        is_failing = i in failing_indices
        n_days = rng.randint(90, 181)  # 3-6 ay arası veri
        fail_side = rng.choice(["left", "right"]) if is_failing else None
        aircraft_seed = seed + i * 100

        df = _generate_single_aircraft(
            aircraft_id=f"AC-{i+1:03d}",
            is_failing=is_failing,
            n_days=n_days,
            fail_side=fail_side,
            seed=aircraft_seed,
        )
        dfs.append(df)

    fleet_df = pd.concat(dfs, ignore_index=True)
    return fleet_df


if __name__ == "__main__":
    data = generate_fleet_data()
    print(f"Toplam kayıt: {len(data)}")
    print(f"Uçak sayısı: {data['aircraft_id'].nunique()}")
    print(f"Pozitif oran: {data['label'].mean():.2%}")
    print(data.groupby("aircraft_id")["label"].mean())
