'''Synthetic ride-hailing dataset generator.

Realism over randomness. Every metric is earned from upstream dependencies.

Calibration:
- Zone weights
- Weekday variance: Fri/Sat +18-25%, Mon -15%
- Seasonality: Summer evenings +12%, winter mornings +8%
- Wait-time driver: hour-load + zone-load (sigmoid pressure)
- Surge driver: hour-load + zone-load + 0.02 * wait_time (earned correlation)
- Cancellations: base 5%, + sigmoid(wait_time) + sigmoid(surge) → 2-35% range
- Distance by zone: Old Town gamma(1.6, 1.0), Gldani gamma(2.9, 1.9)
- Vehicle mix: Economy 72% → Comfort +10% Vake/+6% evening, XL +6% weekend

Outputs single rides table (day, hour, zone, distance, wait, surge, fare, completed).
'''

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class ZoneParams:
    demand_weight: float
    distance_gamma_shape: float
    distance_gamma_scale: float

ZONES: Dict[str, ZoneParams] = {
    # dominant zones
    'Vake': ZoneParams(demand_weight=0.28, distance_gamma_shape=2.2, distance_gamma_scale=1.4),
    'Saburtalo': ZoneParams(demand_weight=0.24, distance_gamma_shape=2.3, distance_gamma_scale=1.5),
    # short-trip zones
    'Old Town': ZoneParams(demand_weight=0.18, distance_gamma_shape=1.6, distance_gamma_scale=1.0),
    'Shardeni': ZoneParams(demand_weight=0.12, distance_gamma_shape=1.7, distance_gamma_scale=1.1),
    # trailing zones
    'Nadzaladevi': ZoneParams(demand_weight=0.10, distance_gamma_shape=2.5, distance_gamma_scale=1.7),
    'Gldani': ZoneParams(demand_weight=0.08, distance_gamma_shape=2.9, distance_gamma_scale=1.9),
}

def generate_synthetic_ride_data(n_rides: int = 450_000, seed: int = 42) -> pd.DataFrame:
    # generating a realistic synthetic ride dataset.

    rng = np.random.default_rng(seed)

    zones = np.array(list(ZONES.keys()))
    zone_weights = np.array([ZONES[z].demand_weight for z in zones], dtype=float)
    zone_probs = zone_weights / zone_weights.sum()

    date_index = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    date_probs = _get_date_probs(date_index)

    picked_dates = rng.choice(date_index.to_numpy(), size=n_rides, p=date_probs)
    picked_months = pd.DatetimeIndex(picked_dates).month.to_numpy()

    # hour distribution changes slightly by season
    season_bucket = _season_bucket(picked_months)
    hours = np.empty(n_rides, dtype=int)
    for bucket in ('winter', 'summer', 'shoulder'):
        mask = season_bucket == bucket
        if not np.any(mask):
            continue
        hours[mask] = rng.choice(24, size=int(mask.sum()), p=_get_hour_weights(bucket))

    minutes = rng.integers(0, 60, size=n_rides)
    seconds = rng.integers(0, 60, size=n_rides)

    timestamps = (
        pd.to_datetime(picked_dates)
        + pd.to_timedelta(hours, unit='h')
        + pd.to_timedelta(minutes, unit='m')
        + pd.to_timedelta(seconds, unit='s')
    )

    pickup_zone = rng.choice(zones, size=n_rides, p=zone_probs)

    # dropoff: 25% same-zone, otherwise zone-weighted
    same_zone = rng.random(n_rides) < 0.25
    dropoff_zone = rng.choice(zones, size=n_rides, p=zone_probs)
    dropoff_zone[same_zone] = pickup_zone[same_zone]

    hour_load = _hour_load_score(hours)
    zone_load = _zone_load_score(pickup_zone, zones, zone_weights)

    wait_time_minutes = _generate_wait_time(rng, hour_load, zone_load)
    surge_multiplier = _generate_surge(rng, hour_load, zone_load, wait_time_minutes)

    vehicle_type = _generate_vehicle_type(rng, pickup_zone, hours, pd.DatetimeIndex(timestamps).day_name().to_numpy())

    distance_km = _generate_distance_by_zone(rng, pickup_zone)
    fare = _calculate_fare(distance_km, surge_multiplier, vehicle_type)

    completed = _generate_completion_flag(rng, wait_time_minutes, surge_multiplier)

    df = pd.DataFrame(
        {
            'ride_id': np.arange(1, n_rides + 1, dtype=np.int64),
            'timestamp': timestamps,
            'pickup_zone': pickup_zone,
            'dropoff_zone': dropoff_zone,
            'vehicle_type': vehicle_type,
            'distance_km': distance_km,
            'wait_time_minutes': wait_time_minutes,
            'surge_multiplier': surge_multiplier,
            'completed': completed,
            'fare': fare,
        }
    )

    dt = pd.to_datetime(df['timestamp'])
    df['hour'] = dt.dt.hour
    df['day_of_week'] = dt.dt.day_name()
    df['date'] = dt.dt.date
    df['month'] = dt.dt.month

    return df.sort_values('timestamp').reset_index(drop=True)

def _get_date_probs(date_index: pd.DatetimeIndex) -> np.ndarray:
    # weekday weights - Mon lower; Fri/Sat higher
    weekday_weights = np.array([0.85, 0.95, 1.00, 1.05, 1.18, 1.25, 1.10], dtype=float)

    # mild seasonality by month (summer a bit higher overall)
    month_weights = np.array(
        [
            0.92,  # Jan
            0.92,  # Feb
            0.98,  # Mar
            1.00,
            1.02,
            1.06,
            1.08,
            1.08,
            1.02,
            1.00,
            0.98,
            0.95,
        ],
        dtype=float,
    )

    weekdays = date_index.weekday.to_numpy()  # 0=Mon
    months = date_index.month.to_numpy()

    weights = weekday_weights[weekdays] * month_weights[months - 1]
    return (weights / weights.sum()).astype(float)

def _season_bucket(months: np.ndarray) -> np.ndarray:
    out = np.empty(months.shape[0], dtype=object)
    winter = np.isin(months, [12, 1, 2])
    summer = np.isin(months, [6, 7, 8])
    out[winter] = 'winter'
    out[summer] = 'summer'
    out[~(winter | summer)] = 'shoulder'
    return out

def _get_hour_weights(season: str) -> np.ndarray:
    # base: typical ride-hailing pattern (morning + evening peaks)
    base = np.array(
        [
            0.020,
            0.015,
            0.012,
            0.010,
            0.010,
            0.014,
            0.024,
            0.040,
            0.050,
            0.040,
            0.032,
            0.030,
            0.032,
            0.032,
            0.030,
            0.032,
            0.038,
            0.050,
            0.070,
            0.082,
            0.075,
            0.060,
            0.045,
            0.030,
        ],
        dtype=float,
    )

    weights = base.copy()
    if season == 'summer':
        # summer evenings a bit higher
        weights[18:23] *= 1.12
        weights[6:10] *= 0.97
    elif season == 'winter':
        # winter mornings a bit higher
        weights[7:11] *= 1.08
        weights[18:23] *= 0.97

    return weights / weights.sum()


def _hour_load_score(hours: np.ndarray) -> np.ndarray:
    # base weights to define relative load score (0..1)
    base = _get_hour_weights('shoulder')
    score = base[hours] / base.max()
    return score.astype(float)

def _zone_load_score(pickup_zone: np.ndarray, zones: np.ndarray, zone_weights: np.ndarray) -> np.ndarray:
    weight_map = {z: w for z, w in zip(zones, zone_weights)}
    z_w = np.vectorize(weight_map.get, otypes=[float])(pickup_zone)
    score = z_w / zone_weights.max()
    return score.astype(float)

def _generate_wait_time(rng: np.random.Generator, hour_load: np.ndarray, zone_load: np.ndarray) -> np.ndarray:
    # wait time with realistic skew
    base = rng.lognormal(mean=1.35, sigma=0.35, size=hour_load.shape[0])  # ~3-6 min typical
    pressure = 9.0 * (hour_load**1.35) + 6.5 * (zone_load**1.25)
    noise = rng.normal(0, 0.8, size=hour_load.shape[0])
    wait = base + pressure + noise
    return np.clip(wait, 1.0, 28.0).round(2)

def _generate_surge(
    rng: np.random.Generator,
    hour_load: np.ndarray,
    zone_load: np.ndarray,
    wait_time_minutes: np.ndarray,
) -> np.ndarray:
    # surge is demand-driven (hour + zone), with a small earned link to wait-time
    raw = 1.0 + 0.95 * (hour_load**1.35) + 0.70 * (zone_load**1.15) + 0.02 * np.clip(wait_time_minutes - 4.0, 0, None)
    raw += rng.normal(0, 0.08, size=hour_load.shape[0])
    return np.clip(raw, 1.0, 3.0).round(2)

def _generate_completion_flag(
    rng: np.random.Generator,
    wait_time_minutes: np.ndarray,
    surge_multiplier: np.ndarray,
) -> np.ndarray:
    # cancellation probability rises with long waits and high surge.
    base_cancel = 0.05

    wait_sig = 1.0 / (1.0 + np.exp(-(wait_time_minutes - 6.0) / 2.0))  # 0..1
    surge_sig = np.clip((surge_multiplier - 1.0) / 2.0, 0.0, 1.0)  # 0..1

    cancel_prob = base_cancel + 0.20 * wait_sig + 0.10 * surge_sig
    cancel_prob = np.clip(cancel_prob, 0.02, 0.35)

    return rng.random(wait_time_minutes.shape[0]) > cancel_prob

def _generate_distance_by_zone(rng: np.random.Generator, pickup_zone: np.ndarray) -> np.ndarray:
    distance = np.empty(pickup_zone.shape[0], dtype=float)

    for zone_name, params in ZONES.items():
        mask = pickup_zone == zone_name
        if not np.any(mask):
            continue
        d = rng.gamma(shape=params.distance_gamma_shape, scale=params.distance_gamma_scale, size=int(mask.sum()))
        # Add a tiny bit of noise, clamp to sane bounds
        d = d + rng.normal(0, 0.25, size=d.shape[0])
        distance[mask] = np.clip(d, 0.6, 22.0)

    return distance.round(2)

def _generate_vehicle_type(
    rng: np.random.Generator,
    pickup_zone: np.ndarray,
    hours: np.ndarray,
    day_names: np.ndarray,
) -> np.ndarray:
    # Base mix
    econ = np.full(pickup_zone.shape[0], 0.72, dtype=float)
    comfort = np.full(pickup_zone.shape[0], 0.20, dtype=float)
    xl = np.full(pickup_zone.shape[0], 0.08, dtype=float)

    evening = (hours >= 18) & (hours <= 23)
    weekend = np.isin(day_names, ['Friday', 'Saturday', 'Sunday'])

    in_vake = pickup_zone == 'Vake'
    in_sab = pickup_zone == 'Saburtalo'

    comfort_bonus = (0.10 * in_vake.astype(float)) + (0.06 * in_sab.astype(float)) + (0.05 * evening.astype(float))
    xl_bonus = 0.06 * weekend.astype(float)

    comfort += comfort_bonus
    xl += xl_bonus
    econ -= (comfort_bonus + xl_bonus)

    # keep probabilities sane
    econ = np.clip(econ, 0.45, 0.90)
    comfort = np.clip(comfort, 0.05, 0.45)
    xl = np.clip(xl, 0.03, 0.25)

    total = econ + comfort + xl
    econ /= total
    comfort /= total

    u = rng.random(pickup_zone.shape[0])
    vehicle = np.where(u < econ, 'Economy', np.where(u < (econ + comfort), 'Comfort', 'XL'))
    return vehicle

def _calculate_fare(distance_km: np.ndarray, surge_multiplier: np.ndarray, vehicle_type: np.ndarray) -> np.ndarray:
    base = np.full(distance_km.shape[0], 1.5, dtype=float)
    rate = np.full(distance_km.shape[0], 0.6, dtype=float)

    is_comfort = vehicle_type == 'Comfort'
    is_xl = vehicle_type == 'XL'

    base[is_comfort] = 2.0
    rate[is_comfort] = 0.8

    base[is_xl] = 2.5
    rate[is_xl] = 1.0

    fare = (base + distance_km * rate) * surge_multiplier
    return np.clip(fare, 1.2, None).round(2)


if __name__ == '__main__':
    print('Generating synthetic ride data...')
    df = generate_synthetic_ride_data(450_000, seed=42)
    print(f'Generated {len(df):,} rides')
    print(f'Date range: {df["timestamp"].min()} to {df["timestamp"].max()}')
    print(f'Zones: {sorted(df["pickup_zone"].unique())}')
    print(f'Vehicle types: {sorted(df["vehicle_type"].unique())}')
    print(f'Completion rate: {df["completed"].mean():.2%}')
    print(f'Average wait: {df["wait_time_minutes"].mean():.2f} min')
    print(f'Average surge: {df["surge_multiplier"].mean():.2f}x')
    print(f'Average fare: {df["fare"].mean():.2f} GEL')
