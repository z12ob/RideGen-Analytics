# Data processing for the RideGen Analytics.
# Loads rides data and exports BI-ready aggregations as CSV.

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RequiredColumns:
    columns: List[str]

REQUIRED = RequiredColumns(
    columns=[
        'ride_id',
        'timestamp',
        'pickup_zone',
        'dropoff_zone',
        'vehicle_type',
        'distance_km',
        'fare',
        'wait_time_minutes',
        'completed',
        'surge_multiplier',
    ]
)

class RideDataProcessor:
    # process and aggregate ride-hailing data for analysis.

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df: pd.DataFrame | None = None

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logger.info('Initialized processor for %s', csv_path)
        
    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.csv_path, parse_dates=['timestamp'])

            missing = [c for c in REQUIRED.columns if c not in df.columns]
            if missing:
                raise ValueError(f'Missing required columns: {missing}')

            df['completed'] = df['completed'].astype(bool)
            df['distance_km'] = pd.to_numeric(df['distance_km'], errors='coerce')
            df['fare'] = pd.to_numeric(df['fare'], errors='coerce')
            df['wait_time_minutes'] = pd.to_numeric(df['wait_time_minutes'], errors='coerce')
            df['surge_multiplier'] = pd.to_numeric(df['surge_multiplier'], errors='coerce')

            # Derived time fields (if not present)
            dt = pd.to_datetime(df['timestamp'], errors='coerce')
            df['date'] = dt.dt.date
            df['hour'] = dt.dt.hour
            df['day_of_week'] = dt.dt.day_name()
            df['month'] = dt.dt.month

            self.df = df
            logger.info('Loaded %s rows', f'{len(df):,}')
            return df
        except FileNotFoundError:
            logger.error('File not found: %s', self.csv_path)
            raise

    def data_quality_check(self) -> Dict[str, object]:
        if self.df is None:
            raise ValueError('Data not loaded. Call load_data() first.')

        df = self.df
        report: Dict[str, object] = {
            'total_records': int(len(df)),
            'duplicates': int(df.duplicated(subset=['ride_id']).sum()),
            'missing_values': df.isna().sum().to_dict(),
            'date_range': (df['timestamp'].min(), df['timestamp'].max()),
            'completion_rate': float(df['completed'].mean()),
        }

        logger.info('Quality check: %s records', f"{report['total_records']:,}")
        if report['duplicates']:
            logger.warning('Found %s duplicate ride_ids', report['duplicates'])

        return report

    def aggregate_by_hour(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError('Data not loaded. Call load_data() first.')

        df = self.df
        g = df.groupby(['date', 'hour'], dropna=False)
        out = g.agg(
            total_rides=('ride_id', 'count'),
            completed_rides=('completed', 'sum'),
            avg_fare=('fare', 'mean'),
            total_revenue=('fare', 'sum'),
            avg_distance=('distance_km', 'mean'),
            avg_wait_time=('wait_time_minutes', 'mean'),
            avg_surge=('surge_multiplier', 'mean'),
        ).reset_index()

        out['completion_rate'] = (out['completed_rides'] / out['total_rides']).replace([np.inf, -np.inf], np.nan)
        # helpful for BI
        dt = pd.to_datetime(out['date'])
        out['day_of_week'] = dt.dt.day_name()
        out['month'] = dt.dt.month
        return out.sort_values(['date', 'hour']).reset_index(drop=True)

    def aggregate_by_zone(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError('Data not loaded. Call load_data() first.')
        df = self.df

        g = df.groupby(['pickup_zone'], dropna=False)
        out = g.agg(
            total_rides=('ride_id', 'count'),
            completed_rides=('completed', 'sum'),
            avg_fare=('fare', 'mean'),
            total_revenue=('fare', 'sum'),
            avg_distance=('distance_km', 'mean'),
            avg_wait_time=('wait_time_minutes', 'mean'),
            avg_surge=('surge_multiplier', 'mean'),
        ).reset_index()

        out['completion_rate'] = (out['completed_rides'] / out['total_rides']).replace([np.inf, -np.inf], np.nan)
        out['ride_share'] = out['total_rides'] / out['total_rides'].sum()
        return out.sort_values("total_rides", ascending=False).reset_index(drop=True)

    def peak_hours_analysis(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError('Data not loaded. Call load_data() first.')

        df = self.df
        out = (
            df.groupby(['day_of_week', 'hour'], dropna=False)
            .agg(ride_count=('ride_id', 'count'), avg_wait_time=('wait_time_minutes', 'mean'), avg_surge=('surge_multiplier', 'mean'))
            .reset_index()
        )

        out['share_of_week'] = out['ride_count'] / out['ride_count'].sum()

        # calendar order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        out['day_of_week'] = pd.Categorical(out['day_of_week'], categories=day_order, ordered=True)
        return out.sort_values(['day_of_week', 'hour']).reset_index(drop=True)

    def vehicle_type_analysis(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError('Data not loaded. Call load_data() first.')
        df = self.df

        out = (
            df.groupby(['pickup_zone', 'vehicle_type'], dropna=False)
            .agg(
                total_rides=('ride_id', 'count'),
                completed_rides=('completed', 'sum'),
                avg_fare=('fare', 'mean'),
                total_revenue=('fare', 'sum'),
                avg_wait_time=('wait_time_minutes', 'mean'),
                avg_surge=('surge_multiplier', 'mean'),
            )
            .reset_index()
        )

        out['completion_rate'] = (out['completed_rides'] / out['total_rides']).replace([np.inf, -np.inf], np.nan)
        out['zone_vehicle_share'] = out.groupby('pickup_zone')['total_rides'].transform(lambda s: s / s.sum())

        return out.sort_values(['pickup_zone', 'total_rides'], ascending=[True, False]).reset_index(drop=True)

    def surge_analysis(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError('Data not loaded. Call load_data() first.')
        df = self.df

        out = (
            df.groupby(['pickup_zone', 'hour'], dropna=False)
            .agg(
                ride_count=('ride_id', 'count'),
                avg_wait_time=('wait_time_minutes', 'mean'),
                avg_surge=('surge_multiplier', 'mean'),
                cancel_rate=('completed', lambda s: 1.0 - float(s.mean())),
            )
            .reset_index()
        )

        return out.sort_values(['pickup_zone', 'hour']).reset_index(drop=True)

    def process_and_save(self, output_dir: str = 'data/processed/') -> Dict[str, str]:
        if self.df is None:
            self.load_data()

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        hourly = self.aggregate_by_hour()
        geographic = self.aggregate_by_zone()
        peak = self.peak_hours_analysis()
        vehicle = self.vehicle_type_analysis()
        surge = self.surge_analysis()

        outputs: Dict[str, str] = {}

        outputs['hourly_metrics'] = str(out_dir / 'hourly_metrics.csv')
        hourly.to_csv(outputs['hourly_metrics'], index=False)

        outputs['geographic_metrics'] = str(out_dir / 'geographic_metrics.csv')
        geographic.to_csv(outputs['geographic_metrics'], index=False)
        outputs['peak_hours'] = str(out_dir / 'peak_hours.csv')
        peak.to_csv(outputs['peak_hours'], index=False)

        outputs['vehicle_type'] = str(out_dir / 'vehicle_type.csv')
        vehicle.to_csv(outputs['vehicle_type'], index=False)

        outputs['surge_analysis'] = str(out_dir / 'surge_analysis.csv')
        surge.to_csv(outputs['surge_analysis'], index=False)

        logger.info('Saved %d output files to %s', len(outputs), out_dir)
        return outputs


if __name__ == '__main__':
    processor = RideDataProcessor('data/raw_rides.csv')
    outputs = processor.process_and_save('data/processed')
    print('Processing complete')
    for k, v in outputs.items():
        print(f'- {k}: {v}')
