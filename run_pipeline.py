from __future__ import annotations

import argparse
from pathlib import Path

from src.synthetic_data_generator import generate_synthetic_ride_data
from src.data_processor import RideDataProcessor

def print_quality_check(csv_path: str) -> None:
    # sanity-check generated data: zone distribution, completion variance, surge/wait correlation.
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    print('\n--- Quality Metrics ---')
    print(f'Rows: {len(df):,}')
    print(f'Date range: {df["timestamp"].min()[:10]} to {df["timestamp"].max()[:10]}')
    
    print('\nZone distribution:')
    zone_share = df['pickup_zone'].value_counts(normalize=True).round(3) * 100
    for zone, pct in zone_share.items():
        print(f'  {zone}: {pct:.1f}%')
    
    print(f'\nCompletion rate: {df["completed"].mean():.2%} (variance: {df["completed"].var():.4f})')
    print(f'Completion by zone:')
    for zone in sorted(df['pickup_zone'].unique()):
        rate = df[df['pickup_zone'] == zone]['completed'].mean()
        print(f'  {zone}: {rate:.2%}')
    
    corr = df['surge_multiplier'].corr(df['wait_time_minutes'])
    print(f'\nSurge ↔ Wait-time correlation: {corr:.3f} (expect 0.1-0.3)')
    print(f'Wait-time range: {df["wait_time_minutes"].min():.1f}-{df["wait_time_minutes"].max():.1f} min')
    print(f'Surge range: {df["surge_multiplier"].min():.2f}-{df["surge_multiplier"].max():.2f}x')
    print()

def main() -> int:
    parser = argparse.ArgumentParser(description='Generate synthetic ride data and build BI-ready aggregations.')
    parser.add_argument('--n-rides', type=int, default=450_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--raw-path', type=str, default=str(Path("data") / "raw_rides.csv"))
    parser.add_argument('--processed-dir', type=str, default=str(Path("data") / "processed"))
    parser.add_argument('--skip-generate', action='store_true', help='Only run processing step')
    parser.add_argument('--check-quality', action='store_true', help='Print data quality metrics after generation')
    args = parser.parse_args()

    raw_path = Path(args.raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_generate:
        df = generate_synthetic_ride_data(n_rides=args.n_rides, seed=args.seed)
        df.to_csv(raw_path, index=False)
        print(f'Saved raw rides: {raw_path} ({len(df):,} rows)')
        
        if args.check_quality:
            print_quality_check(str(raw_path))

    processor = RideDataProcessor(str(raw_path))
    outputs = processor.process_and_save(args.processed_dir)

    print('\nProcessed outputs:')
    for name, out_path in outputs.items():
        print(f'- {name}: {out_path}')

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
