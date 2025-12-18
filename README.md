# RideGen Analytics
## Ride-Hailing Market Dynamics Analysis

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📊 Project Overview

Synthetic ride-hailing dataset generator that produces 450,000 rides with earned relationships. Surge pricing correlates with wait times. Cancellations spike under demand pressure. Distance distributions shift by zone. The data behaves like real ride-hailing systems because the relationships between variables mirror reality.

The pipeline exports 5 analysis-ready CSV files. Each file delivers pre-aggregated metrics for specific business questions. No cleanup needed. Load the CSVs into your BI tool and start building dashboards. The project demonstrates structured data generation, ETL pipeline design, and realistic synthetic modeling for analytics work.

You get clear documentation, reproducible outputs, and a reference implementation for data engineering discipline. Use it to build portfolio projects, test analytics tools, or prototype ride-hailing features without dealing with messy production data or privacy restrictions.

## 🎯 Skills Demonstrated

- Structured data generation with realistic dependencies
- ETL pipeline design (extract, validate, transform, aggregate, export)
- Data quality validation and correlation verification
- Clean Python code with type hints and docstrings
- Git workflow with proper .gitignore structure
- Technical documentation

## 📈 Output Files

Pipeline generates 5 BI-ready CSV files in `data/processed/`:

1. **hourly_metrics.csv** - 8,760 hourly records (365 days × 24 hours) with ride volume, completion rate, average fare, wait time, surge multiplier
2. **geographic_metrics.csv** - 6 zone-level summaries with total rides, revenue, distance, completion variance
3. **peak_hours.csv** - 168 records (24 hours × 7 days) showing demand distribution across week
4. **vehicle_type.csv** - Zone × vehicle type breakdown (18 combinations) with completion rates and revenue
5. **surge_analysis.csv** - Zone × hour grid (144 combinations) correlating surge, wait-time, and cancellations

## 🚀 Quickstart (Windows)

```powershell
pip install -r requirements.txt
python run_pipeline.py --n-rides 450000 --check-quality
```

The `--check-quality` flag prints zone distribution, completion variance, and surge↔wait correlation to verify data realism.

## Data Generation Assumptions (Synthetic)

The generator is intentionally *structured*, not random noise. Every metric is earned from upstream dependencies.

**Zone demand weights:**
- Vake: 28% | Saburtalo: 24% | Old Town: 18% | Shardeni: 12% | Nadzaladevi: 10% | Gldani: 8%

**Time structure:**
- Weekday variance: Friday/Saturday +18-25%, Monday -15%, Wed-Thu baseline
- Seasonality: Summer (Jun-Aug) evenings +12%, winter (Dec-Feb) mornings +8%
- Baseline hourly pattern: morning peak 6-9am, evening peak 6-9pm, nights (10pm-5am) low

**Wait time → demand pressure:**
- Wait time driven by hour-load and zone-load (sigmoid curve)
- High-demand hours (6-9pm) → 6-12 min waits | Off-peak → 2-4 min
- High-demand zones (Vake/Saburtalo) → +2-3 min vs trailing zones

**Surge logic:**
- Surge = f(hour-load, zone-load) + small earned tie to wait-time
- Peak hours + high-demand zones → 1.8-2.2x surge
- Off-peak, low-demand zones → 1.0-1.1x surge
- Range: 1.0–3.0x (hard cap)

**Cancellations (variable, not flat):**
- Base cancel rate: 5%
- Wait-time pressure: sigmoid (5-20 min wait range) adds +0-20% cancel prob
- Surge pressure: +0-10% cancel prob (riders balk at 2.5x+ pricing)
- Result: 2-35% cancel rate depending on hour/zone, not uniform 7%

**Distance by zone (gamma-distributed, not uniform):**
- Old Town: gamma(1.6, 1.0) → short trips (0.6-3 km typical)
- Vake/Saburtalo: gamma(2.2-2.3, 1.4-1.5) → medium (1.5-5 km typical)
- Gldani/Nadzaladevi: gamma(2.5-2.9, 1.7-1.9) → longer (2-8 km typical)

**Vehicle mix (zone + time + weekend aware):**
- Global: 72% Economy, 20% Comfort, 8% XL
- Vake boost: +10% Comfort (premium zone)
- Saburtalo boost: +6% Comfort
- Evening boost (6pm-midnight): +5% Comfort
- Weekend boost: +6% XL (leisure trips)
- Constraints: Economy floor 45%, Comfort ceiling 45%, XL ceiling 25%

The goal is realism that produces *earned* relationships (e.g., surge vs wait-time correlation) and non-flat completion rates.

## 📁 Project Structure

```
Competitive-Analytics-Dashboard/
│
├── run_pipeline.py                  # Main execution script
├── src/
│   ├── synthetic_data_generator.py  # 450K ride generator with realistic logic
│   └── data_processor.py            # ETL pipeline and aggregations
│
├── data/
│   ├── raw_rides.csv                # Generated locally (gitignored)
│   └── processed/                   # 5 output CSVs (gitignored)
│
├── requirements.txt                 # pandas, numpy
├── .gitignore                       # Excludes data outputs
└── README.md
```

## 🔗 Related Projects

- [Bolt Data Analytics](https://github.com/z12ob/Bolt-Data-Analytics) - Demand-supply optimization with 839 hours of operational data
- [Uber Data Analytics](https://github.com/z12ob/Uber-Data-Analytics) - 148K booking analysis with cancellation prediction

## 📌 Notes

- All data files (`data/raw_rides.csv`, `data/processed/*.csv`) are gitignored to keep the repo lightweight
- For screenshots or testing, generate with `--n-rides 5000`
- Generator seed is fixed (42) for reproducibility

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

**Guram Melikidze**
- GitHub: [@z12ob](https://github.com/z12ob)
- LinkedIn: [Guram Melikidze](https://www.linkedin.com/in/guram-melikidze/)

