# Data Quality Analyser

A Python pipeline for automated data quality analysis and event matching across multiple data sources. Designed for time-series comparison, precision/recall evaluation, and visual presentation.

---

## 🔧 Features

- Collects and merges time-series data from various sources
- Aligns similar events using tolerance-based merging
- Computes comparison metrics: deltas, match rates
- Evaluates precision, recall, false positives/negatives
- Visualizes results with interactive Bokeh dashboard

---

## 📁 Key Files

- `adapters.py` – Data source adapters
- `collect.py` – Core data collection logic
- `compare.py` – Event comparison and delta calculation
- `run.py` / `run_v2.py` – Main entry points
- `show.py` / `show_v2.py` – Visualization scripts
- `present.py` – Generates interactive dashboard
- `*.pkl` – Cached intermediate data
- `dashboard.html` – Final visualization output

---

## 🚀 Quick Start

```bash
# Run the pipeline
python run.py

# Generate dashboard
python present.py

# (Optional) Launch Bokeh visualisation
bokeh serve --show show.py