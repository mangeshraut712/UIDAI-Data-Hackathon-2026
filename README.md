<div align="center">

# 🏆 UIDAI Data Hackathon 2026

### Aadhaar enrolment and update analysis with 4.9M records, statistical validation, and publication-ready visuals

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas_+_Polars-2026-150458?style=for-the-badge)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge)](https://scikit-learn.org/)
[![Hackathon Submission](https://img.shields.io/badge/Hackathon-Submission-0EA5E9?style=for-the-badge)]()

[Features](#-features) • [Stack](#-stack) • [Quick Start](#-quick-start) • [Structure](#-project-structure) • [Scripts](#-scripts) • [Contact](#-contact)

</div>

---

## Table of Contents

- [About](#-about)
- [Features](#-features)
- [Stack](#-stack)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Scripts](#-scripts)
- [Contact](#-contact)

## About

This repository contains the full pipeline for a UIDAI Hackathon submission analyzing Aadhaar enrolment, demographic updates, and biometric updates across nearly 5 million records. The workflow combines preprocessing, statistical validation, machine learning, and premium reporting artifacts for a submission-ready package.

## Features

- 4,938,813 combined records across enrolment, demographic, and biometric datasets
- Statistical testing, clustering, forecasting, and anomaly detection
- Publication-style charts and an executive summary infographic
- A deployment-ready interactive HTML dashboard
- A single orchestration entry point that runs the full pipeline end to end

## Stack

| Area | Technologies |
| --- | --- |
| Language | Python 3.12+ |
| Data | pandas, Polars, NumPy, pyarrow |
| Analysis | scikit-learn, SciPy, statsmodels |
| Visualization | Matplotlib, Seaborn, Plotly |
| Output | PDFs, PNGs, and HTML dashboards |

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_analysis.py
```

Generate individual outputs if you want to run specific stages:

```bash
python src/visualization/premium_visualizations.py
python src/visualization/generate_infographic.py
python src/visualization/interactive_dashboard.py
```

## Project Structure

```text
UIDAI-Data-Hackathon-2026/
├── data/raw/                  # Source CSV datasets
├── reports/report/            # Final PDF and LaTeX submission files
├── src/preprocessing/         # Data loading and normalization
├── src/analysis/              # Statistical and ML analysis modules
├── src/visualization/         # Charts, infographic, dashboard
├── visualizations/charts/     # Generated figures
├── visualizations/infographics/# Executive summary assets
├── visualizations/interactive/ # HTML dashboard output
└── run_analysis.py            # Pipeline orchestrator
```

## Scripts

Primary entry points:

```bash
python run_analysis.py
python src/analysis/comprehensive_analysis.py
python src/analysis/advanced_analytics.py
python src/analysis/iit_level_analytics.py
```

Visualization commands:

```bash
python src/visualization/premium_visualizations.py
python src/visualization/wow_factor.py
python src/visualization/generate_infographic.py
python src/visualization/interactive_dashboard.py
```

## Contact

- Project site: [mangeshraut.pro](https://mangeshraut.pro)
- Repository: [mangeshraut712/UIDAI-Data-Hackathon-2026](https://github.com/mangeshraut712/UIDAI-Data-Hackathon-2026)
- Issues: open a GitHub issue in this repository
