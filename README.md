# Insurance Risk Analytics for AlphaCare Insurance Solutions (ACIS)

## Project Overview
This project analyzes 18 months of historical car insurance data (Feb 2014 - Aug 2015) for ACIS in South Africa. The goal is to identify low-risk segments for premium optimization and build predictive models for risk-based pricing.

## Business Objectives
- Discover low-risk customer segments for targeted marketing
- Develop data-driven pricing models
- Optimize insurance premiums based on risk assessment

## Project Structure

insurance-risk-analytics/
├── .github/workflows/ # CI/CD pipelines
├── data/ # Datasets (tracked by DVC)
├── notebooks/ # Jupyter notebooks for EDA, testing, modeling
│ ├── 01_eda.ipynb
│ ├── 02_hypothesis_testing.ipynb
│ └── 03_modeling.ipynb
├── src/ # Reusable Python modules
│ ├── data_loader.py
│ ├── eda_utils.py
│ ├── hypothesis_tests.py
│ └── modeling.py
├── reports/ # Final deliverables
│ └── final_report.md
├── tests/ # Unit tests
└── requirements.txt # Dependencies


## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/insurance-risk-analytics.git
cd insurance-risk-analytics