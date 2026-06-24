# india-health-intelligence-dashboard
End-to-end healthcare data pipeline with ML clustering and interactive BI dashboard built with Python, Plotly Dash, and Scikit-learn
# India Health Intelligence Dashboard

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Dash](https://img.shields.io/badge/Plotly%20Dash-2.17-purple)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Deployed](https://img.shields.io/badge/Deployed-Hugging%20Face-yellow)
![Data](https://img.shields.io/badge/Data-NFHS--5%202019--21-green)

An end-to-end healthcare data pipeline and interactive BI dashboard analysing 
11 health indicators across 20 Indian states using real NFHS-5 government data.

## Live Demo
**[View Dashboard →](https://huggingface.co/spaces/Sibikrish03/healthcare-lakehouse-dashboard)**

---

## Project Overview

This project simulates a production-grade healthcare data lakehouse pipeline:

```
Raw Data (NFHS-5) → ETL Pipeline → SQLite Warehouse → ML Analysis → BI Dashboard → Deployed App
```

---

## Features

| Feature | Description |
|---|---|
| ETL Pipeline | Extract, transform, validate, and load 11 health indicators |
| Star Schema | Fact and dimension tables in SQLite data warehouse |
| ML Clustering | K-Means unsupervised learning auto-tiers 20 states into High / Mid / Needs Attention |
| Overview Tab | All-states ranking, regional comparison, correlation heatmap |
| Single State Analysis | State profile, national rank, strengths and weaknesses, radar chart |
| State Comparison | Head-to-head table, radar overlay, grouped bar chart |
| ML Clustering Tab | Tier breakdown, bubble chart, cluster statistics table |
| Written Insights | Plain-English explanation under every chart |
| Professional UI | No-emoji, clean enterprise-style layout |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Dashboard | Plotly Dash |
| Machine Learning | Scikit-learn (K-Means, StandardScaler) |
| Data Processing | Pandas, NumPy |
| Visualisation | Plotly Express, Plotly Graph Objects |
| Database | SQLite (Star Schema) |
| Deployment | Hugging Face Spaces (Docker) |
| Version Control | GitHub |

---

## Health Indicators Covered

1. Infant Mortality Rate (per 1,000 live births)
2. Maternal Mortality Ratio (per 100,000 live births)
3. Full Immunisation Coverage (%)
4. Child Stunting Rate (% under-5 children)
5. Anaemia in Women (%)
6. Doctors per 10,000 Population
7. Hospital Beds per 10,000 Population
8. Health Expenditure per Capita (INR)
9. Out-of-Pocket Expenditure (%)
10. TB Cases (per 100,000 population)
11. Diabetes Prevalence (%)

---

## Data Sources

- **NFHS-5 (2019-21)** — National Family Health Survey, Ministry of Health and Family Welfare
- **National Health Profile 2022** — Central Bureau of Health Intelligence
- **WHO India** — World Health Organization India Office

---

## Project Structure

```
india-health-intelligence-dashboard/
│
├── app.py               # Main Dash application with ETL + ML + Dashboard
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration for deployment
└── README.md            # Project documentation
```

---

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Sibikrish03/india-health-intelligence-dashboard.git
cd india-health-intelligence-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Open in browser
http://localhost:7860
```

---

## ML Methodology

K-Means clustering was applied to 6 normalised health indicators:
- Infant Mortality Rate
- Maternal Mortality Ratio
- Immunisation Coverage
- Doctor Density
- Health Expenditure
- Out-of-Pocket Cost

States were automatically grouped into 3 tiers without manual labelling,
validating that ML-derived tiers align with known state health rankings.

---

## Key Insights

- Kerala outperforms all states on 9 of 11 indicators, spending 5x more per capita than Bihar
- Assam has India's highest Maternal Mortality Ratio at 215 per 100,000 live births
- ML clustering independently confirms the North-South health divide in India
- Doctor density is the strongest predictor of infant mortality (correlation: -0.9)

---

## Author

**Sibikrish** — Data Engineering & Health Analytics  
[LinkedIn](www.linkedin.com/in/sibi-k-a40b03337) | [Hugging Face](https://huggingface.co/spaces/Sibikrish03/healthcare-lakehouse-dashboard)

---

*Built as part of a Healthcare Data Lakehouse Pipeline project to demonstrate 
end-to-end data engineering, machine learning, and dashboard deployment skills.*
