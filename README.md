# 🛒 Rossmann Sales Forecaster
### End-to-End ML Pipeline · Flask API · Docker · Cloud Deployment

[![Live Demo](https://img.shields.io/badge/Live%20Demo-rossmann--sales--v2.onrender.com-brightgreen?style=for-the-badge&logo=render)](https://rossmann-sales-v2.onrender.com/)
[![Docker](https://img.shields.io/badge/Docker-python%3A3.11--slim-blue?style=for-the-badge&logo=docker)](https://hub.docker.com/)
[![Model](https://img.shields.io/badge/Model-DecisionTreeRegressor-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![R²](https://img.shields.io/badge/R%C2%B2%20Score-0.81-success?style=for-the-badge)](https://rossmann-sales-v2.onrender.com/)

---

## Overview

A **production-ready retail demand forecasting system** built on the [Kaggle Rossmann Store Sales dataset](https://www.kaggle.com/competitions/rossmann-store-sales). This project covers the full ML lifecycle — from raw data ingestion and exploratory analysis through model training, containerization, and automated cloud deployment.

The goal: predict daily store-level sales to support inventory planning and business decision-making, deployed as a live REST API accessible to any downstream system.

---

## Live Demo

> **API Endpoint:** [`https://rossmann-sales-v2.onrender.com/`](https://rossmann-sales-v2.onrender.com/)

Send a `POST /predict` request with store features and receive a sales forecast in real time.

---

## Pipeline Architecture

```
Raw Data (Kaggle)
      │
      ▼
 Data Ingestion          train.csv + store.csv merged (1M+ rows)
      │
      ▼
 EDA & Feature Eng.      Trends · Promotions · Temporal · Store segments
      │
      ▼
 Preprocessing           MinMaxScaler · Null handling · Type casting
      │
      ▼
 Model Training           DecisionTreeRegressor (R²: 0.81, RMSE: 1289)
      │
      ▼
 Flask REST API           POST /predict · Gunicorn production server
      │
      ▼
 Docker Image             python:3.11-slim · 487 MB · Port 5000
      │
      ▼
 Cloud Deployment         Render.com · Auto-deploy from Docker Hub
```

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | [Kaggle – Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) |
| **Files** | `train.csv` + `store.csv` (joined on `Store`) |
| **Size** | ~1 million rows |
| **Target** | `Sales` (daily store revenue) |

**Key Features Used:**

| Feature | Description |
|---|---|
| `Store` | Unique store identifier |
| `DayOfWeek` / `Month` / `WeekOfYear` | Temporal demand signals |
| `Promo` | Whether a promotion was active (binary) |
| `CompetitionDistance` | Distance to nearest competitor store (metres) |
| `StoreType` | Store category (a / b / c / d) |
| `Assortment` | Product assortment level (basic / extra / extended) |

---

## Model & Performance

| Metric | Value |
|---|---|
| **Algorithm** | `DecisionTreeRegressor` |
| **Scaler** | `MinMaxScaler` |
| **R² Score** | `0.81` |
| **MAE** | `1,052` |
| **RMSE** | `1,289` |

> The Decision Tree Regressor was selected over Linear Regression after benchmarking both models on R² and RMSE. Its ability to capture non-linear interactions between promotions, store type, and temporal features gave it a decisive edge.

---

## API Reference

### `POST /predict`

Accepts a JSON body with store features and returns a predicted sales figure.

**Request**
```json
POST https://rossmann-sales-v2.onrender.com/predict
Content-Type: application/json

{
  "Store": 1,
  "DayOfWeek": 5,
  "Promo": 1,
  "StoreType": "a",
  "Assortment": "basic",
  "CompetitionDistance": 1270,
  "Month": 7,
  "WeekOfYear": 31
}
```

**Response**
```json
{
  "predicted_sales": 7842.15
}
```

---

## Running Locally

### Option 1 — Docker (Recommended)

```bash
# Pull the image
docker pull <your-dockerhub-username>/rossmann-sales:latest

# Run the container
docker run -p 5000:5000 <your-dockerhub-username>/rossmann-sales:latest
```

API will be available at `http://localhost:5000`

### Option 2 — From Source

```bash
# Clone the repo
git clone https://github.com/<your-username>/rossmann-sales-forecaster.git
cd rossmann-sales-forecaster

# Install dependencies
pip install -r requirements.txt

# Start the Flask server
gunicorn app:app --bind 0.0.0.0:5000
```

---

## Docker Details

```dockerfile
FROM python:3.11-slim
# Image size: ~487 MB
# Exposed port: 5000
# Server: Gunicorn (production WSGI)
```

The image uses the slim Python 3.11 base to minimise attack surface and keep the deployment footprint lean while remaining fully production-capable.

---

## MLOps & CI/CD

This project follows a lightweight but real MLOps workflow:

```
git push origin main
       │
       ▼
  Docker Hub          Automated image build & push
       │
       ▼
  Render.com          Detects new image → zero-downtime redeploy
```

Every code change triggers a full rebuild and redeploy automatically — no manual SSH, no downtime.

---

## Project Structure

```
rossmann-sales-forecaster/
│
├── app.py                  # Flask application & /predict endpoint
├── model.py                # Training pipeline (EDA → preprocessing → model)
├── model.pkl               # Serialised DecisionTreeRegressor
├── scaler.pkl              # Fitted MinMaxScaler
│
├── data/
│   ├── train.csv           # Rossmann training data
│   └── store.csv           # Store metadata
│
├── notebooks/
│   └── eda.ipynb           # Exploratory data analysis
│
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Key Learnings

- **EDA drives feature selection** — promotional timing and store type were the strongest predictors of sales variance.
- **Tree models outperform linear baselines** on retail data with categorical interactions and seasonal non-linearity.
- **Containerisation simplifies deployment** — Docker eliminated environment inconsistency across dev and prod.
- **Free-tier cloud is viable for POCs** — Render.com's auto-deploy pipeline delivered CI/CD without infrastructure cost.

---

## Future Improvements

- [ ] Add `XGBoostRegressor` / `LightGBM` for performance uplift
- [ ] Incorporate external features (holidays, weather, events)
- [ ] Add prediction confidence intervals
- [ ] Implement model monitoring & drift detection
- [ ] Expose a Streamlit dashboard for non-technical stakeholders

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML | scikit-learn (DecisionTreeRegressor, MinMaxScaler) |
| API | Flask + Gunicorn |
| Container | Docker (`python:3.11-slim`) |
| Registry | Docker Hub |
| Cloud | Render.com (free tier) |
| Data | Kaggle Rossmann Store Sales |

---

*Built as a production-ready ML proof of concept demonstrating the full pipeline from raw data to live cloud API.*
