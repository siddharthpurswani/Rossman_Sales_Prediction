Live Demo: https://rossmann-sales-v2.onrender.com/

End-to-End ML Pipeline: Data ingestion → EDA → Training → Docker → Cloud deployment

Dataset: Kaggle Rossmann (1M rows: train.csv + store.csv)

Features: Store ID, date (day/week/month), promo, competition distance, storetype, store assorment

Model: DecisionTreeRegressor + MinMaxScaler (R²: 0.81)

API: Flask + Gunicorn production server (POST /predict)

Docker: python:3.11-slim, 487MB image, port 5000

Cloud: Render.com free tier (auto-deploy from Docker Hub)

Metrics: R²=0.81, MAE=1052, RMSE=1289

MLOps: Git push → Docker Hub → Render CI/CD (zero downtime)
