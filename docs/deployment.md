# Deployment & Application

`app/main.py` and `app/dashboard.py`

This project includes a dual-layer application architecture: a **FastAPI** backend for programmatic access and a **Streamlit** dashboard for end-users.

## 1. REST API (FastAPI)
The backend service handles real-time inference using the trained XGBoost production model.

### Endpoints
- `GET /`: Health check.
- `POST /predict`: Accepts a JSON object of house features and returns the estimated market value.

### Usage
Start the API:
```bash
uvicorn app.main:app --reload
```

## 2. Interactive Dashboard (Streamlit)
A premium user interface for property valuation.

### Features
- **Real-time Interaction**: Sliders and dropdowns for easy feature input.
- **Dynamic Valuation**: Instant price calculation using the backend engine.
- **Visualization**: Plotly charts showing the influence of key property features.

### Usage
Start the dashboard:
```bash
streamlit run app.dashboard.py
```

## 3. Containerization (Docker)
The entire stack is containerized for seamless deployment to cloud providers (AWS, GCP, Azure).

### Build and Run
```bash
docker build -t house-predictor .
docker run -p 8000:8000 -p 8501:8501 house-predictor
```
