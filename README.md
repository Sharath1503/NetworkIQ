# NetworkIQ — AI-powered inventory analytics

NetworkIQ retains the existing React + TypeScript dashboard and adds a Python ML API for demand forecasting, stockout-risk estimation, cost-aware stock-transfer optimization, and an inventory assistant.

## What is included

- **React dashboard** — existing inventory upload, analytics, search, and recommendations remain available.
- **ML Workbench** (`/app/intelligence`) — train/retrain the model, forecast 7/14/30-day demand, review risk scores, inspect feature importance, optimize transfers, upload sales history, and ask grounded inventory questions.
- **Random Forest demand forecasting** — uses product/store/category, calendar features, price, promotion, lagged sales, and rolling averages. Evaluation uses a chronological holdout and reports MAE, RMSE, and WAPE.
- **Stockout-risk classifier** — included as a demonstration. Current risk labels are proxy/simulated labels; use real historical stockout outcomes before treating its scores as calibrated probabilities.
- **Linear-programming transfer optimization** — SciPy/HiGHS min-cost flow considers source excess, destination forecast shortage, transfer costs, and an unmet-demand penalty.
- **Inventory assistant** — grounded local response mode works without credentials. Optional OpenAI API integration activates when `OPENAI_API_KEY` is configured.
- **Feature importance** — displays aggregated Random Forest feature importance. It is a global model explanation, not proof of causality.
- **Synthetic history** — included in `backend/data/synthetic_sales_history.csv` for a reproducible demo. It is not real sales data.

## Prerequisites

- Node.js 20+ and npm
- Python 3.10+

## Run locally (Windows PowerShell)

Open two terminals from the `networkIQ-AI` folder.

### Terminal 1 — backend

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

On first startup, the API trains the model from the included synthetic sales history. First-time training may take around 30 seconds depending on hardware. API docs are at `http://127.0.0.1:8000/docs`.

### Terminal 2 — frontend

```powershell
npm install
npm run dev
```

Open the local Vite URL shown in the terminal, normally `http://localhost:5173`.

### Optional LLM assistant

Copy `backend/.env.example` to `backend/.env` and set `OPENAI_API_KEY` and optionally `OPENAI_MODEL`. Install `python-dotenv` and load `.env` if you want automatic `.env` loading, or set the environment variables in your terminal before starting the backend. Never commit a real API key.

## Use real sales history

In **ML Workbench**, upload a CSV with these required columns:

- `Date` — parseable date (one row per product/store/day is recommended)
- `Store`
- `Product`
- `UnitsSold`

Optional columns: `Category`, `Price`, `Promotion`.

The API saves the file as `backend/data/sales_history.csv`, trains the model, and then uses it for forecasting. Back up or replace this file when changing datasets. Do not combine unrelated datasets without reconciling product/store identifiers and units.

### Public dataset sources to investigate

- Kaggle retail demand forecasting / store sales datasets
- UCI Machine Learning Repository retail datasets

Check each dataset's license, column definitions, time coverage, and product/store granularity before using it. The included synthetic dataset is for demo and integration testing only. Synthetic-model scores are not evidence of real-world performance.

## API endpoints

- `GET /api/health` — service status
- `POST /api/train-demo` — train/retrain using the current uploaded history or included demo history
- `POST /api/train` — train from included demo history, or optionally upload a CSV file
- `POST /api/upload-history` — upload sales CSV and train
- `POST /api/forecast` — forecast demand and stockout risk
- `POST /api/optimize` — forecast plus transfer optimization
- `POST /api/assistant` — grounded assistant (LLM if configured; local fallback otherwise)

## Important modelling notes

1. The current inventory CSV is a snapshot, not historical sales. The bundled historical file is synthetic, generated from the inventory's `Demand` column as a rough daily baseline.
2. Forecast performance metrics are computed using a chronological holdout, but results on synthetic data should not be represented as results on real retailers.
3. Stockout classifier training uses proxy labels because the demo data does not include observed stockout outcomes. For a defensible model, add historical on-hand stock, replenishment, and actual stockout/availability labels.
4. Transfer cost, revenue protected, and net benefit are estimates based on input columns and assumptions. Validate cost units and operational constraints before real use.
5. The optimizer uses continuous linear programming; the network-flow structure generally yields integral flows for integral inputs, but recommendations are rounded down to whole units for display.
6. No API key is required for the local assistant. The OpenAI-powered response mode is optional and requires your own credentials and usage billing.

## Project structure

```text
networkIQ-AI/
├── src/                      # React app
│   └── pages/IntelligencePage.tsx
├── backend/
│   ├── main.py               # FastAPI endpoints and models
│   ├── generate_demo_data.py # synthetic history generator
│   ├── requirements.txt
│   ├── data/                 # demo history; uploaded history is saved here
│   └── models/               # trained model artifacts (created at runtime)
├── inventory_data_large.csv  # starter inventory snapshot
└── README.md
```
