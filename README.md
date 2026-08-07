# Axion — AI Investment Advisor

> Fundamental analysis engine for NSE/BSE-listed Indian stocks.  
> Calculates 14+ financial ratios, scores each metric on a 0–10 scale, and returns structured investment recommendations with beginner-friendly explanations.

---

## Architecture

```
frontend/           Single-page dashboard (HTML/CSS/JS)
backend/
├── app.py           FastAPI entry point + middleware
├── config/          Settings, scoring rules, sector benchmarks
├── models/          Pydantic request / response schemas
├── routers/         API route definitions
├── security/        Auth, rate-limiter, headers, payload limit
├── services/        Analysis pipeline engines
│   ├── data_fetcher.py        Yahoo Finance data acquisition
│   ├── data_cleaner.py        Normalisation & sanitisation
│   ├── validation_engine.py   Pre-calculation checks
│   ├── ratio_calculator.py    14 financial ratios
│   ├── historical_engine.py   Multi-year trend analysis
│   ├── scoring_engine.py      Metric → score mapping
│   ├── explanation_engine.py  Natural-language explanations
│   ├── consistency_engine.py  Historical stability scoring
│   ├── benchmark_engine.py    Sector-relative benchmarking
│   ├── valuation_engine.py    DCF / PE / PB valuation
│   ├── qualitative_engine.py  Moat & management quality
│   ├── dashboard_engine.py    Research dashboard assembly
│   ├── insight_engine.py      Executive summary generation
│   └── response_builder.py   Final API response assembly
└── tests/           Automated test suite
```

## Analysis Pipeline

```
Request → Fetch → Clean → Validate → Calculate Ratios
→ Historical Context → Score → Explain → Enrich → Insights → Response
```

## Quick Start

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env        # Edit settings as needed
```

### Run

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` for the dashboard, or `http://127.0.0.1:8000/docs` for Swagger.

### Test

```bash
cd backend
pytest -v
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/analyze` | Full fundamental analysis (JSON body) |
| `GET` | `/api/v1/analyze/{ticker}` | Quick analysis (path param) |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI (debug only) |

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "TCS", "exchange": "NSE"}'
```

## Security

- **API Key Auth** — Optional; enable via `REQUIRE_API_KEY=true`
- **Rate Limiting** — Per-IP, configurable via `RATE_LIMIT_PER_MINUTE`
- **Security Headers** — X-Content-Type-Options, X-Frame-Options, CSP
- **Payload Limits** — Configurable max request size (default 100 KB)
- **Input Validation** — Regex-enforced ticker format, Pydantic schemas

## Configuration

All settings are managed via environment variables (`.env` file) or `config/settings.py` defaults.
See [`.env.example`](backend/.env.example) for the full list.

## Tech Stack

- **Backend**: FastAPI, Pydantic v2, uvicorn
- **Data**: yfinance (Yahoo Finance API)
- **Frontend**: Vanilla HTML/CSS/JS (single-page app)
- **Testing**: pytest

## License

This project is part of a university final year project.