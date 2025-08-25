---
description: Repository Information Overview
alwaysApply: true
---

# Ion Chronos Information

## Summary
Ion Chronos is an AI-powered trading assistant platform that combines astrological data analysis, reinforcement learning, and backtesting capabilities. It features a CLI interface with an LLM-powered agent, a web dashboard with frontend/backend components, and specialized tools for financial data processing.

## Structure
- **agent/**: Contains the LLM-powered assistant implementation
- **tools/**: Core utilities for data processing, backtesting, and RL training
- **workspace/**: Main project workspace with multiple components:
  - **backend/**: FastAPI server for job management
  - **frontend/**: React-based dashboard UI
  - **symbols/**: Financial data for various trading symbols
- **ion_env/**: Python virtual environment

## Language & Runtime
**Language**: Python 3.13
**Build System**: Standard Python setuptools
**Package Manager**: pip
**Frontend**: JavaScript/React with Vite

## Dependencies

### Main Python Dependencies
- **Data Processing**: numpy, pandas, scipy, scikit-learn
- **Machine Learning**: torch, stable-baselines3, gymnasium
- **Financial**: yfinance, alpaca-trade-api, ccxt, backtrader, vectorbt
- **Visualization**: matplotlib, plotly, dash
- **Web**: flask, fastapi, uvicorn
- **Astronomy**: ephem, astroquery
- **Database**: pymongo, sqlalchemy, psycopg2-binary

### Backend Dependencies
- **Web Framework**: fastapi, uvicorn[standard]
- **Authentication**: python-jose[cryptography], passlib[bcrypt]
- **Database**: sqlalchemy, asyncpg, alembic
- **Job Queue**: redis, rq

### Frontend Dependencies
- **Framework**: react (^18.2.0)
- **Visualization**: plotly.js-dist-min (^2.24.1)
- **Build Tool**: vite (^5.0.0)

## Build & Installation

### Main Application
```bash
# Create and activate virtual environment
python -m venv ion_env
source ion_env/Scripts/activate  # On Windows: ion_env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the CLI assistant
python main.py
```

### Backend
```bash
cd workspace/backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd workspace/frontend
npm install
npm run dev  # Development server
npm run build  # Production build
```

## Docker
**Docker Compose**: workspace/docker-compose.yml
**Services**:
- **postgres**: PostgreSQL 15 database
- **redis**: Redis 6 for job queue
- **backend**: FastAPI server with mounted volumes
- **worker**: Background job processor
- **frontend**: Vite-based React application

**Run Command**:
```bash
cd workspace
docker-compose up -d
```

## Main Components

### CLI Assistant
- **Entry Point**: main.py
- **Agent Implementation**: agent/ion_chronos_agent.py
- **Memory Storage**: workspace/.memory/

### Backend API
- **Entry Point**: workspace/backend/main.py
- **Job Runner**: workspace/backend/job_runner.py
- **Worker**: workspace/backend/worker.py

### Trading Tools
- **Dataset Builder**: tools/astro_dataset.py
- **Backtesting**: tools/backtest.py
- **RL Training**: tools/rl_train.py
- **Pipeline**: tools/pipeline.py