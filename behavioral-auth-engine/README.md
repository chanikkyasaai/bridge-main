# Behavioral Authentication ML Engine

A sophisticated machine learning engine for behavioral authentication that learns user patterns and provides real-time fraud detection.

## Features

- Real-time behavioral pattern analysis
- FAISS-based similarity search
- Transformer-based advanced analysis
- Behavioral drift detection
- Risk-based policy engine
- Vector storage with HDF5
- Supabase integration
- WebSocket real-time data ingestion

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Run tests:
```bash
pytest tests/
```

4. Start the development server:
```bash
uvicorn src.api.main:app --reload
```

## Architecture

The engine follows a layered architecture:
- **Preprocessing Layer**: Feature extraction and data cleaning
- **FAISS Layer**: Fast similarity search
- **Transformer Layer**: Advanced behavioral analysis
- **Policy Engine**: Risk-based decision making
- **Drift Detector**: Behavioral pattern monitoring

## Documentation

See `/docs` for detailed documentation on API endpoints, model architecture, and deployment guides.
