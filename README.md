# ML Stock Prediction (Alpaca + Random Forest)

by Camille Yabu and Antonio Wydler

This project pulls daily stock bar data from Alpaca (Paper Trading) and trains a machine learning model to predict the next trading day's movement.

## What it does
- Downloads historical daily bars (e.g., from 2022 to present)
- Builds technical indicator features (SMA, RSI, volatility)
- Trains a Random Forest model
- Evaluates:
  - MAE on next-day return regression
  - Direction accuracy (up vs down)

## Example Result
On AAPL daily data (2022–recent), the model reached ~57.7% next-day direction accuracy.

## Setup

### 1) Create a virtual environment (optional)
```bash
python -m venv venv
source venv/bin/activate   # mac/linux
# venv\Scripts\activate    # windows