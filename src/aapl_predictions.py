import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from datetime import datetime, timezone, timedelta

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"  # paper trading

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

symbol = "AAPL"
start = "2022-01-01"
end = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

df = api.get_bars(symbol, TimeFrame.Day, start=start, end=end, feed="iex").df
df = df.reset_index()

# --- Features ---
df["return_1"] = df["close"].pct_change()

df["sma_10"] = df["close"].rolling(10).mean()
df["sma_20"] = df["close"].rolling(20).mean()

df["vol_10"] = df["return_1"].rolling(10).std()

delta = df["close"].diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df["rsi_14"] = 100 - (100 / (1 + rs))

# --- Target: tomorrow return ---
df["target_return_next"] = df["close"].pct_change().shift(-1)

# Clean rows with NaNs from rolling + shift
df = df.dropna().copy()

features = ["return_1", "sma_10", "sma_20", "vol_10", "rsi_14"]
X = df[features]
y = df["target_return_next"]

# --- Time-based split (80/20) ---
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
print("MAE (next-day return):", mae)

# --- Direction accuracy (up/down) ---
actual_dir = (y_test.values > 0).astype(int)   # 1 = up, 0 = down or flat
pred_dir = (pred > 0).astype(int)

direction_acc = (actual_dir == pred_dir).mean()
print("Direction accuracy:", direction_acc)

tp = ((pred_dir == 1) & (actual_dir == 1)).sum()
tn = ((pred_dir == 0) & (actual_dir == 0)).sum()
fp = ((pred_dir == 1) & (actual_dir == 0)).sum()
fn = ((pred_dir == 0) & (actual_dir == 1)).sum()

print("TP:", tp, "FP:", fp, "TN:", tn, "FN:", fn)

# Optional: show a few predictions
out = df.iloc[split:][["timestamp", "close"]].copy()
out["actual_next_return"] = y_test.values
out["pred_next_return"] = pred
print(out.head(10))

# --- Plot actual vs predicted returns ---
plt.figure()
plt.plot(out["timestamp"], out["actual_next_return"], label="Actual")
plt.plot(out["timestamp"], out["pred_next_return"], label="Predicted")
plt.legend()
plt.title(f"{symbol} Next-Day Return: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Return")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()