# Preprocessing

Feature engineering pipeline for stock market data. Each module extracts a distinct category of market signal, scales it, and optionally reduces dimensionality via PCA before model training.

---

## Architecture

All processors extend `PCAProcessor` (defined in `base.py`), which implements a standard pipeline:

```
_feature_engineer() → _scale() → PCA groups → final PCA → .dataset
```

Subclasses override `_feature_engineer()` and optionally `_scale()`. The `.dataset` property returns the final transformed DataFrame aligned to training dates.

---

## Files

### `base.py`

Foundation for all processors.

**`LogStandardScaler`** — sklearn-compatible transformer that applies `log1p` before standard scaling. Useful for right-skewed distributions like volatility.

**`PCAProcessor`** — Abstract base class. Key responsibilities:
- Loads price, EMA, and indicator data from the database via `get_data(dates)`
- Supports six scaler types: `minmax`, `robust`, `standard`, `log_standard`, `quantile`, `power`
- PCA component selection via Kaiser's rule (eigenvalue > 1)
- Grouped PCA: separate transformations per feature family, then an optional final PCA stage
- Arcsinh clipping for outlier handling before scaling

**Shared feature helpers:**
| Function | Description |
|---|---|
| `fe_velocity(windows)` | EMA-based velocity and acceleration for consecutive window pairs; uses log-ratios for positive values, subtraction for bounded indicators |
| `fe_oscillator_momentum(windows)` | Same as above but forces subtraction method (bounded/symmetric indicators) |
| `sig_span(window)` | Signal EMA span ≈ 35% of window size |

**Visualization:** `plot_scale_compare()`, `plot_corr()`, `plot_pca_corr()`, `plot_final_pca_corr()`, `compare_scalers()`

---

### `candles.py`

Candlestick and OHLC price action features.

**`Candle`** extends `PCAProcessor`

Features:
| Name | Formula |
|---|---|
| Body size | `(Close - Open) / Range` |
| Body log | `log(Close / Open)` |
| Range log | `log(High / Low)` |
| Skew log | `log(High / max(O,C)) − log(min(O,C) / Low)` |
| Gap log | `log(Open / Close.shift(1))` |
| Wick ratios | Upper / lower / body as proportion of total range |

- **Scaler:** Robust; arcsinh clipping (±3.5) for log features; body ratio normalized to `[−1, 1]`
- **PCA:** None — features are largely uncorrelated
- **Visualization:** `plot_fe()` draws candlesticks with annotated ratios

---

### `momentum.py`

Overbought/oversold oscillator features.

**`Momentum`** extends `PCAProcessor`

**Source indicators** (from DB): RSI, CCI, Williams %R, Bollinger Band %, Stochastic K

Features per indicator:
- Raw value
- EMAs at windows 4, 8, 16 (configurable by `feature_set`)
- Velocity: EMA differences (`vel4_8`, `vel8_16`, …)
- Acceleration: velocity minus EMA(velocity)

Additional features:
- **OB/OS count:** number of indicators simultaneously beyond their thresholds

Thresholds: RSI 30–70 | CCI ±100 | WillR 20–80 | Stoch K 20–80 | BB% 0.05–0.95

- **Scaler:** Standard
- **PCA groups:** `main` (raw + EMA), `vel`, `acc`
- **Visualization:** `plot_fe()` shows prices, EMAs, indicators, velocity, and acceleration

---

### `trend.py`

Price trend and directional movement features.

**`Trend`** extends `PCAProcessor`

Features:
| Family | Description |
|---|---|
| Relative | `log(Close / EMA)` for each EMA window |
| Velocity | `log(EMA_fast / EMA_slow)` for consecutive EMA pairs |
| Acceleration | velocity − EMA(velocity) |
| Directional | ADX, +DI, −DI, DI difference |

EMA windows by `feature_set`: Low `[6,12,26,52]` → Med adds 104 → High adds 208

- **Scaler:** Standard
- **PCA groups:** `rel`, `vel`, `acc`, `di`
- **Final PCA:** Yes (0.95 variance retention)
- **Visualization:** `plot_fe()` shows prices, EMAs, velocities, accelerations, and DI indicators

---

### `volatility.py`

Volatility and price dispersion features.

**`Volatility`** extends `PCAProcessor`

| Feature | Description |
|---|---|
| BB bandwidth | `(BB_upper − BB_lower) / BB_middle` |
| Normalized ATR | `ATR / Close` |
| Standard vol | Rolling std of log returns |
| Parkinson vol | `√(mean((log(H/L))² / 4ln2))` |
| Rogers-Satchell vol | Log-based high-low dispersion |
| Garman-Klass vol | *(high feature_set only)* |
| Realized vol proxy | *(high feature_set only)* |

Windows by `feature_set`: Low `[7, 20]` → Med/High add 100

- **Scaler:** `log_standard` (log transform + standard scaling)
- **PCA:** Kaiser's rule; final PCA (0.95 variance)
- **Visualization:** `plot_fe()` shows close prices and scaled volatility measures

---

### `volume.py`

Volume momentum and price-volume divergence features.

**`Volume`** extends `PCAProcessor`

**Momentum block** (CMF and MFI):
- Raw values, EMAs at `[4, 8, 16]`, velocity, and acceleration

**Z-score velocity block** (EMA windows `[12, 26, 52]`):
- `(vel − rolling_mean) / rolling_std` for price EMAs, OBV EMAs, and AD EMAs

**Divergence features:**
- `price_ad_diff`: price velocity minus AD velocity
- `obv_ad_diff`: OBV velocity minus AD velocity

- **Scaler:** Custom `_scale()` — standard scaling for all features except z-velocity columns (passed through unchanged)
- **PCA groups:** `mom_main`, `mom_vel`, `z_vel1`, `z_vel2`, `vel_diff12_26`, `vel_diff26_52`
- **Final PCA:** Yes (0.95 variance)
- **Visualization:** `plot_fe()` (price + indicators + velocities), `plot_raw()` (raw volume data)

---

## Feature Set Levels

Each processor accepts a `feature_set` argument that controls the breadth of features computed:

| Level | Description |
|---|---|
| `low` | Minimal windows; fastest; smallest input dimension |
| `med` | Medium windows; balanced |
| `high` | All windows and advanced features; richest representation |
