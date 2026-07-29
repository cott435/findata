# findata.preprocess.calculators

Uniform "derive columns from input frames" abstraction shared by technical
indicators, derived fundamentals, and feature-engineering groups. One base
class (`Calculator`), one registry, one metadata shape — so the same
definitions drive on-the-fly DB computation, the grouped auto-save policy,
and the workbench's hierarchical indicator picker.

## Layout

- `base.py` — `Calculator` ABC, `OutputSpec`, `CALCULATOR_REGISTRY`,
  `register`, and the naming authority `column_name` / `parse_column` /
  `output_owner`. `registry_tree()` is the family → calculator → outputs →
  periods view the UI selectors consume.
- `technical.py` — the moved technical indicators (RSI, Bollinger, ADX, OBV,
  EMAs, …): stateless functions + `calculate_all` + seeding, wrapped in
  `TechnicalCalculator` instances carrying metadata. `INDICATOR_FUNCS` /
  `INDICATOR_OUTPUTS` / `EMA_WINDOWS` / `INDICATOR_WINDOWS` live here (plus
  the legacy `DATA_MAP` / `TECHNICAL_WINDOWS` aliases configs re-exports).
- `fundamental.py` — `build_quarter_panel` (per-ticker merged quarterly view
  with TTM / market-cap / EV helpers) and seven `FundamentalCalculator`
  groups deriving the 24-item modeling catalog; `compute_fundamentals`
  produces the point-in-time long frame stored in `fundamental_data`.

## Import discipline

This package must stay import-light: `findata.database` imports it, so it
may not import anything from findata beyond stdlib + numpy/pandas (no
configs, no utils). That keeps `findata.database → calculators` one-way and
avoids partial-initialization cycles. `findata/preprocess/__init__.py`
resolves its heavy names lazily (PEP 562) for the same reason.

## Adding a calculator

1. Implement it as a `Calculator` subclass (or, for indicators, wrap a
   stateless function in a `TechnicalCalculator`). Set `name`, `family`,
   `group`, `outputs` (one `OutputSpec` per column, with `fe_bounds` /
   `fe_scaling` / `fe_notes`), and `default_params` (`{'periods': [...]}`).
2. `register()` the instance (module import time). Output item names must be
   globally unique.
3. Technical outputs become storable `technical_data` columns automatically
   (`column_name(item, period)`); a new default window also needs adding to
   `EMA_WINDOWS` / `INDICATOR_WINDOWS` if it should be part of the standard
   init set. Fundamental items are stored long in `fundamental_data`.

## Save policy

`findata.configs.SAVE_POLICY` (a `SavePolicy`) maps a calculator's `group`
to whether its on-the-fly results auto-persist. Keys match by longest dotted
prefix: `technical.core` (default windows), `technical.custom` (ad-hoc
windows), `fundamental[.<group>]`, `feature` (never persisted). `DBManager`
consults it in `get_items(..., persist=None)`; pass `persist=True/False` to
override per call.
