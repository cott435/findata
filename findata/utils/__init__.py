"""findata.utils — DB access wrappers, featurization entry points, misc helpers.

Public names resolve lazily (PEP 562) so importing a light submodule (e.g.
``findata.utils.timing`` from the database layer) doesn't execute the
matplotlib/seaborn legacy plotting module or the sklearn-heavy featurization
chain, and doesn't re-enter ``db_access`` while ``db_manager`` is mid-import.
"""

import importlib

_DB_ACCESS = 'findata.utils.db_access'
_BUILD = 'findata.utils.build_features'

_LAZY = {
    'get_all_tickers': _DB_ACCESS, 'get_ticker_meta': _DB_ACCESS,
    'get_ticker_data_df': _DB_ACCESS, 'get_all_data': _DB_ACCESS,
    'get_price_data': _DB_ACCESS,
    'build_features': _BUILD, 'apply_features': _BUILD,
    'FeatureBundle': _BUILD, 'FeatureState': _BUILD,
    'sample_universe': 'findata.utils.universe',
    'plot_dfs': 'findata.utils.plottinglegacy',
    'plot_df_hists': 'findata.utils.plottinglegacy',
    'timed': 'findata.utils.timing', 'log_timing': 'findata.utils.timing',
}

__all__ = list(_LAZY)


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'findata.utils' has no attribute {name!r}")
    value = getattr(importlib.import_module(target), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
