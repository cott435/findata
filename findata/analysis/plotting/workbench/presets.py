"""Named default views for the stock tab, rendered as buttons.

Extend by editing PRESETS: 'overlays' are price-scale keys drawn on the main
chart, 'panels' is a list of indicator panels, each a list of selection keys
('rsi_14', 'fund:earnings_yield', ...).
"""

PRESETS = {
    'Momentum': {
        'overlays': ['ema_close_26', 'ema_close_52'],
        'panels': [['rsi_14', 'rsi_21'], ['stochastic_14', 'willr_14']],
    },
    'Trend': {
        'overlays': ['ema_close_26', 'ema_close_52', 'ema_close_104'],
        'panels': [['adx_14', 'plus_di_14', 'minus_di_14'], ['aroon_up_14', 'aroon_down_14']],
    },
    'Volume/Flow': {
        'overlays': ['ema_close_26'],
        'panels': [['obv', 'ema_obv_26'], ['mfi_20', 'cmf_20']],
    },
    'Volatility': {
        'overlays': ['bb_upper_20', 'bb_middle_20', 'bb_lower_20'],
        'panels': [['atr_14'], ['cci_20']],
    },
    'Fundamentals': {
        'overlays': ['ema_close_52'],
        'panels': [['fund:earnings_yield', 'fund:fcf_yield'],
                   ['fund:sue', 'fund:revenue_surprise']],
    },
}
