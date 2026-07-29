"""Uniform calculator abstraction for derived data.

A :class:`Calculator` is a stateless "derive columns from input frames" unit
that carries its own metadata: the columns it produces (:class:`OutputSpec`),
the save-policy group its results belong to, default windows, and short
descriptions of each output's meaning and feature-engineering treatment.
The module-level registry built from these powers three consumers:

- the database's on-the-fly computation of missing (item, period) columns,
- the grouped auto-save policy (``findata.configs.SAVE_POLICY``),
- the workbench's hierarchical indicator selector (family -> calculator ->
  outputs -> periods) via :func:`registry_tree`.

Import discipline: the database layer imports this package, so it must not
import anything from findata beyond the stdlib + numpy/pandas (no configs,
no utils) — that keeps ``findata.database -> findata.preprocess.calculators``
a one-way street and avoids partial-init cycles.
"""

from __future__ import annotations

import functools
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutputSpec:
    """One output column of a calculator, with modeling metadata.

    ``fe_bounds``/``fe_scaling``/``fe_notes`` describe how the value should be
    treated when engineered into model features: its natural range, the
    suggested scaler (a ``get_scaler`` key or a treatment keyword), and any
    extra guidance (winsorize limits, arcsinh, cross-sectional rank, ...).
    """
    item: str
    description: str = ''
    fe_bounds: tuple | None = None
    fe_scaling: str = 'standard'
    fe_notes: str = ''


def _timed_calculate(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = func(self, *args, **kwargs)
        elapsed = time.perf_counter() - start
        log = logging.getLogger(type(self).__module__)
        log.log(logging.DEBUG if elapsed < 0.1 else logging.INFO,
                '%s.calculate done in %.2fs', type(self).__name__, elapsed)
        return result
    return wrapper


class Calculator(ABC):
    """Base for every derived-data producer (technical, fundamental, feature).

    Subclasses set the metadata class attributes and implement
    :meth:`calculate`. ``calculate`` is timing-wrapped automatically, so every
    calculator logs how long its runs take on its own module logger.
    """

    name: str = None
    family: str = None            # 'technical' | 'fundamental' | 'feature'
    group: str = None             # save-policy key, e.g. 'technical.core'
    outputs: tuple[OutputSpec, ...] = ()
    default_params: dict = {}
    description: str = ''
    requires: tuple[str, ...] = ()   # input columns / statement items needed
    seeded: tuple[str, ...] = ()     # outputs needing seeds on incremental update

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'calculate' in cls.__dict__:
            cls.calculate = _timed_calculate(cls.__dict__['calculate'])

    @abstractmethod
    def calculate(self, data, **params) -> pd.DataFrame | pd.Series:
        ...

    def describe(self) -> dict:
        return {
            'name': self.name,
            'family': self.family,
            'group': self.group,
            'description': self.description,
            'params': dict(self.default_params),
            'outputs': [spec.item for spec in self.outputs],
            'seeded': list(self.seeded),
        }

    def output_specs(self) -> dict[str, OutputSpec]:
        return {spec.item: spec for spec in self.outputs}

    def __repr__(self):
        return f'{type(self).__name__}(name={self.name!r}, group={self.group!r})'


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CALCULATOR_REGISTRY: dict[str, Calculator] = {}
_ITEM_TO_CALC: dict[str, str] = {}   # output item -> owning calculator name


def register(calc: Calculator) -> Calculator:
    """Add a calculator instance to the registry (name must be unique)."""
    if calc.name in CALCULATOR_REGISTRY:
        raise ValueError(f'Calculator {calc.name!r} already registered')
    CALCULATOR_REGISTRY[calc.name] = calc
    for spec in calc.outputs:
        other = _ITEM_TO_CALC.get(spec.item)
        if other is not None and other != calc.name:
            raise ValueError(f'Output item {spec.item!r} claimed by both '
                             f'{other!r} and {calc.name!r}')
        _ITEM_TO_CALC[spec.item] = calc.name
    return calc


def output_owner(item: str) -> str:
    """The calculator name that produces ``item`` (e.g. 'bb_upper' -> 'bollinger')."""
    return _ITEM_TO_CALC[item]


def known_items() -> tuple[str, ...]:
    return tuple(_ITEM_TO_CALC)


def column_name(item: str, period: int) -> str:
    """Wide storage column for an (item, period); period 0 keeps the bare name."""
    return item if not period else f'{item}_{period}'


def parse_column(col: str, strict: bool = True) -> tuple[str, int]:
    """Inverse of :func:`column_name` for registry-known items.

    'rsi_14' -> ('rsi', 14); 'ema_close_26' -> ('ema_close', 26);
    'obv' -> ('obv', 0). Items themselves may contain underscores
    ('aroon_up_14'), so the item match takes precedence over the period split.
    ``strict=False`` still splits unregistered columns on a trailing _<int>
    (legacy stored items) instead of raising.
    """
    if col in _ITEM_TO_CALC:
        return col, 0
    stem, _, tail = col.rpartition('_')
    if stem and tail.isdigit() and stem in _ITEM_TO_CALC:
        return stem, int(tail)
    if not strict:
        if stem and tail.isdigit():
            return stem, int(tail)
        return col, 0
    raise KeyError(f'Column {col!r} does not match any registered calculator output')


def registry_tree(families: tuple[str, ...] | None = None) -> dict:
    """Nested metadata view driving hierarchical selection UIs.

    {family: {calculator: {'description', 'group', 'outputs': [items],
    'default_periods': [ints]}}} — outputs/periods combine into storage
    columns via :func:`column_name`.
    """
    tree: dict = {}
    for calc in CALCULATOR_REGISTRY.values():
        if families is not None and calc.family not in families:
            continue
        periods = calc.default_params.get('periods') or [0]
        tree.setdefault(calc.family, {})[calc.name] = {
            'description': calc.description,
            'group': calc.group,
            'outputs': [spec.item for spec in calc.outputs],
            'default_periods': list(periods),
        }
    return tree
