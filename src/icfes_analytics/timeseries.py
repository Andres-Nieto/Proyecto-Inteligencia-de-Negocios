from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


def parse_periodo_flexible(p) -> Optional[pd.Timestamp]:
    s = ''.join(ch for ch in str(p) if ch.isdigit())
    if len(s) == 6:  # YYYYMM
        y = int(s[:4]); m = int(s[4:6])
        m = min(max(m, 1), 12)
        return pd.Timestamp(year=y, month=m, day=1)
    if len(s) == 5:  # YYYYS (S in {1,2,3,4})
        y = int(s[:4]); q = int(s[4])
        mapping = {1:3, 2:6, 3:9, 4:12}
        m = mapping.get(q, 12)
        return pd.Timestamp(year=y, month=m, day=1)
    if len(s) == 4:  # YYYY
        y = int(s)
        return pd.Timestamp(year=y, month=12, day=1)
    try:
        return pd.to_datetime(p)
    except Exception:
        return pd.NaT


def aggregate_series(df: pd.DataFrame, period_col: str = 'periodo', value_col: str = 'punt_global',
                     parser: Callable = parse_periodo_flexible) -> pd.Series:
    agg = (df[[period_col, value_col]]
           .dropna()
           .assign(ts=lambda d: d[period_col].apply(parser))
           .dropna(subset=['ts'])
           .groupby('ts', as_index=True)[value_col]
           .mean()
           .sort_index())
    return agg


def stl_adf(series: pd.Series, period: int = 4):
    series = series.dropna()
    adf_stat, pvalue, usedlag, nobs, crit, icbest = adfuller(series.values, autolag='AIC')
    try:
        decomp = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
    except Exception:
        decomp = None
    return {
        'adf_stat': adf_stat,
        'pvalue': pvalue,
        'crit': crit,
        'decomp': decomp,
    }


def fit_arima_small_grid(
    series: pd.Series,
    orders: Optional[Iterable[Tuple[int, int, int]]] = None,
    p_values: Optional[Iterable[int]] = None,
    d_values: Optional[Iterable[int]] = None,
    q_values: Optional[Iterable[int]] = None,
    retry_small_if_none: bool = True,
):
    """Selecciona un ARIMA por AIC probando una malla de órdenes y pronostica 1 paso.

    Parámetros
    - series: Serie temporal indexada por fecha.
    - orders: Iterable opcional de tuplas (p,d,q) a probar. Si se proporciona, ignora p/d/q_values.
    - p_values, d_values, q_values: Rango de valores a probar si no se pasa `orders`.
      Por defecto usa una malla ampliada: p,q in {0,1,2,3}, d in {0,1,2} (excluye (0,0,0)).
    - retry_small_if_none: Si ningún modelo ajusta, reintenta con malla pequeña p,d,q in {0,1}.

    Retorna
    - (order_sel, res_sel, y_pred, y_true, train, test, metrics)
    """
    series = series.asfreq('QS-MAR').sort_index().dropna()
    # Construir malla por defecto (ampliada) si no se especifica
    if orders is None:
        if p_values is None:
            p_values = [0, 1, 2, 3]
        if d_values is None:
            d_values = [0, 1, 2]
        if q_values is None:
            q_values = [0, 1, 2, 3]
        orders = [(p, d, q) for p in p_values for d in d_values for q in q_values if not (p == 0 and d == 0 and q == 0)]

    train = series.iloc[:-1]
    test = series.iloc[-1:]

    def _try_fit(order_list: Iterable[Tuple[int, int, int]]):
        best_local = None
        best_aic_local = np.inf
        for order in order_list:
            try:
                res = sm.tsa.ARIMA(train, order=order).fit()
                if res.aic < best_aic_local:
                    best_aic_local = res.aic
                    best_local = (order, res)
            except Exception:
                continue
        return best_local, best_aic_local

    best, best_aic = _try_fit(orders)

    # Fallback robusto para series muy cortas
    if best is None and retry_small_if_none:
        small_orders = [(p, d, q) for p in [0, 1] for d in [0, 1] for q in [0, 1] if not (p == 0 and d == 0 and q == 0)]
        best, best_aic = _try_fit(small_orders)

    if best is None:
        raise RuntimeError("No fue posible ajustar ningún modelo ARIMA con los órdenes probados.")

    order_sel, res_sel = best
    fc = res_sel.forecast(steps=1)
    y_pred = float(fc.iloc[0]); y_true = float(test.iloc[0])
    mae = mean_absolute_error([y_true],[y_pred])
    rmse = mean_squared_error([y_true],[y_pred]) ** 0.5
    metrics = {'AIC': float(best_aic), 'MAE': float(mae), 'RMSE': float(rmse)}
    return order_sel, res_sel, y_pred, y_true, train, test, metrics


def plot_arima_forecast(train: pd.Series, test: pd.Series, y_pred: float, order_sel: Tuple[int,int,int]):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(train.index, train.values, label='train', color='#1f77b4', lw=2)
    ax.plot(test.index, test.values, label='test', color='black', lw=2)
    ax.plot(test.index, [y_pred], 'o', color='#d62728', ms=6, label='ARIMA pred')
    ax.set_title(f'ARIMA{order_sel} - pronóstico 1 paso')
    ax.set_xlabel('Periodo'); ax.set_ylabel('Puntaje promedio')
    ax.grid(True, alpha=0.9)
    ax.legend(frameon=False)
    for s in ['top','right']:
        ax.spines[s].set_visible(False)
    plt.show()
