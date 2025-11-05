"""ICFES Analytics package: clustering and time-series helpers.

Modules:
- clustering: six clustering methods and plotting helpers
- timeseries: flexible period parser, aggregation, ARIMA utilities
- rnn_numpy: minimal NumPy RNN for one-step forecast
- plots: shared plotting style (BTC-like)
"""

from .plots import apply_btc_style
from .timeseries import (
    parse_periodo_flexible,
    aggregate_series,
    stl_adf,
    fit_arima_small_grid,
    plot_arima_forecast,
)
from .rnn_numpy import (
    build_sequences,
    rnn_train,
    rnn_predict_next,
    forecast_one_step_numpy,
)
