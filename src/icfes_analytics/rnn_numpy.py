from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def build_sequences(y: np.ndarray, window: int):
    Xs, ys = [], []
    for i in range(window, len(y)):
        Xs.append(y[i-window:i])   # (w,1)
        ys.append(y[i])            # (1,)
    return np.array(Xs), np.array(ys)


def rnn_train(X: np.ndarray, y: np.ndarray, hidden_size=16, epochs=600, lr=0.01, clip=5.0, seed=42, verbose: bool = False):
    rng = np.random.default_rng(seed)
    in_size = 1
    W_xh = rng.normal(0, 0.1, size=(in_size, hidden_size))
    W_hh = rng.normal(0, 0.1, size=(hidden_size, hidden_size))
    b_h  = np.zeros((hidden_size,))
    W_hy = rng.normal(0,  0.1, size=(hidden_size, 1))
    b_y  = np.zeros((1,))

    for epoch in range(epochs):
        loss_epoch = 0.0
        for i in range(X.shape[0]):
            x_seq = X[i]
            target = y[i]
            h_prev = np.zeros((hidden_size,))
            hs, xs = [], []
            for t in range(x_seq.shape[0]):
                xt = x_seq[t, 0]
                xs.append(xt)
                a_t = xt * W_xh[0] + h_prev @ W_hh + b_h
                h_t = np.tanh(a_t)
                hs.append(h_t)
                h_prev = h_t
            h_last = hs[-1]
            y_hat = h_last @ W_hy + b_y
            err = y_hat - target
            loss_epoch += 0.5 * float(err**2)
            # grads
            dW_xh = np.zeros_like(W_xh)
            dW_hh = np.zeros_like(W_hh)
            db_h  = np.zeros_like(b_h)
            dW_hy = np.zeros_like(W_hy)
            db_y  = np.zeros_like(b_y)
            dy = err
            dW_hy += np.outer(h_last, dy)
            db_y  += dy
            dh = (W_hy @ dy).ravel()
            for t in reversed(range(x_seq.shape[0])):
                h_t = hs[t]
                h_prev_t = hs[t-1] if t > 0 else np.zeros_like(h_t)
                dtanh = (1.0 - h_t*h_t) * dh
                dW_xh[0] += xs[t] * dtanh
                dW_hh    += np.outer(h_prev_t, dtanh)
                db_h     += dtanh
                dh = W_hh @ dtanh
            for g in (dW_xh, dW_hh, db_h, dW_hy, db_y):
                np.clip(g, -clip, clip, out=g)
            W_xh -= lr * dW_xh
            W_hh -= lr * dW_hh
            b_h  -= lr * db_h
            W_hy -= lr * dW_hy
            b_y  -= lr * db_y
        if verbose and ((epoch+1) % 100 == 0 or epoch == 0):
            print(f"Epoch {epoch+1}/{epochs} - loss {loss_epoch/max(1,len(X)):.6f}")
    return {
        'W_xh': W_xh, 'W_hh': W_hh, 'b_h': b_h, 'W_hy': W_hy, 'b_y': b_y,
        'hidden_size': hidden_size, 'window': X.shape[1]
    }


def rnn_predict_next(model, last_seq: np.ndarray) -> float:
    W_xh, W_hh, b_h, W_hy, b_y = model['W_xh'], model['W_hh'], model['b_h'], model['W_hy'], model['b_y']
    h = np.zeros((model['hidden_size'],))
    for t in range(last_seq.shape[0]):
        xt = float(last_seq[t, 0])
        a = xt * W_xh[0] + h @ W_hh + b_h
        h = np.tanh(a)
    y_hat = float(h @ W_hy + b_y)
    return y_hat


def forecast_one_step_numpy(agg_series: pd.Series, freq='QS-MAR', window=4, hidden_size=16, epochs=600,
                            lr=0.01, clip=5.0, seed=42, plot=True, verbose: bool = False):
    series_q = agg_series.asfreq(freq).sort_index().dropna()
    train = series_q.iloc[:-1]
    test = series_q.iloc[-1:]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    y_train = scaler.fit_transform(train.values.reshape(-1,1)).astype(np.float64)
    n_train = y_train.shape[0]
    if n_train <= 2:
        y_pred = float(train.iloc[-1])
        y_true = float(test.iloc[0])
    else:
        window = max(2, min(window, n_train-1))
        X_seq, y_seq = build_sequences(y_train, window)
        if X_seq.shape[0] == 0:
            y_pred = float(train.iloc[-1])
            y_true = float(test.iloc[0])
        else:
            model = rnn_train(X_seq, y_seq, hidden_size=hidden_size, epochs=epochs, lr=lr, clip=clip, seed=seed, verbose=verbose)
            last_seq = y_train[-window:]
            y_pred_norm = rnn_predict_next(model, last_seq.reshape(window,1))
            y_pred = scaler.inverse_transform([[y_pred_norm]])[0,0]
            y_true = float(test.iloc[0])
    mae = mean_absolute_error([y_true], [y_pred])
    rmse = mean_squared_error([y_true], [y_pred]) ** 0.5
    metrics = {'MAE': float(mae), 'RMSE': float(rmse)}
    if plot:
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(train.index, train.values, label='train', color='#1f77b4', lw=2)
        ax.plot(test.index, test.values, label='test', color='black', lw=2)
        ax.plot(test.index, [y_pred], 'o', color='#2ca02c', ms=6, label='RNN (NumPy) pred')
        ax.set_title('RNN NumPy (Elman) - pronóstico 1 paso')
        ax.set_xlabel('Periodo'); ax.set_ylabel('Puntaje promedio')
        ax.grid(True, alpha=0.9)
        ax.legend(frameon=False)
        for s in ['top','right']:
            ax.spines[s].set_visible(False)
        plt.show()
    return y_pred, y_true, metrics


def forecast_multi_step_numpy(agg_series: pd.Series, horizon: int = 5, freq='QS-MAR', window=4, hidden_size=16,
                              epochs=600, lr=0.01, clip=5.0, seed=42, plot=True, verbose: bool = False):
    """Pronostica varios pasos adelante usando la misma RNN Elman entrenada.

    Entrena sobre la parte de entrenamiento (todos menos el último punto real) y
    genera predicciones iterativas: cada nuevo paso usa la predicción previa como
    entrada extendida del último window.
    Métricas (MAE/RMSE) solo para el primer paso donde existe valor real.
    """
    if horizon < 1:
        raise ValueError("horizon debe ser >= 1")
    series_q = agg_series.asfreq(freq).sort_index().dropna()
    train = series_q.iloc[:-1]
    test = series_q.iloc[-1:]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    y_train = scaler.fit_transform(train.values.reshape(-1,1)).astype(np.float64)
    n_train = y_train.shape[0]
    if n_train <= 2:
        # Demasiado corta: repite último valor para todos los pasos
        base = float(train.iloc[-1])
        preds = [base for _ in range(horizon)]
        y_true_first = float(test.iloc[0])
    else:
        window = max(2, min(window, n_train-1))
        X_seq, y_seq = build_sequences(y_train, window)
        if X_seq.shape[0] == 0:
            base = float(train.iloc[-1])
            preds = [base for _ in range(horizon)]
            y_true_first = float(test.iloc[0])
        else:
            model = rnn_train(X_seq, y_seq, hidden_size=hidden_size, epochs=epochs, lr=lr, clip=clip, seed=seed, verbose=verbose)
            seq = y_train[-window:].copy()  # (window,1) flattened later
            preds = []
            for step in range(horizon):
                y_pred_norm = rnn_predict_next(model, seq.reshape(window,1))
                y_pred_val = scaler.inverse_transform([[y_pred_norm]])[0,0]
                preds.append(float(y_pred_val))
                # Actualiza la secuencia para siguiente paso (roll)
                y_pred_norm_arr = np.array([[y_pred_norm]])
                seq = np.vstack([seq[1:], y_pred_norm_arr])
            y_true_first = float(test.iloc[0])
    # Métricas primer paso
    mae = mean_absolute_error([y_true_first],[preds[0]])
    rmse = mean_squared_error([y_true_first],[preds[0]]) ** 0.5
    metrics = {'MAE_1step': float(mae), 'RMSE_1step': float(rmse)}
    # Construir serie de predicciones con índices futuros comenzando en test.index
    # El primer índice coincide con el punto test real
    start_idx = test.index[0]
    idx = pd.date_range(start=start_idx, periods=horizon, freq=freq)
    y_pred_series = pd.Series(preds, index=idx)
    if plot:
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(train.index, train.values, label='train', color='#1f77b4', lw=2)
        ax.plot(test.index, test.values, label='test real', color='black', lw=2)
        ax.plot(y_pred_series.index, y_pred_series.values, '--o', color='#2ca02c', ms=5, label='RNN pred')
        ax.set_title(f'RNN NumPy (Elman) - pronóstico {horizon} pasos')
        ax.set_xlabel('Periodo'); ax.set_ylabel('Puntaje promedio')
        ax.grid(True, alpha=0.9)
        ax.legend(frameon=False)
        for s in ['top','right']:
            ax.spines[s].set_visible(False)
        plt.show()
    return y_pred_series, y_true_first, metrics
