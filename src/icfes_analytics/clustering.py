from __future__ import annotations

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, mixture
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph


def seleccionar_cols_puntaje(df):
    cols = df.columns
    candidatas = [
        'punt_global','punt_lectura_critica','punt_matematicas',
        'punt_c_naturales','punt_sociales_ciudadanas','punt_ingles',
        'mod_lectura_critica_punt','mod_razona_cuantitat_punt',
        'mod_ingles_punt','mod_competen_ciudada_punt',
    ]
    por_patron = [c for c in cols if 'punt' in c]
    out = []
    for c in candidatas + por_patron:
        if c not in out and c in cols:
            out.append(c)
    return out


def run_six_clustering_plots(X_scaled: np.ndarray, n_clusters=5, random_state=42, max_sample=10_000):
    rng = np.random.default_rng(random_state)
    idx_sample = rng.choice(X_scaled.shape[0], size=min(max_sample, X_scaled.shape[0]), replace=False)
    X_s = X_scaled[idx_sample]
    pca_s = PCA(n_components=2, random_state=random_state)
    X_2d_s = pca_s.fit_transform(X_s)
    xmin, xmax = X_2d_s[:,0].min(), X_2d_s[:,0].max()
    ymin, ymax = X_2d_s[:,1].min(), X_2d_s[:,1].max()
    px = 0.03 * (xmax - xmin) if xmax > xmin else 0.1
    py = 0.03 * (ymax - ymin) if ymax > ymin else 0.1
    xlim = (xmin - px, xmax + px)
    ylim = (ymin - py, ymax + py)

    def plot(ax, XY, labels, title):
        ax.scatter(XY[:, 0], XY[:, 1], c=labels, s=12, alpha=0.85, cmap='tab10')
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.margins(0)

    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
    axes = axes.ravel()
    resumen = {}
    pi = 0

    # 1) MiniBatchKMeans (sobre todo el conjunto; se grafica el subconjunto)
    km = cluster.MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)
    labels_full = km.fit_predict(X_scaled)
    resumen['MiniBatchKMeans'] = Counter(labels_full)
    plot(axes[pi], X_2d_s, labels_full[idx_sample], "MiniBatchKMeans"); pi += 1

    # Conectividad kNN para Ward/Avg (sobre el subconjunto)
    try:
        conn = kneighbors_graph(X_s, n_neighbors=5, include_self=False)
        conn = 0.5 * (conn + conn.T)
    except Exception:
        conn = None

    # 2) Ward
    ward = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', connectivity=conn)
    labels = ward.fit_predict(X_s)
    resumen['Ward'] = Counter(labels)
    plot(axes[pi], X_2d_s, labels, "Ward"); pi += 1

    # 3) Agglomerative Average (cityblock) con reducción opcional
    X_fit = X_s
    XY_plot = X_2d_s
    if X_fit.shape[0] > 1500:
        prelim = min(600, X_fit.shape[0] // 5)
        kmini_red = cluster.MiniBatchKMeans(n_clusters=prelim, random_state=random_state)
        X_fit = kmini_red.fit(X_fit).cluster_centers_
        XY_plot = pca_s.transform(X_fit)
    try:
        conn_avg = kneighbors_graph(X_fit, n_neighbors=5, include_self=False)
        conn_avg = 0.5 * (conn_avg + conn_avg.T)
    except Exception:
        conn_avg = None
    agg_avg = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='average', metric='cityblock', connectivity=conn_avg)
    labels = agg_avg.fit_predict(X_fit)
    resumen['AgglomerativeAvg'] = Counter(labels)
    plot(axes[pi], XY_plot, labels, "AgglomerativeAvg"); pi += 1

    # 4) DBSCAN
    db = cluster.DBSCAN(eps=1.2, min_samples=3)
    labels = db.fit_predict(X_s)
    resumen['DBSCAN'] = Counter(labels)
    plot(axes[pi], X_2d_s, labels, "DBSCAN"); pi += 1

    # 5) BIRCH
    birch = cluster.Birch(n_clusters=n_clusters)
    labels = birch.fit_predict(X_s)
    resumen['BIRCH'] = Counter(labels)
    plot(axes[pi], X_2d_s, labels, "BIRCH"); pi += 1

    # 6) GaussianMixture
    gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=random_state)
    labels = gmm.fit_predict(X_s)
    resumen['GaussianMixture'] = Counter(labels)
    plot(axes[pi], X_2d_s, labels, "GaussianMixture"); pi += 1

    plt.subplots_adjust(left=0.03, right=0.99, bottom=0.06, top=0.92, wspace=0.06, hspace=0.10)
    plt.show()

    return resumen
