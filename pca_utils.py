# pca_utils.py

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go


def run_pca(df: pd.DataFrame, n_components: int = 3):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric data found in the uploaded file.")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    # Apply PCA
    n_components = min(n_components, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    singular_values = np.sqrt(pca.explained_variance_ * (X_scaled.shape[0] - 1))

    result = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "S": singular_values.tolist(),
        "V": pca.components_.tolist(),
        "principal_components": X_pca.tolist(),
        "column_names": numeric_df.columns.tolist()
    }
    return result


def create_pca_scatter_plot(pca_result, labels=None):
    pc = np.array(pca_result["principal_components"])
    fig = px.scatter(
        x=pc[:, 0], y=pc[:, 1],
        color=labels if labels is not None else None,
        labels={'x': 'PC1', 'y': 'PC2'},
        title="PCA Scatter Plot (PC1 vs PC2)"
    )
    return fig.to_json()


def create_pca_loading_plot(pca_result):
    components = np.array(pca_result["V"])
    column_names = pca_result["column_names"]

    fig = go.Figure()
    for i, comp in enumerate(components):
        fig.add_trace(go.Bar(
            x=column_names,
            y=comp,
            name=f"PC{i+1}"
        ))

    fig.update_layout(
        title="PCA Component Coefficients (Loadings)",
        barmode='group',
        xaxis_title="Original Features",
        yaxis_title="Contribution"
    )
    return fig.to_json()
