import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

def create_correlation_heatmap(df, figsize=(12, 8)):
    """
    Create an interactive correlation heatmap using Plotly
    
    Args:
        df: DataFrame containing the features
        figsize: Figure size (width, height)
    
    Returns:
        plotly.graph_objects.Figure: Interactive heatmap
    """
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        width=figsize[0] * 80,
        height=figsize[1] * 80,
        xaxis_title="Features",
        yaxis_title="Features"
    )
    
    return fig

def create_geographic_plot(df, zoom_level=5, map_style="open-street-map"):
    """
    Create an interactive geographic scatter plot with Shift+scroll wheel zoom
    
    Args:
        df: DataFrame containing latitude, longitude, and target values
        zoom_level: Initial zoom level for the map (default: 5)
        map_style: Map style to use (default: "open-street-map")
    
    Returns:
        plotly.graph_objects.Figure: Interactive map with Shift+scroll zoom
    """
    # Calculate center coordinates for California
    center_lat = df["Latitude"].mean()
    center_lon = df["Longitude"].mean()
    
    # Create the scatter mapbox plot
    fig = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        color="MedHouseVal",
        size="Population",
        hover_data=["MedInc", "HouseAge", "AveRooms"],
        color_continuous_scale="viridis",
        size_max=15,
        zoom=zoom_level,
        center={"lat": center_lat, "lon": center_lon},
        title="Geographic Distribution of Housing Prices in California"
    )
    
    # Configure layout with standard settings
    fig.update_layout(
        mapbox={
            'style': map_style,
            'center': {"lat": center_lat, "lon": center_lon},
            'zoom': zoom_level,
            'bearing': 0,
            'pitch': 0
        },
        height=600,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        showlegend=True,
        dragmode='pan'
    )
    
    # Add custom hover template
    fig.update_traces(
        hovertemplate="<b>Location:</b> (%{lat:.2f}, %{lon:.2f})<br>" +
                      "<b>House Value:</b> $%{color:.1f}K<br>" +
                      "<b>Population:</b> %{marker.size}<br>" +
                      "<b>Median Income:</b> $%{customdata[0]:.1f}K<br>" +
                      "<b>House Age:</b> %{customdata[1]:.0f} years<br>" +
                      "<b>Avg Rooms:</b> %{customdata[2]:.1f}<br>" +
                      "<extra></extra>"
    )
    
    return fig

def create_distribution_plots(df, features):
    """
    Create distribution plots for multiple features
    
    Args:
        df: DataFrame containing the features
        features: List of feature names to plot
    
    Returns:
        plotly.graph_objects.Figure: Subplot figure with distributions
    """
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=features,
        specs=[[{"secondary_y": False}] * n_cols for _ in range(n_rows)]
    )
    
    for i, feature in enumerate(features):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        fig.add_trace(
            go.Histogram(
                x=df[feature],
                name=feature,
                nbinsx=30,
                opacity=0.7,
                showlegend=False
            ),
            row=row,
            col=col
        )
    
    fig.update_layout(
        title="Distribution of Features",
        height=200 * n_rows,
        showlegend=False
    )
    
    return fig

def create_feature_importance_plot(importance_df):
    """
    Create feature importance plot
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
    
    Returns:
        plotly.graph_objects.Figure: Bar plot of feature importance
    """
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title="Feature Importance Rankings",
        color='importance',
        color_continuous_scale="viridis"
    )
    
    fig.update_layout(
        height=max(400, len(importance_df) * 30),
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Importance Score",
        yaxis_title="Features"
    )
    
    return fig

def create_prediction_vs_actual_plot(y_true, y_pred):
    """
    Create prediction vs actual values scatter plot
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        plotly.graph_objects.Figure: Scatter plot
    """
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        opacity=0.6,
        marker=dict(
            color='blue',
            size=4,
            line=dict(width=0.5, color='darkblue')
        )
    ))
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title="Predicted vs Actual House Values",
        xaxis_title="Actual Values ($100K)",
        yaxis_title="Predicted Values ($100K)",
        height=500,
        showlegend=True
    )
    
    return fig

def create_residuals_plot(y_true, y_pred):
    """
    Create residuals analysis plots
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        plotly.graph_objects.Figure: Subplot with residuals analysis
    """
    residuals = y_true - y_pred
    
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Residuals Distribution", "Residuals vs Predicted"],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Residuals histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            name="Residuals",
            opacity=0.7,
            marker_color='lightblue'
        ),
        row=1,
        col=1
    )
    
    # Residuals vs predicted scatter
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals vs Predicted',
            opacity=0.6,
            marker=dict(color='orange', size=4)
        ),
        row=1,
        col=2
    )
    
    # Add horizontal line at y=0 for residuals plot
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title="Residuals Analysis",
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Residuals ($100K)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Predicted Values ($100K)", row=1, col=2)
    fig.update_yaxes(title_text="Residuals ($100K)", row=1, col=2)
    
    return fig

def create_price_range_analysis(df, target_col):
    """
    Create price range analysis visualization
    
    Args:
        df: DataFrame containing the data
        target_col: Name of the target column
    
    Returns:
        plotly.graph_objects.Figure: Box plot by price ranges
    """
    # Create price ranges
    df_copy = df.copy()
    df_copy['PriceRange'] = pd.cut(
        df_copy[target_col],
        bins=5,
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    fig = px.box(
        df_copy,
        x='PriceRange',
        y=target_col,
        title="House Value Distribution by Price Range",
        color='PriceRange',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(
        xaxis_title="Price Range",
        yaxis_title="House Value ($100K)",
        height=500,
        showlegend=False
    )
    
    return fig
