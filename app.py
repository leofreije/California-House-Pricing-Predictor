import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Utility Imports
from model_utils import train_models, evaluate_model, get_feature_importance ## custom functions for model training and evaluation
from visualization_utils import create_correlation_heatmap, create_geographic_plot, create_distribution_plots ## imports functions from visualization_utils.py

# Configure page
st.set_page_config(page_title="California Housing Price Predictor",
                   page_icon="🏠",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    color: #2c3e50;
    border-bottom: 2px solid #1f77b4;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
/* Improved tooltip positioning and styling */
[data-testid="stTooltipHoverTarget"] {
    position: relative !important;
}
[data-baseweb="tooltip"] {
    max-width: 400px !important;
    width: auto !important;
    text-align: left !important;
    line-height: 1.4 !important;
    padding: 12px 16px !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    background-color: #2c3e50 !important;
    color: white !important;
    font-size: 14px !important;
    z-index: 9999 !important;
    position: fixed !important;
    transform: none !important;
}
/* Ensure tooltips don't overflow screen edges */
[data-baseweb="tooltip"][data-placement^="top"] {
    margin-bottom: 8px !important;
}
[data-baseweb="tooltip"][data-placement^="bottom"] {
    margin-top: 8px !important;
}
[data-baseweb="tooltip"][data-placement^="left"] {
    margin-right: 8px !important;
}
[data-baseweb="tooltip"][data-placement^="right"] {
    margin-left: 8px !important;
}
/* Tooltip arrow styling */
[data-baseweb="tooltip"] .css-1n76uvr {
    border-color: #2c3e50 transparent !important;
}
/* Better responsive behavior for tooltips */
@media (max-width: 768px) {
    [data-baseweb="tooltip"] {
        max-width: 300px !important;
        font-size: 13px !important;
        padding: 10px 14px !important;
    }
}
/* Prevent tooltip content from being cut off */
[data-baseweb="tooltip"] {
    word-wrap: break-word !important;
    white-space: normal !important;
    overflow-wrap: break-word !important;
}
/* Enhanced metric container styling for better tooltip positioning */
[data-testid="metric-container"] {
    position: relative !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}
[data-testid="metric-container"] [data-testid="stTooltipHoverTarget"] {
    margin-left: 4px !important;
    flex-shrink: 0 !important;
}
</style>

<script>
// Dynamic tooltip positioning to prevent off-screen issues
function adjustTooltipPositioning() {
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList') {
                const tooltips = document.querySelectorAll('[data-baseweb="tooltip"]');
                tooltips.forEach(function(tooltip) {
                    if (tooltip.style.position === 'fixed') {
                        const rect = tooltip.getBoundingClientRect();
                        const viewportWidth = window.innerWidth;
                        const viewportHeight = window.innerHeight;
                        
                        // Adjust horizontal position if tooltip goes off right edge
                        if (rect.right > viewportWidth - 10) {
                            tooltip.style.left = (viewportWidth - rect.width - 20) + 'px';
                        }
                        
                        // Adjust horizontal position if tooltip goes off left edge
                        if (rect.left < 10) {
                            tooltip.style.left = '20px';
                        }
                        
                        // Adjust vertical position if tooltip goes off bottom edge
                        if (rect.bottom > viewportHeight - 10) {
                            tooltip.style.top = (viewportHeight - rect.height - 20) + 'px';
                        }
                        
                        // Adjust vertical position if tooltip goes off top edge
                        if (rect.top < 10) {
                            tooltip.style.top = '20px';
                        }
                    }
                });
            }
        });
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
}

// Initialize when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', adjustTooltipPositioning);
} else {
    adjustTooltipPositioning();
}
</script>
""",
            unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load the California housing dataset"""
    try: # try to load the data, if it fails, return None
        california_housing = fetch_california_housing(as_frame=True) ## load the dataset as a pandas DataFrame
        df = california_housing.frame ## extract the DataFrame from the dataset
        feature_names = california_housing.feature_names ## extract the feature names from the dataset
        target_name = california_housing.target_names[0] ## extract the target name from the dataset

        # Add some derived features for better predictions
        df['rooms_per_household'] = df['AveRooms'] / df['AveOccup']
        df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']
        df['population_per_household'] = df['Population'] / df['AveOccup']

        return df, feature_names, target_name
    except Exception as e: # if it fails, print the error message
        st.error(f"Error loading data: {str(e)}")
        return None, None, None


def main():
     
    # Main title with icon
    st.markdown(
        '<h1 class="main-header">🏠 California Housing Price Predictor</h1>',
        unsafe_allow_html=True)

    st.markdown("""
    **Welcome to the California Housing Price Predictor!** 
    
    This application uses machine learning to predict median house values in California based on 1990 census data.
    Explore the data, understand the relationships between features, and make your own predictions! Hover over the question marks (?) for helpful tips and explanations.
    """)

    # Load data
    with st.spinner("Loading California housing data..."):
        df, feature_names, target_name = load_data()

    if df is None:
        st.error("Failed to load the dataset. Please refresh the page.")
        return

    # Set default page if not already set
    if "page" not in st.session_state:
        st.session_state.page = "📊 Dataset Overview"
    
    # Sidebar Navigation
    st.sidebar.title("🧭 Navigation")

    # Always visible buttons
    if st.sidebar.button("📊 Dataset Overview"):
        st.session_state.page = "📊 Dataset Overview"

    if st.sidebar.button("🔍 Exploratory Data Analysis"):
        st.session_state.page = "🔍 Exploratory Data Analysis"

    if st.sidebar.button("🤖 Model Training & Evaluation"):
        st.session_state.page = "🤖 Model Training & Evaluation"

    # Conditionally visible buttons (based on model readiness)
    if 'model_ready' not in st.session_state:
        st.session_state.model_ready = False
    
    if st.session_state.model_ready:
        if st.sidebar.button("🎯 Make Predictions"):
            st.session_state.page = "🎯 Make Predictions"
        if st.sidebar.button("📈 Model Insights"):
            st.session_state.page = "📈 Model Insights"
    else:
        # Show as grayed-out when not ready
        st.sidebar.markdown("""
        <style>
        .disabled-button {
            background-color: #f0f2f6;
            color: gray;
            padding: 0.5em 1em;
            border-radius: 0.25rem;
            text-align: center;
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 600;
            border: 1px solid #d3d3d3;
            cursor: not-allowed;
        }
        </style>
        <div class="disabled-button">🚫 Make Predictions (Train model first)</div>
        <div class="disabled-button">🚫 Model Insights (Train model first)</div>
        """, unsafe_allow_html=True)
        
    # Dataset Overview
    def show_dataset():
        st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>',
                    unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4) ## Create 4 columns for metrics
        with col1:
            st.metric("Total Samples", f"{len(df):,}")
        with col2:
            st.metric("Features", len(feature_names))
        with col3:
            st.metric("Avg. House Value", f"${df[target_name].mean():.0f}K")
        with col4:
            st.metric("Max House Value", f"${df[target_name].max():.0f}K")

        with st.expander("📋 Dataset Sample"):
            st.dataframe(df.head(10), use_container_width=True) ## Show first 10 rows of the dataset

        with st.expander("📈 Statistical Summary"):
            st.dataframe(df.describe(), use_container_width=True) ## Show statistical summary of the dataset

        st.subheader("🔍 Feature Descriptions")
        feature_descriptions = [
            ('MedInc', 'Median income in block group (in tens of thousands)'),
            ('HouseAge', 'Median house age in block group'),
            ('AveRooms', 'Average number of rooms per household'),
            ('AveBedrms', 'Average number of bedrooms per household'),
            ('Population', 'Block group population'),
            ('AveOccup', 'Average number of household members'),
            ('Latitude & Longitude',
             'Geographic coordinates (latitude and longitude) of the block group location'
             ),
            ('MedHouseVal',
             'Median house value (target variable, in hundreds of thousands)')
        ]

        ## Display feature descriptions in a numbered list
        for i, (feature, description) in enumerate(feature_descriptions, 1):
            if feature == 'Latitude & Longitude':
                if 'Latitude' in df.columns and 'Longitude' in df.columns:
                    st.write(f"**{i}. {feature}**: {description}")
            elif feature in df.columns:
                st.write(f"**{i}. {feature}**: {description}")

    # Exploratory Data Analysis
    def show_eda():
        st.markdown(
            '<h2 class="section-header">🔍 Exploratory Data Analysis</h2>',
            unsafe_allow_html=True)

        # Geographic distribution with zoom controls
        st.subheader("🗺️ Geographic Distribution of Housing Prices")

        # Initialize zoom level in session state
        if 'map_zoom' not in st.session_state:
            st.session_state.map_zoom = 5

        # Add map style and zoom controls
        col1, col2 = st.columns([2, 1])
        with col1:
            map_style = st.selectbox("🗺️ Map Style",
                                     options=[
                                         "open-street-map", "carto-positron",
                                         "carto-darkmatter"
                                     ],
                                     index=0,
                                     help="Choose map appearance style")
        with col2:
            st.markdown("**Zoom Controls:**")
            zoom_col1, zoom_col2, zoom_col3, zoom_col4, zoom_col5 = st.columns(
                [1, 1, 2, 1, 1])
            with zoom_col1:
                if st.button("➖", help="Zoom Out"):
                    st.session_state.map_zoom = max(
                        3, st.session_state.map_zoom - 1)
                    st.rerun()
            with zoom_col2:
                st.markdown("") ## spacer
            with zoom_col3:
                st.markdown(f"**Level {st.session_state.map_zoom}**",
                            help=f"Current zoom level (3=Wide, 10=Close)")
            with zoom_col4:
                st.markdown("") ## spacer
            with zoom_col5:
                if st.button("➕", help="Zoom In"):
                    st.session_state.map_zoom = min(
                        10, st.session_state.map_zoom + 1)
                    st.rerun()

            # Reset button on new line
            if st.button("🎯 Reset View", help="Reset to default zoom level"):
                st.session_state.map_zoom = 5
                st.rerun()

        # Create geographic plot with user-selected zoom and style
        geo_fig = create_geographic_plot(df,
                                         zoom_level=st.session_state.map_zoom,
                                         map_style=map_style)
        st.plotly_chart(geo_fig,
                        use_container_width=True,
                        key="california_map")

        # Show current map status
        st.info(
            f"**Current Map Style: {map_style}** | Zoom controls available above"
        )

        # Add helpful instructions for the intuitive controls
        st.success("**Map Navigation Guide:**")
        st.markdown("""
        - **➕➖ Zoom Buttons:** Click - to zoom out, + to zoom in (current level shown between buttons)
        - **🎯 Reset Button:** Click to return to default California view (Level 5)
        - **🖱️ Click & Drag:** Pan around California to explore different regions  
        - **📊 Hover:** Point cursor over any circle to see detailed housing information
        - **🎨 Style Selector:** Use dropdown above to change map appearance
        - **🖱️ Regular Scroll:** Use normal scroll to move up/down the webpage
        """)

        # Distribution plots
        with st.expander("📊 Feature Distributions"):
            dist_fig = create_distribution_plots(df,
                                                 feature_names + [target_name]) ## Create distribution plots for all features and target
            st.plotly_chart(dist_fig, use_container_width=True) ## Display distribution plots

        # Correlation analysis
        with st.expander("🔗 Feature Correlations"):
            corr_fig = create_correlation_heatmap(df) ## Create correlation heatmap
            st.plotly_chart(corr_fig, use_container_width=True) ## Display correlation heatmap

        # Interactive scatter plots
        st.subheader("🎯 Interactive Feature Analysis")
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("Select X-axis feature:",
                                     feature_names,
                                     index=0)
        with col2:
            y_feature = st.selectbox("Select Y-axis feature:",
                                     [target_name] + list(feature_names),
                                     index=0)

        scatter_fig = px.scatter(df,
                                 x=x_feature,
                                 y=y_feature,
                                 color=target_name,
                                 title=f"{y_feature} vs {x_feature}",
                                 color_continuous_scale="viridis",
                                 opacity=0.6)
        scatter_fig.update_layout(height=500)
        st.plotly_chart(scatter_fig, use_container_width=True) ## Display scatter plot

    # Model Training & Evaluation
    def show_training():
        st.markdown(
            '<h2 class="section-header">🤖 Model Training & Evaluation</h2>',
            unsafe_allow_html=True)

        # Model selection
        st.subheader("⚙️ Model Configuration")
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox(
                "Select Model Type:", ["Random Forest", "Linear Regression"],
                help=
                "Random Forest: Uses multiple decision trees to make predictions. Great for complex patterns but harder to interpret.\n\nLinear Regression: Creates a straight line relationship between features and prices. Simple and interpretable but may miss complex patterns."
            )
        with col2:
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)

        if model_type == "Random Forest":
            st.subheader("🌲 Random Forest Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
            with col2:
                max_depth = st.slider("Max Depth", 3, 20, 10)
            with col3:
                min_samples_split = st.slider("Min Samples Split", 2, 10, 2)

        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models..."):
                # Prepare data
                X = df[feature_names + [
                    'rooms_per_household', 'bedrooms_per_room',
                    'population_per_household'
                ]]
                y = df[target_name]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42)

                # Train models
                if model_type == "Random Forest":
                    model_params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'random_state': 42
                    }
                else:
                    model_params = {}

                model, scaler = train_models(X_train, y_train, model_type, model_params)

                # Make predictions
                if model_type == "Linear Regression":
                    X_test_scaled = scaler.transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                else:
                    y_pred = model.predict(X_test)

                # Metrics
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                # Store everything in session_state
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.model_type = model_type
                st.session_state.feature_cols = X.columns.tolist()
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.r2 = r2
                st.session_state.mae = mae
                st.session_state.rmse = rmse
                st.session_state.model_ready = True # flag for model readiness
                st.session_state.show_training_success = True  # flag for showing training success message

            st.rerun()  # trigger clean UI rerun
        
        if st.session_state.get('show_training_success', False): # show training results if flag is set
            st.success("✅ Model training completed!")

            st.subheader("📊 Model Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "R² Score",
                    f"{st.session_state.r2:.4f}",
                    help="R² Score (R-squared): Measures how well the model explains the variation in house prices. "
                         "Values range from 0 to 1, where 1 means perfect predictions and 0 means the model is no better "
                         "than just guessing the average price."
                )
            with col2:
                st.metric(
                    "MAE",
                    f"${st.session_state.mae:.2f}K",
                    help="MAE (Mean Absolute Error): The average difference between predicted and actual house prices. "
                         "Lower values are better."
                )
            with col3:
                st.metric(
                    "RMSE",
                    f"${st.session_state.rmse:.2f}K",
                    help="RMSE (Root Mean Square Error): Similar to MAE but gives more penalty to large prediction errors. "
                         "Lower values are better. RMSE is always equal to or higher than MAE."
                )

            st.subheader("🎯 Predictions vs Actual Values")
            pred_fig = go.Figure()
            pred_fig.add_trace(
                go.Scatter(x=st.session_state.y_test,
                           y=st.session_state.y_pred,
                           mode='markers',
                           name='Predictions',
                           opacity=0.6,
                           marker=dict(color='blue', size=4)))
            pred_fig.add_trace(
                go.Scatter(x=[st.session_state.y_test.min(), st.session_state.y_test.max()],
                           y=[st.session_state.y_test.min(), st.session_state.y_test.max()],
                           mode='lines',
                           name='Perfect Prediction',
                           line=dict(color='red', dash='dash')))
            pred_fig.update_layout(title="Predicted vs Actual House Values",
                                   xaxis_title="Actual Values ($100K)",
                                   yaxis_title="Predicted Values ($100K)",
                                   height=500)
            st.plotly_chart(pred_fig, use_container_width=True)

            # Clear flag so it doesn't show again unless retrained
            st.session_state.show_training_success = False

    # Make Predictions
    def show_predictions():
        st.markdown(
            '<h2 class="section-header">🎯 Make Your Own Predictions</h2>',
            unsafe_allow_html=True)

        if 'model' not in st.session_state:
            st.warning(
                "⚠️ Please train a model first in the 'Model Training & Evaluation' section."
            )
            return

        st.markdown(
            "**Adjust the sliders below to predict house prices for different scenarios:**"
        )

        # Input widgets
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏡 Property Features")
            med_inc = st.slider("Median Income (tens of thousands)",
                                float(df['MedInc'].min()),
                                float(df['MedInc'].max()),
                                float(df['MedInc'].mean()))
            house_age = st.slider("House Age (years)",
                                  int(df['HouseAge'].min()),
                                  int(df['HouseAge'].max()),
                                  int(df['HouseAge'].mean()))
            ave_rooms = st.slider("Average Rooms per Household",
                                  float(df['AveRooms'].min()),
                                  float(df['AveRooms'].max()),
                                  float(df['AveRooms'].mean()))
            ave_bedrms = st.slider("Average Bedrooms per Household",
                                   float(df['AveBedrms'].min()),
                                   float(df['AveBedrms'].max()),
                                   float(df['AveBedrms'].mean()))

        with col2:
            st.subheader("📍 Location & Demographics")
            population = st.slider("Population", int(df['Population'].min()),
                                   int(df['Population'].max()),
                                   int(df['Population'].mean()))
            ave_occup = st.slider("Average Occupancy",
                                  float(df['AveOccup'].min()),
                                  float(df['AveOccup'].max()),
                                  float(df['AveOccup'].mean()))

            latitude = st.slider("Latitude", float(df['Latitude'].min()),
                                 float(df['Latitude'].max()),
                                 float(df['Latitude'].mean()))
            longitude = st.slider("Longitude", float(df['Longitude'].min()),
                                  float(df['Longitude'].max()),
                                  float(df['Longitude'].mean()))

        # Calculate derived features
        rooms_per_household = ave_rooms / ave_occup if ave_occup > 0 else 0
        bedrooms_per_room = ave_bedrms / ave_rooms if ave_rooms > 0 else 0
        population_per_household = population / ave_occup if ave_occup > 0 else 0

        # Create input array
        input_data = np.array([[
            med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup,
            latitude, longitude, rooms_per_household, bedrooms_per_room,
            population_per_household
        ]])

        # Make prediction
        model = st.session_state.model
        scaler = st.session_state.scaler
        model_type = st.session_state.model_type

        if model_type == "Linear Regression":
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
        else:
            prediction = model.predict(input_data)[0]

        # Display prediction
        st.subheader("🎯 Price Prediction")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Predicted Price",
                f"${prediction:.2f}K",
                help=
                "Estimated median house value in hundreds of thousands of dollars, calculated by the trained model using your     location and housing characteristics."
            )
        with col2:
            st.metric("Price Range",
                      f"${prediction*0.9:.0f}K - ${prediction*1.1:.0f}K",
                      help="Estimated ±10% confidence interval")
        with col3:
            if prediction < df[target_name].quantile(0.33):
                price_category = "💰 Affordable"
            elif prediction < df[target_name].quantile(0.67):
                price_category = "💵 Moderate"
            else:
                price_category = "💎 Premium"
            st.metric(
                "Price Category",
                price_category,
                help=
                "Houses are grouped into three categories: Affordable (bottom 33%), Moderate (middle 33%), and Premium (top 33%) based on actual California housing prices."
            )

        # Show input summary
        st.subheader("📋 Input Summary")
        input_summary = pd.DataFrame({
            'Feature': [
                'Median Income', 'House Age', 'Avg Rooms', 'Avg Bedrooms',
                'Population', 'Avg Occupancy', 'Latitude', 'Longitude'
            ],
            'Your Input': [
                f"${med_inc*10:.0f}K", f"{house_age} years",
                f"{ave_rooms:.1f}", f"{ave_bedrms:.1f}", f"{population:,}",
                f"{ave_occup:.1f}", f"{latitude:.2f}", f"{longitude:.2f}"
            ],
            'Dataset Average': [
                f"${df['MedInc'].mean()*10:.0f}K",
                f"{df['HouseAge'].mean():.0f} years",
                f"{df['AveRooms'].mean():.1f}",
                f"{df['AveBedrms'].mean():.1f}",
                f"{df['Population'].mean():.0f}",
                f"{df['AveOccup'].mean():.1f}", f"{df['Latitude'].mean():.2f}",
                f"{df['Longitude'].mean():.2f}"
            ]
        })
        st.dataframe(input_summary, use_container_width=True)

    # Model Insights
    def show_insights():
        st.markdown('<h2 class="section-header">📈 Model Insights</h2>',
                    unsafe_allow_html=True)

        if all(key in st.session_state for key in ['r2', 'mae', 'rmse']):
            st.subheader("📊 Model Performance Recap")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{st.session_state.r2:.4f}")
            with col2:
                st.metric("MAE", f"${st.session_state.mae:.2f}K")
            with col3:
                st.metric("RMSE", f"${st.session_state.rmse:.2f}K")
        else:
            st.warning("⚠️ No model metrics available. Please train a model in the '🤖 Model Training & Evaluation' section.")
        
        model = st.session_state.model
        model_type = st.session_state.model_type
        feature_cols = st.session_state.feature_cols

        # Feature importance (for Random Forest)
        if model_type == "Random Forest":
            st.subheader("🔍 Feature Importance Analysis")
            importance_df = get_feature_importance(model, feature_cols)

            # Feature importance bar chart
            importance_fig = px.bar(importance_df,
                                    x='importance',
                                    y='feature',
                                    orientation='h',
                                    title="Feature Importance Rankings",
                                    color='importance',
                                    color_continuous_scale="viridis")
            importance_fig.update_layout(
                height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(importance_fig, use_container_width=True)

            st.subheader("🏆 Top 5 Most Important Features")
            top_features = importance_df.head()
            for idx, row in top_features.iterrows():
                st.write(
                    f"**{idx+1}. {row['feature']}**: {row['importance']:.3f}")

        # Model performance insights
        if all(key in st.session_state
               for key in ['X_test', 'y_test', 'y_pred']):
            st.subheader("📊 Model Performance Analysis")

            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred

            # Residuals analysis
            st.info(
                "📝 **What are Residuals?** Residuals are the differences between actual and predicted house prices. Good models should have residuals clustered around zero with no clear patterns."
            )
            residuals = y_test - y_pred

            col1, col2 = st.columns(2)

            with col1:
                # Residuals histogram
                residual_fig = px.histogram(x=residuals,
                                            nbins=30,
                                            title="Distribution of Residuals",
                                            labels={
                                                'x': 'Residuals ($100K)',
                                                'y': 'Frequency'
                                            })
                st.plotly_chart(residual_fig, use_container_width=True)

            with col2:
                # Residuals vs predicted
                residual_scatter = px.scatter(
                    x=y_pred,
                    y=residuals,
                    title="Residuals vs Predicted Values",
                    labels={
                        'x': 'Predicted Values ($100K)',
                        'y': 'Residuals ($100K)'
                    },
                    opacity=0.6)
                residual_scatter.add_hline(y=0,
                                           line_dash="dash",
                                           line_color="red")
                st.plotly_chart(residual_scatter, use_container_width=True)

            # Performance by price range
            st.subheader("💰 Performance by Price Range")
            st.info(
                "📝 **Understanding the table below:** MAE shows average prediction error in thousands of dollars for each price range. R² shows how well the model performs for that price range (closer to 1.0 is better)."
            )

            price_ranges = pd.cut(
                y_test,
                bins=5,
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            performance_by_range = pd.DataFrame({
                'Price Range':
                price_ranges.cat.categories,
                'Count': [
                    sum(price_ranges == cat)
                    for cat in price_ranges.cat.categories
                ],
                'MAE': [
                    mean_absolute_error(y_test[price_ranges == cat],
                                        y_pred[price_ranges == cat])
                    for cat in price_ranges.cat.categories
                ],
                'R²': [
                    r2_score(y_test[price_ranges == cat],
                             y_pred[price_ranges == cat])
                    for cat in price_ranges.cat.categories
                ]
            })

            st.dataframe(performance_by_range, use_container_width=True)

        # Model comparison recommendation
        st.subheader("💡 Model Insights & Recommendations")
        if model_type == "Random Forest":
            st.info("""
            **Random Forest Model Insights:**
            - ✅ Handles non-linear relationships well
            - ✅ Provides feature importance rankings
            - ✅ Robust to outliers
            - ⚠️ May overfit with small datasets
            - 💡 Consider tuning hyperparameters for better performance
            """)
        else:
            st.info("""
            **Linear Regression Model Insights:**
            - ✅ Simple and interpretable
            - ✅ Fast training and prediction
            - ✅ Good baseline model
            - ⚠️ Assumes linear relationships
            - 💡 Consider feature engineering for better performance
            """)
    
    #shows selected page when button is clicked
    page = st.session_state.page

    if page == "📊 Dataset Overview":
        show_dataset()
    elif page == "🔍 Exploratory Data Analysis":
        show_eda()
    elif page == "🤖 Model Training & Evaluation":
        show_training()
    elif page == "📈 Model Insights":
        show_insights()
    elif page == "🎯 Make Predictions":
        show_predictions()

if __name__ == "__main__":
    main()
