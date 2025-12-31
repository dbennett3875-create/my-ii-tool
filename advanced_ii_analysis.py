"""
Advanced I&I Analysis Tool
A comprehensive Streamlit application for Infiltration & Inflow analysis
Designed to be intuitive for novices yet powerful for advanced users
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats, signal
from sklearn.ensemble import IsolationForest
import io
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="I&I Analysis Tool",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def convert_units(value: float, from_unit: str, to_unit: str, unit_type: str = 'flow') -> float:
    """Convert between different units"""
    if unit_type == 'flow':
        conversions = {
            ('gpm', 'mgd'): lambda x: x * 0.00144,
            ('mgd', 'gpm'): lambda x: x / 0.00144,
            ('gpm', 'cfs'): lambda x: x * 0.002228,
            ('cfs', 'gpm'): lambda x: x / 0.002228,
            ('mgd', 'cfs'): lambda x: x * 1.547,
            ('cfs', 'mgd'): lambda x: x / 1.547,
        }
    elif unit_type == 'rain':
        conversions = {
            ('in', 'mm'): lambda x: x * 25.4,
            ('mm', 'in'): lambda x: x / 25.4,
            ('in', 'cm'): lambda x: x * 2.54,
            ('cm', 'in'): lambda x: x / 2.54,
        }

    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        return conversions[key](value)
    return value


def validate_dataframe(df: pd.DataFrame, required_cols: List[str], df_name: str) -> Tuple[bool, str]:
    """Validate uploaded dataframe structure"""
    if df is None or df.empty:
        return False, f"{df_name} is empty"

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"{df_name} missing columns: {', '.join(missing_cols)}"

    return True, "Valid"


def parse_timestamp(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """Parse and validate timestamp column"""
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error parsing timestamps: {e}")
        return None


def detect_outliers(series: pd.Series, method: str = 'iqr', threshold: float = 3.0) -> pd.Series:
    """Detect outliers in a data series"""
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        return (series < lower) | (series > upper)
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(series.fillna(series.mean())))
        return z_scores > threshold
    elif method == 'isolation_forest':
        model = IsolationForest(contamination=0.1, random_state=42)
        predictions = model.fit_predict(series.values.reshape(-1, 1))
        return predictions == -1
    return pd.Series([False] * len(series))


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class DataLoader:
    """Handle data loading and preprocessing"""

    @staticmethod
    def load_file(uploaded_file) -> pd.DataFrame:
        """Load CSV, Excel, or JSON file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file format. Use CSV, Excel, or JSON.")
                return None

            # Check file size (200MB limit)
            if uploaded_file.size > 200 * 1024 * 1024:
                st.warning("File exceeds 200MB. Processing may be slow.")

            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None

    @staticmethod
    def preprocess_flow_data(df: pd.DataFrame, flow_col: str, timestamp_col: str = 'timestamp',
                            basin_col: Optional[str] = None) -> pd.DataFrame:
        """Preprocess flow data"""
        df = parse_timestamp(df, timestamp_col)
        if df is None:
            return None

        # Handle missing values
        if df[flow_col].isnull().any():
            n_missing = df[flow_col].isnull().sum()
            st.warning(f"Found {n_missing} missing flow values. Interpolating...")
            df[flow_col] = df[flow_col].interpolate(method='time')

        return df

    @staticmethod
    def preprocess_rain_data(df: pd.DataFrame, rain_col: str, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Preprocess rainfall data"""
        df = parse_timestamp(df, timestamp_col)
        if df is None:
            return None

        # Fill missing rainfall with 0
        df[rain_col] = df[rain_col].fillna(0)

        return df


# ============================================================================
# ANALYSIS ENGINE
# ============================================================================

class IIAnalyzer:
    """Core I&I analysis engine"""

    def __init__(self, flow_df: pd.DataFrame, rain_df: pd.DataFrame,
                 asset_df: Optional[pd.DataFrame] = None):
        self.flow_df = flow_df
        self.rain_df = rain_df
        self.asset_df = asset_df
        self.results = {}

    def detect_rain_events(self, rain_threshold: float = 0.1,
                          min_gap_hours: int = 6) -> List[Dict]:
        """Detect rain events from rainfall data"""
        rain_series = self.rain_df.set_index('timestamp')['rainfall_in']

        # Find periods with rain above threshold
        rain_mask = rain_series > rain_threshold

        events = []
        in_event = False
        event_start = None
        event_rain = 0

        for idx, value in rain_series.items():
            if value > rain_threshold:
                if not in_event:
                    event_start = idx
                    in_event = True
                    event_rain = value
                else:
                    event_rain += value
            else:
                if in_event:
                    # Check if gap is long enough to end event
                    if event_start and (idx - event_start).total_seconds() / 3600 > min_gap_hours:
                        events.append({
                            'start': event_start,
                            'end': idx,
                            'total_rainfall': event_rain,
                            'duration_hours': (idx - event_start).total_seconds() / 3600
                        })
                        in_event = False
                        event_rain = 0

        return events

    def calculate_baseline_flow(self, method: str = 'percentile',
                                percentile: float = 10) -> float:
        """Calculate baseline (dry weather) flow"""
        if method == 'percentile':
            return self.flow_df['flow_rate_gpm'].quantile(percentile / 100)
        elif method == 'minimum':
            return self.flow_df['flow_rate_gpm'].min()
        elif method == 'moving_minimum':
            return self.flow_df['flow_rate_gpm'].rolling(window=24*7, min_periods=1).min().median()
        return self.flow_df['flow_rate_gpm'].quantile(0.1)

    def calculate_ii_volume(self, events: List[Dict], baseline_flow: float) -> Dict:
        """Calculate I&I volume for each event"""
        flow_series = self.flow_df.set_index('timestamp')['flow_rate_gpm']

        event_results = []
        for event in events:
            # Get flow during event
            event_flow = flow_series[event['start']:event['end']]

            # Calculate excess flow (I&I)
            excess_flow = event_flow - baseline_flow
            excess_flow = excess_flow[excess_flow > 0]

            # Calculate volume (convert GPM to gallons)
            ii_volume_gallons = excess_flow.sum() * (len(excess_flow) / len(flow_series) * 24 * 60)
            peak_flow = event_flow.max()

            event_results.append({
                **event,
                'baseline_flow_gpm': baseline_flow,
                'peak_flow_gpm': peak_flow,
                'ii_volume_gallons': ii_volume_gallons,
                'ii_volume_mgal': ii_volume_gallons / 1_000_000,
                'peak_to_baseline_ratio': peak_flow / baseline_flow if baseline_flow > 0 else 0
            })

        return event_results

    def hydrograph_separation(self, alpha: float = 0.925) -> pd.DataFrame:
        """Separate baseflow from total flow using recursive digital filter"""
        flow = self.flow_df['flow_rate_gpm'].values
        baseflow = np.zeros_like(flow)
        baseflow[0] = flow[0]

        for i in range(1, len(flow)):
            baseflow[i] = ((alpha * baseflow[i-1]) + ((1 - alpha) / 2) * (flow[i] + flow[i-1]))
            baseflow[i] = min(baseflow[i], flow[i])

        result_df = self.flow_df.copy()
        result_df['baseflow_gpm'] = baseflow
        result_df['quickflow_gpm'] = flow - baseflow

        return result_df

    def calculate_rdii_coefficients(self, event_results: List[Dict]) -> Dict:
        """Calculate RDII (Rainfall Dependent Infiltration/Inflow) coefficients"""
        if not event_results:
            return {}

        total_rainfall = sum(e['total_rainfall'] for e in event_results)
        total_ii_volume = sum(e['ii_volume_gallons'] for e in event_results)

        # R-value: ratio of I&I volume to rainfall volume
        # Assuming drainage area (placeholder - should come from asset data)
        drainage_area_acres = 100  # Default
        if self.asset_df is not None and 'area_acres' in self.asset_df.columns:
            drainage_area_acres = self.asset_df['area_acres'].sum()

        rainfall_volume_gallons = total_rainfall * drainage_area_acres * 43560 / 12  # Convert to gallons

        r_value = total_ii_volume / rainfall_volume_gallons if rainfall_volume_gallons > 0 else 0

        return {
            'r_value': r_value,
            'total_rainfall_in': total_rainfall,
            'total_ii_volume_mgal': total_ii_volume / 1_000_000,
            'drainage_area_acres': drainage_area_acres,
            'unit_ii_gal_per_inch_acre': total_ii_volume / (total_rainfall * drainage_area_acres) if total_rainfall > 0 else 0
        }

    def basin_prioritization(self, event_results: List[Dict]) -> pd.DataFrame:
        """Prioritize basins for I&I reduction"""
        if 'basin_id' not in self.flow_df.columns:
            return pd.DataFrame()

        basins = self.flow_df['basin_id'].unique()
        basin_scores = []

        for basin in basins:
            basin_flow = self.flow_df[self.flow_df['basin_id'] == basin]

            # Calculate metrics
            total_ii = basin_flow['flow_rate_gpm'].sum()
            peak_flow = basin_flow['flow_rate_gpm'].max()

            # Get asset data if available
            pipe_age = 0
            pipe_length = 0
            if self.asset_df is not None:
                basin_assets = self.asset_df[self.asset_df['basin_id'] == basin]
                if not basin_assets.empty:
                    pipe_age = basin_assets['pipe_age_years'].mean()
                    pipe_length = basin_assets['pipe_length_miles'].sum()

            # Multi-criteria score (normalized 0-100)
            volume_score = min(total_ii / 1000, 100)
            age_score = min(pipe_age, 100)
            length_score = min(pipe_length * 10, 100)

            composite_score = (volume_score * 0.5 + age_score * 0.3 + length_score * 0.2)

            basin_scores.append({
                'Basin': basin,
                'Total I&I (MGal)': total_ii / 1_000_000,
                'Peak Flow (GPM)': peak_flow,
                'Pipe Age (years)': pipe_age,
                'Pipe Length (mi)': pipe_length,
                'Priority Score': composite_score
            })

        df = pd.DataFrame(basin_scores)
        df = df.sort_values('Priority Score', ascending=False)
        df['Rank'] = range(1, len(df) + 1)

        return df

    def detect_anomalies(self, method: str = 'isolation_forest') -> pd.DataFrame:
        """Detect anomalies in flow data"""
        result_df = self.flow_df.copy()
        outliers = detect_outliers(self.flow_df['flow_rate_gpm'], method=method)
        result_df['is_anomaly'] = outliers
        return result_df


# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Create interactive visualizations"""

    @staticmethod
    def plot_hydrograph(flow_df: pd.DataFrame, rain_df: pd.DataFrame,
                       events: List[Dict] = None, show_baseflow: bool = False) -> go.Figure:
        """Create interactive hydrograph with rainfall"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.3, 0.7],
            subplot_titles=('Rainfall', 'Flow Rate')
        )

        # Rainfall bar chart
        fig.add_trace(
            go.Bar(
                x=rain_df['timestamp'],
                y=rain_df['rainfall_in'],
                name='Rainfall',
                marker_color='#1f77b4',
                opacity=0.6
            ),
            row=1, col=1
        )

        # Flow line chart
        fig.add_trace(
            go.Scatter(
                x=flow_df['timestamp'],
                y=flow_df['flow_rate_gpm'],
                name='Total Flow',
                line=dict(color='#2ca02c', width=2)
            ),
            row=2, col=1
        )

        # Add baseflow if available
        if show_baseflow and 'baseflow_gpm' in flow_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=flow_df['timestamp'],
                    y=flow_df['baseflow_gpm'],
                    name='Baseflow',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                ),
                row=2, col=1
            )

        # Highlight rain events
        if events:
            for event in events:
                fig.add_vrect(
                    x0=event['start'],
                    x1=event['end'],
                    fillcolor='lightblue',
                    opacity=0.2,
                    layer='below',
                    line_width=0,
                    row=2, col=1
                )

        fig.update_xaxes(title_text="Date/Time", row=2, col=1)
        fig.update_yaxes(title_text="Rainfall (in)", row=1, col=1)
        fig.update_yaxes(title_text="Flow Rate (GPM)", row=2, col=1)

        fig.update_layout(
            height=700,
            hovermode='x unified',
            showlegend=True,
            title_text="I&I Analysis - Hydrograph"
        )

        return fig

    @staticmethod
    def plot_event_summary(event_results: List[Dict]) -> go.Figure:
        """Create summary visualization of rain events"""
        if not event_results:
            return go.Figure()

        df = pd.DataFrame(event_results)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('I&I Volume by Event', 'Peak Flow vs Rainfall'),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}]]
        )

        # Bar chart of I&I volumes
        fig.add_trace(
            go.Bar(
                x=[f"Event {i+1}" for i in range(len(df))],
                y=df['ii_volume_mgal'],
                name='I&I Volume (MGal)',
                marker_color='#d62728'
            ),
            row=1, col=1
        )

        # Scatter plot of peak flow vs rainfall
        fig.add_trace(
            go.Scatter(
                x=df['total_rainfall'],
                y=df['peak_flow_gpm'],
                mode='markers',
                name='Events',
                marker=dict(size=10, color=df['ii_volume_mgal'], colorscale='Viridis',
                           showscale=True, colorbar=dict(title="I&I (MGal)"))
            ),
            row=1, col=2
        )

        fig.update_xaxes(title_text="Event", row=1, col=1)
        fig.update_xaxes(title_text="Rainfall (in)", row=1, col=2)
        fig.update_yaxes(title_text="I&I Volume (MGal)", row=1, col=1)
        fig.update_yaxes(title_text="Peak Flow (GPM)", row=1, col=2)

        fig.update_layout(height=400, showlegend=False)

        return fig

    @staticmethod
    def plot_basin_comparison(basin_df: pd.DataFrame) -> go.Figure:
        """Create basin comparison visualization"""
        if basin_df.empty:
            return go.Figure()

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=basin_df['Basin'],
            y=basin_df['Priority Score'],
            text=basin_df['Rank'],
            textposition='outside',
            texttemplate='#%{text}',
            marker_color=basin_df['Priority Score'],
            marker_colorscale='Reds',
            showscale=False
        ))

        fig.update_layout(
            title="Basin Prioritization",
            xaxis_title="Basin",
            yaxis_title="Priority Score",
            height=400
        )

        return fig


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<p class="main-header">💧 I&I Analysis Tool</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Infiltration & Inflow Analysis for Wastewater Systems</p>', unsafe_allow_html=True)

    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Mode selection
        analysis_mode = st.radio(
            "Analysis Mode",
            ["🎯 Basic (Recommended)", "🔧 Advanced"],
            help="Basic mode uses sensible defaults. Advanced mode provides full control."
        )

        advanced_mode = analysis_mode == "🔧 Advanced"

        st.markdown("---")

        # Advanced settings
        if advanced_mode:
            with st.expander("🔍 Detection Settings"):
                rain_threshold = st.slider(
                    "Rain Event Threshold (inches)",
                    0.01, 0.5, 0.1, 0.01,
                    help="Minimum rainfall to consider as an event"
                )
                min_gap_hours = st.slider(
                    "Minimum Gap Between Events (hours)",
                    1, 24, 6,
                    help="Minimum dry period to separate events"
                )
                baseline_method = st.selectbox(
                    "Baseline Flow Method",
                    ["percentile", "minimum", "moving_minimum"],
                    help="Method to calculate dry weather flow"
                )
                baseline_percentile = st.slider(
                    "Baseline Percentile",
                    5, 25, 10,
                    help="Percentile for baseline calculation"
                )
        else:
            rain_threshold = 0.1
            min_gap_hours = 6
            baseline_method = "percentile"
            baseline_percentile = 10

        with st.expander("📊 Analysis Features"):
            perform_hydrograph_sep = st.checkbox(
                "Hydrograph Separation",
                value=advanced_mode,
                help="Separate baseflow from quickflow"
            )
            perform_rdii = st.checkbox(
                "Calculate RDII Coefficients",
                value=True,
                help="Calculate rainfall-dependent I&I metrics"
            )
            perform_anomaly = st.checkbox(
                "Anomaly Detection",
                value=advanced_mode,
                help="Detect unusual flow patterns"
            )
            if perform_anomaly:
                anomaly_method = st.selectbox(
                    "Anomaly Method",
                    ["isolation_forest", "iqr", "zscore"]
                )
            else:
                anomaly_method = "isolation_forest"

        with st.expander("📈 Visualization"):
            show_baseflow = st.checkbox("Show Baseflow", value=perform_hydrograph_sep)
            show_events = st.checkbox("Highlight Rain Events", value=True)

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Data Upload", "📊 Analysis", "📈 Results", "📄 Report"])

    with tab1:
        st.header("Upload Your Data")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Flow Data (Required)")
            st.info("💡 CSV/Excel with columns: timestamp, flow_rate_gpm")
            flow_file = st.file_uploader(
                "Upload Flow Data",
                type=['csv', 'xlsx', 'xls', 'json'],
                key='flow',
                help="Timestamp and flow rate data"
            )

            if flow_file:
                flow_df = DataLoader.load_file(flow_file)
                if flow_df is not None:
                    st.success(f"✅ Loaded {len(flow_df)} records")

                    # Auto-detect columns
                    flow_col = st.selectbox(
                        "Flow Rate Column",
                        flow_df.columns,
                        index=list(flow_df.columns).index('flow_rate_gpm') if 'flow_rate_gpm' in flow_df.columns else 0
                    )
                    time_col = st.selectbox(
                        "Timestamp Column",
                        flow_df.columns,
                        index=list(flow_df.columns).index('timestamp') if 'timestamp' in flow_df.columns else 0
                    )

                    # Preview
                    with st.expander("Preview Data"):
                        st.dataframe(flow_df.head())

        with col2:
            st.subheader("Rainfall Data (Required)")
            st.info("💡 CSV/Excel with columns: timestamp, rainfall_in")
            rain_file = st.file_uploader(
                "Upload Rainfall Data",
                type=['csv', 'xlsx', 'xls', 'json'],
                key='rain',
                help="Timestamp and rainfall data"
            )

            if rain_file:
                rain_df = DataLoader.load_file(rain_file)
                if rain_df is not None:
                    st.success(f"✅ Loaded {len(rain_df)} records")

                    rain_col = st.selectbox(
                        "Rainfall Column",
                        rain_df.columns,
                        index=list(rain_df.columns).index('rainfall_in') if 'rainfall_in' in rain_df.columns else 0
                    )

                    with st.expander("Preview Data"):
                        st.dataframe(rain_df.head())

        with col3:
            st.subheader("Asset Data (Optional)")
            st.info("💡 CSV/Excel with: basin_id, area_acres, pipe_age_years, etc.")
            asset_file = st.file_uploader(
                "Upload Asset Data",
                type=['csv', 'xlsx', 'xls', 'json'],
                key='asset',
                help="Basin/pipe characteristics"
            )

            if asset_file:
                asset_df = DataLoader.load_file(asset_file)
                if asset_df is not None:
                    st.success(f"✅ Loaded {len(asset_df)} records")

                    with st.expander("Preview Data"):
                        st.dataframe(asset_df.head())
            else:
                asset_df = None

    with tab2:
        st.header("Run Analysis")

        if flow_file and rain_file:
            # Preprocess data
            flow_df_processed = DataLoader.preprocess_flow_data(
                flow_df.copy(), flow_col, time_col,
                'basin_id' if 'basin_id' in flow_df.columns else None
            )
            rain_df_processed = DataLoader.preprocess_rain_data(
                rain_df.copy(), rain_col, time_col
            )

            if flow_df_processed is not None and rain_df_processed is not None:
                # Rename columns to standard names
                flow_df_processed = flow_df_processed.rename(columns={flow_col: 'flow_rate_gpm', time_col: 'timestamp'})
                rain_df_processed = rain_df_processed.rename(columns={rain_col: 'rainfall_in', time_col: 'timestamp'})

                # Run analysis button
                if st.button("▶️ Run I&I Analysis", type="primary"):
                    with st.spinner("Analyzing data..."):
                        # Initialize analyzer
                        analyzer = IIAnalyzer(flow_df_processed, rain_df_processed, asset_df)

                        # Detect rain events
                        events = analyzer.detect_rain_events(rain_threshold, min_gap_hours)
                        st.session_state.events = events

                        # Calculate baseline
                        baseline = analyzer.calculate_baseline_flow(baseline_method, baseline_percentile)
                        st.session_state.baseline = baseline

                        # Calculate I&I volumes
                        event_results = analyzer.calculate_ii_volume(events, baseline)
                        st.session_state.event_results = event_results

                        # Additional analyses
                        if perform_hydrograph_sep:
                            sep_df = analyzer.hydrograph_separation()
                            st.session_state.sep_df = sep_df
                            flow_df_processed = sep_df  # Update for visualization

                        if perform_rdii:
                            rdii = analyzer.calculate_rdii_coefficients(event_results)
                            st.session_state.rdii = rdii

                        if perform_anomaly:
                            anomaly_df = analyzer.detect_anomalies(anomaly_method)
                            st.session_state.anomaly_df = anomaly_df

                        # Basin prioritization
                        if 'basin_id' in flow_df_processed.columns:
                            basin_priority = analyzer.basin_prioritization(event_results)
                            st.session_state.basin_priority = basin_priority

                        # Store processed data
                        st.session_state.flow_df = flow_df_processed
                        st.session_state.rain_df = rain_df_processed
                        st.session_state.analysis_complete = True

                    st.success("✅ Analysis Complete!")
                    st.balloons()
        else:
            st.warning("⚠️ Please upload Flow and Rainfall data first")

    with tab3:
        st.header("Analysis Results")

        if st.session_state.analysis_complete:
            # Summary metrics
            st.subheader("📊 Summary Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Rain Events Detected",
                    len(st.session_state.events)
                )

            with col2:
                st.metric(
                    "Baseline Flow",
                    f"{st.session_state.baseline:.1f} GPM"
                )

            with col3:
                total_ii = sum(e['ii_volume_mgal'] for e in st.session_state.event_results)
                st.metric(
                    "Total I&I Volume",
                    f"{total_ii:.2f} MGal"
                )

            with col4:
                if 'rdii' in st.session_state:
                    st.metric(
                        "R-Value",
                        f"{st.session_state.rdii.get('r_value', 0):.3f}"
                    )

            st.markdown("---")

            # Main hydrograph
            st.subheader("📈 Hydrograph")
            fig_hydro = Visualizer.plot_hydrograph(
                st.session_state.flow_df,
                st.session_state.rain_df,
                st.session_state.events if show_events else None,
                show_baseflow
            )
            st.plotly_chart(fig_hydro, use_container_width=True)

            # Event summary
            if st.session_state.event_results:
                st.subheader("🌧️ Event Summary")
                fig_events = Visualizer.plot_event_summary(st.session_state.event_results)
                st.plotly_chart(fig_events, use_container_width=True)

                # Event table
                with st.expander("📋 Event Details"):
                    event_df = pd.DataFrame(st.session_state.event_results)
                    st.dataframe(event_df, use_container_width=True)

            # Basin comparison
            if 'basin_priority' in st.session_state and not st.session_state.basin_priority.empty:
                st.subheader("🏆 Basin Prioritization")
                fig_basin = Visualizer.plot_basin_comparison(st.session_state.basin_priority)
                st.plotly_chart(fig_basin, use_container_width=True)

                st.dataframe(st.session_state.basin_priority, use_container_width=True)
        else:
            st.info("👈 Run the analysis first to see results")

    with tab4:
        st.header("Generate Report")

        if st.session_state.analysis_complete:
            st.subheader("📄 Summary Report")

            # Text summary
            st.markdown(f"""
            ### Analysis Summary

            **Period:** {st.session_state.flow_df['timestamp'].min()} to {st.session_state.flow_df['timestamp'].max()}

            **Rain Events:** {len(st.session_state.events)} events detected

            **Baseline Flow:** {st.session_state.baseline:.1f} GPM

            **Total I&I Volume:** {sum(e['ii_volume_mgal'] for e in st.session_state.event_results):.2f} MGal
            """)

            if 'rdii' in st.session_state:
                st.markdown(f"""
                **RDII Metrics:**
                - R-Value: {st.session_state.rdii.get('r_value', 0):.3f}
                - Unit I&I: {st.session_state.rdii.get('unit_ii_gal_per_inch_acre', 0):.1f} gal/in/acre
                """)

            # Download options
            st.subheader("💾 Download Options")

            col1, col2, col3 = st.columns(3)

            with col1:
                # Download event summary as CSV
                if st.session_state.event_results:
                    event_csv = pd.DataFrame(st.session_state.event_results).to_csv(index=False)
                    st.download_button(
                        "📥 Download Event Data (CSV)",
                        event_csv,
                        "ii_event_summary.csv",
                        "text/csv"
                    )

            with col2:
                # Download full results as JSON
                results_json = json.dumps({
                    'baseline_flow_gpm': st.session_state.baseline,
                    'events': st.session_state.event_results,
                    'rdii': st.session_state.rdii if 'rdii' in st.session_state else {}
                }, indent=2, default=str)
                st.download_button(
                    "📥 Download Results (JSON)",
                    results_json,
                    "ii_analysis_results.json",
                    "application/json"
                )

            with col3:
                # Download basin priority
                if 'basin_priority' in st.session_state and not st.session_state.basin_priority.empty:
                    basin_csv = st.session_state.basin_priority.to_csv(index=False)
                    st.download_button(
                        "📥 Download Basin Priority (CSV)",
                        basin_csv,
                        "basin_prioritization.csv",
                        "text/csv"
                    )
        else:
            st.info("👈 Complete the analysis first to generate a report")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>I&I Analysis Tool v2.0 | Built with Streamlit</p>
        <p>💡 <b>Tip:</b> Hover over any chart for interactive details</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
