#!/usr/bin/env python3
"""
================================================================================
INFILTRATION & INFLOW (I&I) ANALYSIS TOOL FOR SANITARY SEWER SYSTEMS
================================================================================

Author: Expert Python Developer for Civil/Water Engineering
Purpose: Comprehensive I&I analysis using EPA standard methods
Version: 1.0

DESCRIPTION:
This tool analyzes sanitary sewer flow data against rainfall data to quantify
infiltration and inflow (I&I) contributions. It follows EPA methodologies for
identifying dry-weather baselines, detecting rain events, and separating
quick-response inflow from slower infiltration.

INPUTS:
1. flow_data.csv: timestamp, flow_rate_gpm, [optional: manhole_id, basin_id]
2. rainfall_data.csv: timestamp, rainfall_in
3. [optional] asset_data.csv: basin_id, area_acres, pipe_age_years, pipe_length_miles

OUTPUTS:
- Matplotlib hydrographs (flow vs. rainfall with I&I highlighted)
- Excel report with summary metrics, event details, and charts
- Prioritization scores for targeted remediation

METHODS:
- Dry-weather baseline: 7-day rolling median during dry periods
- Event detection: Automatic rain event separation
- Inflow: Excess flow within 2-6 hours of rainfall start
- Infiltration: Delayed excess flow 24-72 hours post-rain
- Metrics: Peak I&I, I&I volume/inch, % I&I contribution

USAGE:
    python ii_analysis.py --flow flow_data.csv --rain rainfall_data.csv
    
    Or use the Streamlit interface:
    streamlit run ii_analysis.py

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import seaborn as sns

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# UNIT CONVERSION UTILITIES
# ============================================================================

class UnitConverter:
    """Handle unit conversions for flow and rainfall data."""
    
    FLOW_CONVERSIONS = {
        'gpm': 1.0,                    # Gallons per minute (base)
        'gpd': 1440.0,                 # Gallons per day
        'mgd': 1_440_000.0,            # Million gallons per day
        'cfs': 448.831,                # Cubic feet per second
        'cmd': 1440 / 264.172,         # Cubic meters per day
        'lps': 3.78541,                # Liters per second
        'lpm': 3.78541 / 60,           # Liters per minute
    }
    
    RAINFALL_CONVERSIONS = {
        'in': 1.0,                     # Inches (base)
        'mm': 25.4,                    # Millimeters
        'cm': 2.54,                    # Centimeters
    }
    
    @staticmethod
    def convert_flow(value: float, from_unit: str, to_unit: str = 'gpm') -> float:
        """Convert flow rate between units."""
        if from_unit == to_unit:
            return value
        # Convert to base (gpm) then to target
        base_value = value * UnitConverter.FLOW_CONVERSIONS.get(from_unit.lower(), 1.0)
        return base_value / UnitConverter.FLOW_CONVERSIONS.get(to_unit.lower(), 1.0)
    
    @staticmethod
    def convert_rainfall(value: float, from_unit: str, to_unit: str = 'in') -> float:
        """Convert rainfall between units."""
        if from_unit == to_unit:
            return value
        # Convert to base (inches) then to target
        base_value = value * UnitConverter.RAINFALL_CONVERSIONS.get(from_unit.lower(), 1.0)
        return base_value / UnitConverter.RAINFALL_CONVERSIONS.get(to_unit.lower(), 1.0)


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class DataLoader:
    """Load and preprocess flow, rainfall, and asset data."""
    
    def __init__(self, flow_file: str, rain_file: str, asset_file: Optional[str] = None,
                 flow_unit: str = 'gpm', rain_unit: str = 'in'):
        """
        Initialize data loader.
        
        Args:
            flow_file: Path to flow data CSV
            rain_file: Path to rainfall data CSV
            asset_file: Optional path to asset/basin data CSV
            flow_unit: Unit of flow measurements
            rain_unit: Unit of rainfall measurements
        """
        self.flow_file = flow_file
        self.rain_file = rain_file
        self.asset_file = asset_file
        self.flow_unit = flow_unit
        self.rain_unit = rain_unit
        
    def load_flow_data(self) -> pd.DataFrame:
        """Load and validate flow data."""
        print("Loading flow data...")
        df = pd.read_csv(self.flow_file, parse_dates=['timestamp'])
        
        # Validate required columns
        required = ['timestamp', 'flow_rate_gpm']
        if not all(col in df.columns for col in required):
            # Try to find flow column with different name
            flow_cols = [col for col in df.columns if 'flow' in col.lower()]
            if flow_cols:
                df = df.rename(columns={flow_cols[0]: 'flow_rate_gpm'})
            else:
                raise ValueError(f"Flow data must contain 'timestamp' and 'flow_rate_gpm' columns")
        
        # Convert units if necessary
        if self.flow_unit.lower() != 'gpm':
            print(f"Converting flow from {self.flow_unit} to GPM...")
            df['flow_rate_gpm'] = df['flow_rate_gpm'].apply(
                lambda x: UnitConverter.convert_flow(x, self.flow_unit, 'gpm')
            )
        
        # Handle timezone awareness
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Handle negative flows (sensor errors)
        df.loc[df['flow_rate_gpm'] < 0, 'flow_rate_gpm'] = np.nan
        
        print(f"  Loaded {len(df)} flow records from {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Flow range: {df['flow_rate_gpm'].min():.1f} - {df['flow_rate_gpm'].max():.1f} GPM")
        
        return df
    
    def load_rainfall_data(self) -> pd.DataFrame:
        """Load and validate rainfall data."""
        print("Loading rainfall data...")
        df = pd.read_csv(self.rain_file, parse_dates=['timestamp'])
        
        # Validate required columns
        if 'rainfall_in' not in df.columns:
            rain_cols = [col for col in df.columns if 'rain' in col.lower()]
            if rain_cols:
                df = df.rename(columns={rain_cols[0]: 'rainfall_in'})
            else:
                raise ValueError("Rainfall data must contain 'timestamp' and 'rainfall_in' columns")
        
        # Convert units if necessary
        if self.rain_unit.lower() != 'in':
            print(f"Converting rainfall from {self.rain_unit} to inches...")
            df['rainfall_in'] = df['rainfall_in'].apply(
                lambda x: UnitConverter.convert_rainfall(x, self.rain_unit, 'in')
            )
        
        # Handle timezone awareness
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Handle negative rainfall (sensor errors)
        df.loc[df['rainfall_in'] < 0, 'rainfall_in'] = 0
        
        # If rainfall is cumulative, convert to incremental
        if df['rainfall_in'].is_monotonic_increasing:
            print("  Detected cumulative rainfall - converting to incremental...")
            df['rainfall_in'] = df['rainfall_in'].diff().fillna(0)
        
        print(f"  Loaded {len(df)} rainfall records")
        print(f"  Total rainfall: {df['rainfall_in'].sum():.2f} inches")
        
        return df
    
    def load_asset_data(self) -> Optional[pd.DataFrame]:
        """Load optional asset/basin data."""
        if self.asset_file is None:
            return None
        
        print("Loading asset data...")
        df = pd.read_csv(self.asset_file)
        print(f"  Loaded asset data for {len(df)} basins/areas")
        return df
    
    def merge_and_resample(self, flow_df: pd.DataFrame, rain_df: pd.DataFrame, 
                           interval: str = '1H') -> pd.DataFrame:
        """
        Merge flow and rainfall data and resample to consistent interval.
        
        Args:
            flow_df: Flow data DataFrame
            rain_df: Rainfall data DataFrame
            interval: Resampling interval (default: '1H' for hourly)
            
        Returns:
            Merged and resampled DataFrame
        """
        print(f"\nMerging and resampling data to {interval} intervals...")
        
        # Set timestamp as index
        flow_df = flow_df.set_index('timestamp')
        rain_df = rain_df.set_index('timestamp')
        
        # Resample flow (mean) and rainfall (sum)
        flow_resampled = flow_df.resample(interval).agg({
            'flow_rate_gpm': 'mean',
            **{col: 'first' for col in flow_df.columns if col not in ['flow_rate_gpm']}
        })
        
        rain_resampled = rain_df.resample(interval).agg({'rainfall_in': 'sum'})
        
        # Merge on timestamp
        merged = flow_resampled.join(rain_resampled, how='outer')
        
        # Forward fill flow data for short gaps (up to 2 intervals)
        merged['flow_rate_gpm'] = merged['flow_rate_gpm'].fillna(method='ffill', limit=2)
        
        # Fill rainfall NaN with 0
        merged['rainfall_in'] = merged['rainfall_in'].fillna(0)
        
        # Calculate antecedent rainfall (72-hour rolling sum)
        merged['antecedent_rain_72h'] = merged['rainfall_in'].rolling(
            window=72 if interval == '1H' else 72 * 24, min_periods=1
        ).sum()
        
        merged = merged.reset_index()
        
        print(f"  Merged dataset: {len(merged)} records")
        print(f"  Missing flow data: {merged['flow_rate_gpm'].isna().sum()} records ({merged['flow_rate_gpm'].isna().sum()/len(merged)*100:.1f}%)")
        
        return merged


# ============================================================================
# RAIN EVENT DETECTION
# ============================================================================

class RainEventDetector:
    """Detect and characterize individual rain events."""
    
    def __init__(self, min_event_rainfall: float = 0.1, 
                 min_dry_period_hours: int = 6,
                 dry_day_threshold: float = 0.1):
        """
        Initialize rain event detector.
        
        Args:
            min_event_rainfall: Minimum total rainfall to count as event (inches)
            min_dry_period_hours: Hours of dry weather to separate events
            dry_day_threshold: Max antecedent rain for "dry day" classification
        """
        self.min_event_rainfall = min_event_rainfall
        self.min_dry_period_hours = min_dry_period_hours
        self.dry_day_threshold = dry_day_threshold
    
    def detect_events(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Detect rain events and classify wet/dry periods.
        
        Args:
            df: Merged data with timestamp, rainfall_in, antecedent_rain_72h
            
        Returns:
            Tuple of (enhanced DataFrame, list of event summaries)
        """
        print("\nDetecting rain events...")
        
        # Mark dry days (< 0.1" rain in prior 72 hours)
        df['is_dry_day'] = df['antecedent_rain_72h'] < self.dry_day_threshold
        
        # Identify rain periods (any rainfall > 0)
        df['is_raining'] = df['rainfall_in'] > 0
        
        # Create event groups (separate events by dry periods)
        df['rain_event_id'] = 0
        event_id = 0
        in_event = False
        dry_count = 0
        
        for idx, row in df.iterrows():
            if row['is_raining']:
                if not in_event:
                    event_id += 1
                    in_event = True
                dry_count = 0
                df.at[idx, 'rain_event_id'] = event_id
            else:
                if in_event:
                    dry_count += 1
                    # Continue event during short dry periods
                    if dry_count <= self.min_dry_period_hours:
                        df.at[idx, 'rain_event_id'] = event_id
                    else:
                        in_event = False
        
        # Summarize events
        events = []
        for event_id in df[df['rain_event_id'] > 0]['rain_event_id'].unique():
            event_data = df[df['rain_event_id'] == event_id]
            total_rain = event_data['rainfall_in'].sum()
            
            if total_rain >= self.min_event_rainfall:
                events.append({
                    'event_id': int(event_id),
                    'start_time': event_data['timestamp'].min(),
                    'end_time': event_data['timestamp'].max(),
                    'duration_hours': len(event_data),
                    'total_rainfall_in': total_rain,
                    'max_intensity_in_hr': event_data['rainfall_in'].max(),
                    'antecedent_dry_hours': self._get_antecedent_dry_period(df, event_data['timestamp'].min())
                })
        
        # Mark wet periods (during and after rain events)
        df['wet_period'] = False
        for event in events:
            # Mark wet from event start to 72 hours after event end
            wet_start = event['start_time']
            wet_end = event['end_time'] + timedelta(hours=72)
            df.loc[(df['timestamp'] >= wet_start) & (df['timestamp'] <= wet_end), 'wet_period'] = True
        
        print(f"  Detected {len(events)} rain events")
        print(f"  Dry periods: {df['is_dry_day'].sum()} hours ({df['is_dry_day'].sum()/len(df)*100:.1f}%)")
        
        return df, events
    
    def _get_antecedent_dry_period(self, df: pd.DataFrame, event_start: datetime) -> int:
        """Calculate hours of dry weather before event."""
        prior_data = df[df['timestamp'] < event_start].tail(168)  # Look back 7 days max
        if len(prior_data) == 0:
            return 0
        
        # Count consecutive dry hours before event
        dry_hours = 0
        for idx in range(len(prior_data) - 1, -1, -1):
            if prior_data.iloc[idx]['rainfall_in'] == 0:
                dry_hours += 1
            else:
                break
        return dry_hours


# ============================================================================
# DRY-WEATHER BASELINE CALCULATION
# ============================================================================

class BaselineCalculator:
    """Calculate dry-weather flow baseline."""
    
    def __init__(self, window_days: int = 7, method: str = 'median'):
        """
        Initialize baseline calculator.
        
        Args:
            window_days: Rolling window size in days
            method: 'median', 'mean', or 'percentile_25'
        """
        self.window_days = window_days
        self.method = method
    
    def calculate_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate dry-weather baseline flow.
        
        Args:
            df: DataFrame with flow_rate_gpm and is_dry_day columns
            
        Returns:
            DataFrame with baseline_flow_gpm column added
        """
        print("\nCalculating dry-weather baseline...")
        
        # Calculate rolling baseline using only dry-day flows
        window_hours = self.window_days * 24
        
        if self.method == 'median':
            baseline = df[df['is_dry_day']]['flow_rate_gpm'].rolling(
                window=window_hours, center=True, min_periods=window_hours // 2
            ).median()
        elif self.method == 'mean':
            baseline = df[df['is_dry_day']]['flow_rate_gpm'].rolling(
                window=window_hours, center=True, min_periods=window_hours // 2
            ).mean()
        elif self.method == 'percentile_25':
            baseline = df[df['is_dry_day']]['flow_rate_gpm'].rolling(
                window=window_hours, center=True, min_periods=window_hours // 2
            ).quantile(0.25)
        else:
            raise ValueError(f"Unknown baseline method: {self.method}")
        
        # Reindex to full dataset and forward/backward fill
        df['baseline_flow_gpm'] = baseline.reindex(df.index)
        df['baseline_flow_gpm'] = df['baseline_flow_gpm'].fillna(method='ffill').fillna(method='bfill')
        
        # Also calculate diurnal pattern (hourly average baseline for each hour of day)
        if 'timestamp' in df.columns:
            df['hour_of_day'] = df['timestamp'].dt.hour
            dry_hourly = df[df['is_dry_day']].groupby('hour_of_day')['flow_rate_gpm'].median()
            df['diurnal_baseline'] = df['hour_of_day'].map(dry_hourly)
            
            # Use maximum of rolling baseline and diurnal pattern
            df['baseline_flow_gpm'] = df[['baseline_flow_gpm', 'diurnal_baseline']].max(axis=1)
        
        avg_baseline = df['baseline_flow_gpm'].mean()
        print(f"  Average baseline flow: {avg_baseline:.1f} GPM")
        print(f"  Baseline range: {df['baseline_flow_gpm'].min():.1f} - {df['baseline_flow_gpm'].max():.1f} GPM")
        
        return df


# ============================================================================
# I&I QUANTIFICATION
# ============================================================================

class IIAnalyzer:
    """Quantify infiltration and inflow contributions."""
    
    def __init__(self, inflow_window_hours: int = 6, 
                 infiltration_start_hours: int = 24,
                 infiltration_end_hours: int = 72):
        """
        Initialize I&I analyzer.
        
        Args:
            inflow_window_hours: Time window for inflow response (2-6 hours typical)
            infiltration_start_hours: Start of infiltration window (24 hours typical)
            infiltration_end_hours: End of infiltration window (72 hours typical)
        """
        self.inflow_window_hours = inflow_window_hours
        self.infiltration_start_hours = infiltration_start_hours
        self.infiltration_end_hours = infiltration_end_hours
    
    def analyze_ii(self, df: pd.DataFrame, events: List[Dict]) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Quantify I&I for each rain event.
        
        Args:
            df: DataFrame with flow, baseline, and event data
            events: List of rain event dictionaries
            
        Returns:
            Tuple of (enhanced DataFrame, enhanced events list)
        """
        print("\nQuantifying I&I contributions...")
        
        # Calculate excess flow (total flow minus baseline)
        df['excess_flow_gpm'] = df['flow_rate_gpm'] - df['baseline_flow_gpm']
        df['excess_flow_gpm'] = df['excess_flow_gpm'].clip(lower=0)  # No negative excess
        
        # Initialize I&I component columns
        df['inflow_gpm'] = 0.0
        df['infiltration_gpm'] = 0.0
        
        # Analyze each event
        for event in events:
            event_start = event['start_time']
            event_end = event['end_time']
            
            # Inflow period: during rain and first N hours after
            inflow_end = event_end + timedelta(hours=self.inflow_window_hours)
            inflow_mask = (df['timestamp'] >= event_start) & (df['timestamp'] <= inflow_end)
            
            # Infiltration period: delayed response
            infil_start = event_end + timedelta(hours=self.infiltration_start_hours)
            infil_end = event_end + timedelta(hours=self.infiltration_end_hours)
            infil_mask = (df['timestamp'] >= infil_start) & (df['timestamp'] <= infil_end)
            
            # Assign excess flow to components
            df.loc[inflow_mask, 'inflow_gpm'] = df.loc[inflow_mask, 'excess_flow_gpm']
            df.loc[infil_mask, 'infiltration_gpm'] = df.loc[infil_mask, 'excess_flow_gpm']
            
            # Calculate event metrics
            event_data = df[inflow_mask | infil_mask]
            
            if len(event_data) > 0:
                # Peak flows
                event['peak_total_flow_gpm'] = df.loc[inflow_mask, 'flow_rate_gpm'].max()
                event['peak_excess_flow_gpm'] = df.loc[inflow_mask, 'excess_flow_gpm'].max()
                event['peak_inflow_gpm'] = df.loc[inflow_mask, 'inflow_gpm'].max()
                
                # Volumes (convert GPM to gallons: GPM * 60 min/hr * hours)
                event['inflow_volume_gal'] = df.loc[inflow_mask, 'inflow_gpm'].sum() * 60
                event['infiltration_volume_gal'] = df.loc[infil_mask, 'infiltration_gpm'].sum() * 60
                event['total_ii_volume_gal'] = event['inflow_volume_gal'] + event['infiltration_volume_gal']
                
                # Normalized metrics
                if event['total_rainfall_in'] > 0:
                    event['ii_gal_per_inch'] = event['total_ii_volume_gal'] / event['total_rainfall_in']
                else:
                    event['ii_gal_per_inch'] = 0
                
                # Percentage of total flow
                total_flow_vol = df.loc[inflow_mask | infil_mask, 'flow_rate_gpm'].sum() * 60
                if total_flow_vol > 0:
                    event['ii_percent_of_total'] = (event['total_ii_volume_gal'] / total_flow_vol) * 100
                else:
                    event['ii_percent_of_total'] = 0
                
                # Response time (hours from rain start to peak flow)
                peak_idx = df.loc[inflow_mask, 'flow_rate_gpm'].idxmax()
                if pd.notna(peak_idx):
                    peak_time = df.loc[peak_idx, 'timestamp']
                    event['response_time_hours'] = (peak_time - event_start).total_seconds() / 3600
                else:
                    event['response_time_hours'] = np.nan
        
        # Calculate summary statistics
        total_ii_volume = sum(e.get('total_ii_volume_gal', 0) for e in events)
        total_rainfall = sum(e['total_rainfall_in'] for e in events)
        
        print(f"  Analyzed {len(events)} events")
        print(f"  Total I&I volume: {total_ii_volume:,.0f} gallons")
        if total_rainfall > 0:
            print(f"  Average I&I: {total_ii_volume / total_rainfall:,.0f} gal/inch")
        
        return df, events


# ============================================================================
# BASIN-LEVEL ANALYSIS
# ============================================================================

class BasinAnalyzer:
    """Analyze I&I by basin or sub-area."""
    
    def __init__(self, asset_data: Optional[pd.DataFrame] = None):
        """
        Initialize basin analyzer.
        
        Args:
            asset_data: Optional DataFrame with basin metadata
        """
        self.asset_data = asset_data
    
    def analyze_by_basin(self, df: pd.DataFrame, events: List[Dict]) -> Dict:
        """
        Calculate I&I metrics by basin/area.
        
        Args:
            df: Main DataFrame with basin_id or manhole_id column
            events: List of rain events
            
        Returns:
            Dictionary of basin-level metrics
        """
        basin_col = None
        if 'basin_id' in df.columns:
            basin_col = 'basin_id'
        elif 'manhole_id' in df.columns:
            basin_col = 'manhole_id'
        
        if basin_col is None:
            print("\nNo basin/manhole ID column found - skipping basin-level analysis")
            return {}
        
        print(f"\nAnalyzing I&I by {basin_col}...")
        
        basins = {}
        for basin_id in df[basin_col].dropna().unique():
            basin_data = df[df[basin_col] == basin_id]
            
            # Calculate metrics for this basin
            total_ii = basin_data['excess_flow_gpm'].sum() * 60  # Convert to gallons
            total_inflow = basin_data['inflow_gpm'].sum() * 60
            total_infiltration = basin_data['infiltration_gpm'].sum() * 60
            
            avg_baseline = basin_data['baseline_flow_gpm'].mean()
            avg_wet_flow = basin_data[basin_data['wet_period']]['flow_rate_gpm'].mean()
            
            basins[basin_id] = {
                'basin_id': basin_id,
                'total_ii_volume_gal': total_ii,
                'inflow_volume_gal': total_inflow,
                'infiltration_volume_gal': total_infiltration,
                'avg_baseline_gpm': avg_baseline,
                'avg_wet_weather_gpm': avg_wet_flow,
                'peak_flow_gpm': basin_data['flow_rate_gpm'].max(),
                'ii_percent': (total_ii / (basin_data['flow_rate_gpm'].sum() * 60)) * 100 if basin_data['flow_rate_gpm'].sum() > 0 else 0
            }
            
            # Add asset data if available
            if self.asset_data is not None and basin_col in self.asset_data.columns:
                asset_row = self.asset_data[self.asset_data[basin_col] == basin_id]
                if len(asset_row) > 0:
                    asset_row = asset_row.iloc[0]
                    basins[basin_id]['area_acres'] = asset_row.get('area_acres', np.nan)
                    basins[basin_id]['pipe_age_years'] = asset_row.get('pipe_age_years', np.nan)
                    basins[basin_id]['pipe_length_miles'] = asset_row.get('pipe_length_miles', np.nan)
                    
                    # Calculate normalized metrics
                    if 'area_acres' in asset_row and pd.notna(asset_row['area_acres']):
                        basins[basin_id]['ii_gal_per_acre'] = total_ii / asset_row['area_acres']
                    if 'pipe_length_miles' in asset_row and pd.notna(asset_row['pipe_length_miles']):
                        basins[basin_id]['ii_gal_per_mile'] = total_ii / asset_row['pipe_length_miles']
        
        print(f"  Analyzed {len(basins)} basins/areas")
        return basins


# ============================================================================
# PRIORITIZATION ENGINE
# ============================================================================

class Prioritizer:
    """Prioritize basins/areas for I&I remediation."""
    
    def prioritize_basins(self, basin_metrics: Dict, events: List[Dict]) -> pd.DataFrame:
        """
        Score and rank basins for remediation priority.
        
        Args:
            basin_metrics: Dictionary of basin-level metrics
            events: List of rain events
            
        Returns:
            DataFrame with prioritization scores
        """
        if not basin_metrics:
            return pd.DataFrame()
        
        print("\nPrioritizing basins for remediation...")
        
        df = pd.DataFrame(basin_metrics.values())
        
        # Normalize metrics for scoring (0-100 scale)
        def normalize(series):
            if series.max() == series.min():
                return pd.Series([50] * len(series))
            return ((series - series.min()) / (series.max() - series.min())) * 100
        
        # Individual component scores
        df['volume_score'] = normalize(df['total_ii_volume_gal'])
        df['percent_score'] = normalize(df['ii_percent'])
        df['peak_score'] = normalize(df['peak_flow_gpm'])
        
        # Bonus for asset data
        if 'pipe_age_years' in df.columns:
            df['age_score'] = normalize(df['pipe_age_years'].fillna(0))
        else:
            df['age_score'] = 0
        
        if 'ii_gal_per_acre' in df.columns:
            df['intensity_score'] = normalize(df['ii_gal_per_acre'].fillna(0))
        else:
            df['intensity_score'] = 0
        
        # Composite priority score (weighted average)
        weights = {
            'volume_score': 0.35,
            'percent_score': 0.25,
            'peak_score': 0.20,
            'age_score': 0.10,
            'intensity_score': 0.10
        }
        
        df['priority_score'] = sum(df[col] * weight for col, weight in weights.items())
        
        # Rank basins
        df = df.sort_values('priority_score', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        # Add recommendations
        df['recommendation'] = df.apply(self._generate_recommendation, axis=1)
        
        print(f"  Ranked {len(df)} basins")
        print(f"  Top priority: Basin {df.iloc[0]['basin_id']} (score: {df.iloc[0]['priority_score']:.1f})")
        
        return df
    
    def _generate_recommendation(self, row) -> str:
        """Generate remediation recommendation for a basin."""
        recommendations = []
        
        if row['ii_percent'] > 50:
            recommendations.append("High I&I percentage - investigate system integrity")
        
        if row.get('peak_flow_gpm', 0) / row.get('avg_baseline_gpm', 1) > 3:
            recommendations.append("Large flow spikes - check for illegal connections")
        
        inflow_ratio = row.get('inflow_volume_gal', 0) / max(row.get('total_ii_volume_gal', 1), 1)
        if inflow_ratio > 0.7:
            recommendations.append("Dominated by inflow - inspect roof drains, sump pumps")
        elif inflow_ratio < 0.3:
            recommendations.append("Dominated by infiltration - inspect pipe joints, manholes")
        
        if row.get('pipe_age_years', 0) > 40:
            recommendations.append("Aging infrastructure - consider rehabilitation/replacement")
        
        if not recommendations:
            recommendations.append("Monitor and reassess after additional rain events")
        
        return "; ".join(recommendations)


# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Create plots and charts for I&I analysis."""
    
    def __init__(self, output_dir: str = './output'):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def plot_hydrograph(self, df: pd.DataFrame, events: List[Dict], 
                        title: str = "I&I Analysis Hydrograph") -> str:
        """
        Create comprehensive hydrograph plot.
        
        Args:
            df: Main DataFrame with all data
            events: List of rain events
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # Flow plot
        ax1.plot(df['timestamp'], df['flow_rate_gpm'], 'b-', linewidth=1, 
                 label='Observed Flow', alpha=0.7)
        ax1.plot(df['timestamp'], df['baseline_flow_gpm'], 'g--', linewidth=2, 
                 label='Dry Weather Baseline')
        ax1.fill_between(df['timestamp'], df['baseline_flow_gpm'], df['flow_rate_gpm'],
                         where=(df['flow_rate_gpm'] > df['baseline_flow_gpm']),
                         color='red', alpha=0.3, label='I&I Contribution')
        
        # Highlight rain events
        for event in events:
            ax1.axvspan(event['start_time'], event['end_time'], 
                       color='lightblue', alpha=0.2, zorder=0)
        
        ax1.set_ylabel('Flow Rate (GPM)', fontsize=12, fontweight='bold')
        ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = (
            f"Avg Baseline: {df['baseline_flow_gpm'].mean():.1f} GPM\n"
            f"Peak Flow: {df['flow_rate_gpm'].max():.1f} GPM\n"
            f"Total I&I Events: {len(events)}"
        )
        ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Rainfall plot
        ax2.bar(df['timestamp'], df['rainfall_in'], width=0.04, 
                color='steelblue', alpha=0.7, label='Rainfall')
        ax2.set_ylabel('Rainfall (inches)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'hydrograph.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved hydrograph to {filepath}")
        return str(filepath)
    
    def plot_event_summary(self, events: List[Dict]) -> str:
        """Create event summary plots."""
        if not events:
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        events_df = pd.DataFrame(events)
        
        # 1. I&I Volume vs Rainfall
        ax = axes[0, 0]
        ax.scatter(events_df['total_rainfall_in'], 
                  events_df['total_ii_volume_gal'] / 1000,  # Convert to thousands
                  s=100, alpha=0.6, c='steelblue', edgecolors='black')
        ax.set_xlabel('Total Rainfall (inches)', fontweight='bold')
        ax.set_ylabel('I&I Volume (1000 gal)', fontweight='bold')
        ax.set_title('I&I Volume vs Rainfall', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add trendline
        if len(events_df) > 1:
            z = np.polyfit(events_df['total_rainfall_in'], 
                          events_df['total_ii_volume_gal'] / 1000, 1)
            p = np.poly1d(z)
            x_line = np.linspace(events_df['total_rainfall_in'].min(), 
                                events_df['total_rainfall_in'].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
        
        # 2. Inflow vs Infiltration
        ax = axes[0, 1]
        x = np.arange(len(events_df))
        width = 0.35
        ax.bar(x, events_df['inflow_volume_gal'] / 1000, width, 
               label='Inflow', color='orangered', alpha=0.7)
        ax.bar(x, events_df['infiltration_volume_gal'] / 1000, width, 
               bottom=events_df['inflow_volume_gal'] / 1000,
               label='Infiltration', color='dodgerblue', alpha=0.7)
        ax.set_xlabel('Event Number', fontweight='bold')
        ax.set_ylabel('Volume (1000 gal)', fontweight='bold')
        ax.set_title('Inflow vs Infiltration by Event', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Response Time Distribution
        ax = axes[1, 0]
        valid_response = events_df['response_time_hours'].dropna()
        if len(valid_response) > 0:
            ax.hist(valid_response, bins=15, color='seagreen', alpha=0.7, edgecolor='black')
            ax.axvline(valid_response.median(), color='red', linestyle='--', 
                      linewidth=2, label=f'Median: {valid_response.median():.1f} hr')
            ax.set_xlabel('Response Time (hours)', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title('I&I Response Time Distribution', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Peak I&I Flow
        ax = axes[1, 1]
        ax.bar(range(len(events_df)), events_df['peak_excess_flow_gpm'], 
               color='crimson', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Event Number', fontweight='bold')
        ax.set_ylabel('Peak Excess Flow (GPM)', fontweight='bold')
        ax.set_title('Peak I&I Flow by Event', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'event_summary.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved event summary to {filepath}")
        return str(filepath)
    
    def plot_basin_comparison(self, basin_df: pd.DataFrame) -> str:
        """Create basin comparison plots."""
        if basin_df is None or len(basin_df) == 0:
            return ""
        
        # Show top 10 basins only
        plot_df = basin_df.head(10)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Priority Score
        ax = axes[0]
        bars = ax.barh(range(len(plot_df)), plot_df['priority_score'], 
                       color=plt.cm.RdYlGn_r(plot_df['priority_score'] / 100))
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels([f"Basin {bid}" for bid in plot_df['basin_id']])
        ax.set_xlabel('Priority Score', fontweight='bold')
        ax.set_title('Top 10 Basins by Priority', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (idx, row) in enumerate(plot_df.iterrows()):
            ax.text(row['priority_score'] + 1, i, f"{row['priority_score']:.1f}", 
                   va='center', fontweight='bold')
        
        # 2. I&I Volume
        ax = axes[1]
        ax.bar(range(len(plot_df)), plot_df['total_ii_volume_gal'] / 1000,
               color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels([f"Basin {bid}" for bid in plot_df['basin_id']], 
                          rotation=45, ha='right')
        ax.set_ylabel('I&I Volume (1000 gal)', fontweight='bold')
        ax.set_title('Total I&I Volume by Basin', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'basin_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved basin comparison to {filepath}")
        return str(filepath)


# ============================================================================
# EXCEL REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """Generate comprehensive Excel reports."""
    
    def __init__(self, output_dir: str = './output'):
        """Initialize report generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_excel_report(self, df: pd.DataFrame, events: List[Dict],
                             basin_metrics: Dict, basin_priority: pd.DataFrame,
                             plot_files: List[str]) -> str:
        """
        Create comprehensive Excel report.
        
        Args:
            df: Main DataFrame
            events: List of event dictionaries
            basin_metrics: Basin-level metrics
            basin_priority: Basin prioritization DataFrame
            plot_files: List of plot file paths
            
        Returns:
            Path to Excel file
        """
        print("\nGenerating Excel report...")
        
        filepath = self.output_dir / f'II_Analysis_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Sheet 1: Executive Summary
            summary_data = {
                'Metric': [
                    'Analysis Period Start',
                    'Analysis Period End',
                    'Total Duration (days)',
                    'Number of Rain Events',
                    'Total Rainfall (inches)',
                    'Average Dry Weather Flow (GPM)',
                    'Peak Observed Flow (GPM)',
                    'Total I&I Volume (gallons)',
                    'Average I&I per inch (gal/in)',
                    'I&I as % of Total Flow',
                ],
                'Value': [
                    df['timestamp'].min().strftime('%Y-%m-%d %H:%M'),
                    df['timestamp'].max().strftime('%Y-%m-%d %H:%M'),
                    (df['timestamp'].max() - df['timestamp'].min()).days,
                    len(events),
                    f"{df['rainfall_in'].sum():.2f}",
                    f"{df['baseline_flow_gpm'].mean():.1f}",
                    f"{df['flow_rate_gpm'].max():.1f}",
                    f"{sum(e.get('total_ii_volume_gal', 0) for e in events):,.0f}",
                    f"{sum(e.get('total_ii_volume_gal', 0) for e in events) / max(df['rainfall_in'].sum(), 0.001):,.0f}",
                    f"{(df['excess_flow_gpm'].sum() / df['flow_rate_gpm'].sum() * 100):.1f}%",
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # Sheet 2: Rain Events
            if events:
                events_df = pd.DataFrame(events)
                # Format datetime columns
                for col in ['start_time', 'end_time']:
                    if col in events_df.columns:
                        events_df[col] = events_df[col].dt.strftime('%Y-%m-%d %H:%M')
                events_df.to_excel(writer, sheet_name='Rain Events', index=False)
            
            # Sheet 3: Basin Rankings
            if not basin_priority.empty:
                basin_priority.to_excel(writer, sheet_name='Basin Priority', index=False)
            
            # Sheet 4: Time Series Data (sample)
            output_cols = ['timestamp', 'flow_rate_gpm', 'baseline_flow_gpm', 
                          'excess_flow_gpm', 'rainfall_in', 'is_dry_day', 'wet_period']
            sample_df = df[output_cols].copy()
            sample_df['timestamp'] = sample_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            sample_df.to_excel(writer, sheet_name='Time Series Data', index=False)
            
            # Auto-adjust column widths
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"  Saved Excel report to {filepath}")
        return str(filepath)


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def run_analysis(flow_file: str, rain_file: str, asset_file: Optional[str] = None,
                flow_unit: str = 'gpm', rain_unit: str = 'in',
                output_dir: str = './output', resample_interval: str = '1H') -> Dict:
    """
    Run complete I&I analysis pipeline.
    
    Args:
        flow_file: Path to flow data CSV
        rain_file: Path to rainfall data CSV
        asset_file: Optional path to asset data CSV
        flow_unit: Unit of flow measurements
        rain_unit: Unit of rainfall measurements
        output_dir: Directory for output files
        resample_interval: Time interval for resampling
        
    Returns:
        Dictionary with all analysis results
    """
    print("=" * 80)
    print("INFILTRATION & INFLOW (I&I) ANALYSIS")
    print("=" * 80)
    
    # 1. Load Data
    loader = DataLoader(flow_file, rain_file, asset_file, flow_unit, rain_unit)
    flow_df = loader.load_flow_data()
    rain_df = loader.load_rainfall_data()
    asset_df = loader.load_asset_data()
    
    # 2. Merge and Resample
    df = loader.merge_and_resample(flow_df, rain_df, resample_interval)
    
    # 3. Detect Rain Events
    detector = RainEventDetector()
    df, events = detector.detect_events(df)
    
    # 4. Calculate Baseline
    baseline_calc = BaselineCalculator()
    df = baseline_calc.calculate_baseline(df)
    
    # 5. Quantify I&I
    ii_analyzer = IIAnalyzer()
    df, events = ii_analyzer.analyze_ii(df, events)
    
    # 6. Basin-Level Analysis
    basin_analyzer = BasinAnalyzer(asset_df)
    basin_metrics = basin_analyzer.analyze_by_basin(df, events)
    
    # 7. Prioritization
    prioritizer = Prioritizer()
    basin_priority = prioritizer.prioritize_basins(basin_metrics, events)
    
    # 8. Visualization
    print("\nGenerating visualizations...")
    visualizer = Visualizer(output_dir)
    plot_files = []
    plot_files.append(visualizer.plot_hydrograph(df, events))
    plot_files.append(visualizer.plot_event_summary(events))
    if not basin_priority.empty:
        plot_files.append(visualizer.plot_basin_comparison(basin_priority))
    
    # 9. Generate Report
    reporter = ReportGenerator(output_dir)
    excel_file = reporter.generate_excel_report(df, events, basin_metrics, 
                                                basin_priority, plot_files)
    
    # 10. Print Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - Excel Report: {excel_file}")
    print(f"  - Plots: {len(plot_files)} figures")
    
    if not basin_priority.empty:
        print(f"\nTop 3 Priority Basins:")
        for idx, row in basin_priority.head(3).iterrows():
            print(f"  {idx+1}. Basin {row['basin_id']}: {row['recommendation']}")
    
    return {
        'dataframe': df,
        'events': events,
        'basin_metrics': basin_metrics,
        'basin_priority': basin_priority,
        'plots': plot_files,
        'excel_report': excel_file
    }


# ============================================================================
# STREAMLIT WEB INTERFACE
# ============================================================================

def run_streamlit_app():
    """Launch Streamlit web interface."""
    import streamlit as st
    
    st.set_page_config(page_title="I&I Analysis Tool", layout="wide")
    
    st.title("🌧️ Infiltration & Inflow (I&I) Analysis Tool")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    flow_unit = st.sidebar.selectbox("Flow Unit", 
                                     ['gpm', 'gpd', 'mgd', 'cfs', 'lps', 'lpm'],
                                     index=0)
    rain_unit = st.sidebar.selectbox("Rainfall Unit",
                                     ['in', 'mm', 'cm'],
                                     index=0)
    resample_interval = st.sidebar.selectbox("Time Interval",
                                            ['15min', '30min', '1H', '2H', '6H'],
                                            index=2)
    
    # File uploads
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Flow Data")
        flow_file = st.file_uploader("Upload flow_data.csv", type=['csv'])
        if flow_file:
            st.success(f"✓ {flow_file.name}")
    
    with col2:
        st.subheader("🌧️ Rainfall Data")
        rain_file = st.file_uploader("Upload rainfall_data.csv", type=['csv'])
        if rain_file:
            st.success(f"✓ {rain_file.name}")
    
    with col3:
        st.subheader("🏗️ Asset Data (Optional)")
        asset_file = st.file_uploader("Upload asset_data.csv", type=['csv'])
        if asset_file:
            st.success(f"✓ {asset_file.name}")
    
    # Run analysis
    if flow_file and rain_file:
        if st.button("🚀 Run I&I Analysis", type="primary"):
            with st.spinner("Running analysis... This may take a minute."):
                # Save uploaded files temporarily
                temp_dir = Path('./temp_uploads')
                temp_dir.mkdir(exist_ok=True)
                
                flow_path = temp_dir / 'flow_data.csv'
                rain_path = temp_dir / 'rainfall_data.csv'
                
                with open(flow_path, 'wb') as f:
                    f.write(flow_file.getbuffer())
                with open(rain_path, 'wb') as f:
                    f.write(rain_file.getbuffer())
                
                asset_path = None
                if asset_file:
                    asset_path = temp_dir / 'asset_data.csv'
                    with open(asset_path, 'wb') as f:
                        f.write(asset_file.getbuffer())
                
                # Run analysis
                try:
                    results = run_analysis(
                        str(flow_path), str(rain_path), 
                        str(asset_path) if asset_path else None,
                        flow_unit, rain_unit,
                        output_dir='./output',
                        resample_interval=resample_interval
                    )
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    # Show plots
                    st.markdown("---")
                    st.header("📈 Visualizations")
                    
                    for plot_file in results['plots']:
                        if Path(plot_file).exists():
                            st.image(plot_file, use_container_width=True)
                    
                    # Summary metrics
                    st.markdown("---")
                    st.header("📊 Summary Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Rain Events", len(results['events']))
                    with col2:
                        total_ii = sum(e.get('total_ii_volume_gal', 0) for e in results['events'])
                        st.metric("Total I&I Volume", f"{total_ii/1e6:.2f} MG")
                    with col3:
                        avg_baseline = results['dataframe']['baseline_flow_gpm'].mean()
                        st.metric("Avg Baseline Flow", f"{avg_baseline:.0f} GPM")
                    with col4:
                        peak_flow = results['dataframe']['flow_rate_gpm'].max()
                        st.metric("Peak Flow", f"{peak_flow:.0f} GPM")
                    
                    # Basin priority table
                    if not results['basin_priority'].empty:
                        st.markdown("---")
                        st.header("🎯 Basin Prioritization")
                        st.dataframe(
                            results['basin_priority'][['rank', 'basin_id', 'priority_score', 
                                                      'total_ii_volume_gal', 'ii_percent', 
                                                      'recommendation']],
                            use_container_width=True
                        )
                    
                    # Download link
                    st.markdown("---")
                    with open(results['excel_report'], 'rb') as f:
                        st.download_button(
                            label="📥 Download Excel Report",
                            data=f,
                            file_name=Path(results['excel_report']).name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)
    else:
        st.info("👆 Please upload flow and rainfall data files to begin analysis.")
    
    # Help section
    with st.expander("ℹ️ Help & Documentation"):
        st.markdown("""
        ### Required CSV Format
        
        **flow_data.csv:**
        - `timestamp` (datetime): Date and time of measurement
        - `flow_rate_gpm` (float): Flow rate in specified units
        - `basin_id` or `manhole_id` (optional): Area identifier
        
        **rainfall_data.csv:**
        - `timestamp` (datetime): Date and time of measurement
        - `rainfall_in` (float): Rainfall depth in specified units
        
        **asset_data.csv (optional):**
        - `basin_id`: Basin identifier
        - `area_acres`: Contributing area
        - `pipe_age_years`: Average pipe age
        - `pipe_length_miles`: Total pipe length
        
        ### Methodology
        - **Dry Weather Baseline:** 7-day rolling median during dry periods
        - **Inflow:** Quick response (0-6 hours after rain)
        - **Infiltration:** Delayed response (24-72 hours after rain)
        - **Priority Scoring:** Based on volume, percentage, peak flow, and asset condition
        """)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for command line execution."""
    parser = argparse.ArgumentParser(
        description='I&I Analysis Tool for Sanitary Sewer Systems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ii_analysis.py --flow flow.csv --rain rain.csv
  python ii_analysis.py --flow flow.csv --rain rain.csv --asset basins.csv --flow-unit mgd
  streamlit run ii_analysis.py
        """
    )
    
    parser.add_argument('--flow', type=str, help='Path to flow data CSV file')
    parser.add_argument('--rain', type=str, help='Path to rainfall data CSV file')
    parser.add_argument('--asset', type=str, help='Path to asset data CSV file (optional)')
    parser.add_argument('--flow-unit', type=str, default='gpm',
                       choices=['gpm', 'gpd', 'mgd', 'cfs', 'lps', 'lpm'],
                       help='Flow unit (default: gpm)')
    parser.add_argument('--rain-unit', type=str, default='in',
                       choices=['in', 'mm', 'cm'],
                       help='Rainfall unit (default: in)')
    parser.add_argument('--output', type=str, default='./output',
                       help='Output directory (default: ./output)')
    parser.add_argument('--interval', type=str, default='1H',
                       help='Resampling interval (default: 1H)')
    parser.add_argument('--streamlit', action='store_true',
                       help='Launch Streamlit web interface')
    
    args = parser.parse_args()
    
    if args.streamlit:
        run_streamlit_app()
    elif args.flow and args.rain:
        run_analysis(
            args.flow, args.rain, args.asset,
            args.flow_unit, args.rain_unit,
            args.output, args.interval
        )
    else:
        parser.print_help()
        print("\nError: --flow and --rain arguments are required (or use --streamlit)")
        sys.exit(1)


if __name__ == '__main__':
    main()
