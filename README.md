# I&I Analysis Tool - Enhanced Edition

Professional Infiltration & Inflow analysis tool for wastewater systems with advanced features.

## Features

- **Rain Event Detection**: Automatic identification of rainfall events
- **Baseline Flow Calculation**: Statistical baseline using configurable percentile
- **Hydrograph Separation**: Baseflow/quickflow separation using recursive digital filter
- **RDII Coefficients**: Calculate R-value and unit I&I metrics
- **Anomaly Detection**: Multiple methods (IQR, Z-Score, Isolation Forest)
- **Basin Prioritization**: Rank basins by I&I severity and infrastructure age
- **Interactive Visualizations**: Plotly charts with zoom, pan, and hover details
- **Export Options**: Download results as CSV or JSON

## Installation

### Requirements
- Python 3.9 or higher
- pip (Python package manager)

### Setup Steps

1. **Download the tool files**
   - `ii_tool_v2.py` (main application)
   - `requirements.txt` (dependencies)
   - `realistic_flow_data.csv` (sample data)
   - `realistic_rainfall_data.csv` (sample data)
   - `realistic_asset_data.csv` (sample data)

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the tool**
   ```bash
   streamlit run ii_tool_v2.py
   ```

4. **Open in browser**
   - Automatically opens at http://localhost:8501
   - Or manually navigate to the URL shown in terminal

## Usage

### 1. Upload Data

**Required Files:**
- **Flow Data**: CSV with columns `timestamp`, `flow_rate_gpm`, `basin_id` (optional)
- **Rainfall Data**: CSV with columns `timestamp`, `rainfall_in`

**Optional File:**
- **Asset Data**: CSV with columns `basin_id`, `area_acres`, `pipe_age_years`, `pipe_length_miles`

### 2. Configure Settings

**Basic Settings:**
- Rain Event Threshold: Minimum rainfall to trigger event detection (default: 0.1 in)
- Baseline Percentile: Percentile for dry weather flow calculation (default: 10)

**Advanced Analysis (toggle on/off):**
- Hydrograph Separation: Separate baseflow from quickflow
- RDII Coefficients: Calculate rainfall-dependent I&I metrics
- Anomaly Detection: Identify unusual flow patterns

### 3. Run Analysis

Click "🔍 Run Analysis" button to:
- Detect rain events
- Calculate baseline flow
- Perform hydrograph separation (if enabled)
- Calculate RDII coefficients (if enabled)
- Detect anomalies (if enabled)
- Generate visualizations

### 4. Review Results

**Metrics Dashboard:**
- Baseline Flow (GPM)
- Number of Rain Events
- Total Rainfall (inches)
- Total I&I Volume (Million Gallons)
- RDII Coefficients (R-value, Unit I&I)

**Visualizations:**
- Main Hydrograph: Rainfall and flow over time with event highlighting
- Baseflow/Quickflow Separation: Shows components of total flow
- Event Summary: I&I volume by event
- Peak Flow vs Rainfall: Correlation analysis
- Basin Prioritization: Ranking by I&I severity

**Data Tables:**
- Rain Event Details: Start/end times, rainfall, peak flow, I&I volume
- Basin Comparison: Performance metrics and priority scores

### 5. Download Results

**Available Downloads:**
- Event Data (CSV): Detailed rain event information
- Full Results (JSON): Complete analysis with settings and calculations
- Processed Flow Data (CSV): Flow data with baseflow, quickflow, anomalies

## Data Format Examples

### Flow Data (flow_data.csv)
```csv
timestamp,flow_rate_gpm,basin_id
2024-01-01 00:00:00,450.2,Basin_A
2024-01-01 01:00:00,448.5,Basin_A
2024-01-01 02:00:00,445.8,Basin_A
```

### Rainfall Data (rainfall_data.csv)
```csv
timestamp,rainfall_in
2024-01-01 00:00:00,0.0
2024-01-01 01:00:00,0.0
2024-01-01 13:00:00,0.05
2024-01-01 14:00:00,0.35
```

### Asset Data (asset_data.csv)
```csv
basin_id,area_acres,pipe_age_years,pipe_length_miles
Basin_A,125.5,45,8.2
Basin_B,85.3,28,5.8
```

## Sharing the Tool

### Option 1: Simple File Sharing
Send these files via email or cloud storage:
- `ii_tool_v2.py`
- `requirements.txt`
- `README.md`
- Sample data files (optional)

### Option 2: Network Access
Share your local instance on your network:
- Run: `streamlit run ii_tool_v2.py --server.address 0.0.0.0`
- Share the Network URL shown in terminal (e.g., http://192.168.1.148:8501)
- Others on same network can access directly

### Option 3: Cloud Deployment
Deploy to Streamlit Cloud (free):
1. Create GitHub repository
2. Upload all files
3. Connect to https://streamlit.io/cloud
4. Get public URL to share

## Troubleshooting

**Problem**: Dependencies won't install
**Solution**: Ensure Python 3.9+ is installed, try `pip3 install -r requirements.txt`

**Problem**: Port already in use
**Solution**: Run with different port: `streamlit run ii_tool_v2.py --server.port 8502`

**Problem**: Charts not displaying
**Solution**: Ensure plotly is installed: `pip install plotly`

**Problem**: File upload fails
**Solution**: Check CSV format matches examples, ensure timestamps are formatted correctly

## Technical Details

**Hydrograph Separation Method:**
- Recursive digital filter algorithm
- Adjustable filter parameter (α) controls baseflow response speed
- Default α=0.925 suitable for most wastewater systems

**RDII Calculation:**
- R-value: Ratio of I&I volume to rainfall volume
- Unit I&I: Gallons per inch of rain per acre of drainage area
- Requires asset data with drainage area

**Anomaly Detection Methods:**
- IQR: Interquartile range method (1.5 × IQR)
- Z-Score: Standard deviation method (>3σ)
- Isolation Forest: Machine learning approach

## Support

For questions or issues:
1. Check sample data files for format reference
2. Review this README for common solutions
3. Verify all dependencies are installed correctly

## Version

Version 2.0 - Enhanced Edition
- Advanced hydrograph separation
- RDII coefficient calculations
- Multi-method anomaly detection
- Basin prioritization
- Comprehensive export options
