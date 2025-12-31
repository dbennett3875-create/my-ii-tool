# Quick Start Guide - I&I Analysis Tool

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python test_ii_tool.py
   ```

## Usage Options

### 1. Command Line (Simplest)

```bash
# Basic usage with default settings
python ii_analysis.py --flow flow_data.csv --rain rainfall_data.csv

# With all options
python ii_analysis.py \
  --flow flow_data.csv \
  --rain rainfall_data.csv \
  --asset asset_data.csv \
  --flow-unit gpm \
  --rain-unit in \
  --interval 1H \
  --output ./my_results
```

### 2. Streamlit Web App (Most User-Friendly)

```bash
streamlit run ii_analysis.py --streamlit
```

Then:
1. Open browser to http://localhost:8501
2. Upload CSV files using the interface
3. Configure settings
4. Click "Run I&I Analysis"
5. Download Excel report

### 3. Python Script (Most Flexible)

```python
from ii_analysis import run_analysis

results = run_analysis(
    flow_file='flow_data.csv',
    rain_file='rainfall_data.csv',
    asset_file='asset_data.csv',
    flow_unit='gpm',
    rain_unit='in',
    output_dir='./output',
    resample_interval='1H'
)

# Access results
print(f"Detected {len(results['events'])} rain events")
print(results['basin_priority'])
```

## CSV File Requirements

### flow_data.csv
```csv
timestamp,flow_rate_gpm,basin_id
2024-01-01 00:00:00,450.2,Basin_A
2024-01-01 01:00:00,425.8,Basin_A
```

Required: `timestamp`, `flow_rate_gpm`
Optional: `basin_id` or `manhole_id`

### rainfall_data.csv
```csv
timestamp,rainfall_in
2024-01-01 00:00:00,0.0
2024-01-01 01:00:00,0.15
```

Required: `timestamp`, `rainfall_in`

### asset_data.csv (optional)
```csv
basin_id,area_acres,pipe_age_years,pipe_length_miles
Basin_A,125.5,35,8.2
```

## Common Command Examples

### Test with sample data:
```bash
python ii_analysis.py \
  --flow sample_flow_data.csv \
  --rain sample_rainfall_data.csv \
  --asset sample_asset_data.csv
```

### Process MGD data with mm rainfall:
```bash
python ii_analysis.py \
  --flow wwtp_flow.csv \
  --rain gauge_data.csv \
  --flow-unit mgd \
  --rain-unit mm
```

### High-resolution 15-minute analysis:
```bash
python ii_analysis.py \
  --flow meter_15min.csv \
  --rain rain_15min.csv \
  --interval 15min
```

## Output Files

After running, check the `output/` directory for:
- **Excel Report**: Comprehensive multi-sheet workbook
- **hydrograph.png**: Main flow vs rainfall plot
- **event_summary.png**: Event analysis charts
- **basin_comparison.png**: Basin rankings (if basins provided)

## Troubleshooting

**Problem**: "ModuleNotFoundError"
**Solution**: Run `pip install -r requirements.txt`

**Problem**: "No events detected"
**Solution**: Check that rainfall_in column has values > 0

**Problem**: "FileNotFoundError"
**Solution**: Ensure CSV files are in the current directory

**Problem**: Plots look strange
**Solution**: Check timestamp format is YYYY-MM-DD HH:MM:SS

## Support

- Read **README.md** for complete documentation
- Check code comments for technical details
- Review sample data files for format examples

## Tips

1. **Data Quality**: Ensure at least 5-10 rain events for meaningful analysis
2. **Baseline**: Need sufficient dry periods (>30% of dataset)
3. **Units**: Tool handles conversions automatically
4. **Time Zones**: Timestamps are converted to UTC internally
5. **Missing Data**: Short gaps (<2 hours) are auto-filled

Ready to analyze your I&I data! 🚀
