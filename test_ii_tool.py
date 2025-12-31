#!/usr/bin/env python3
"""
Quick test script for I&I Analysis Tool
Run this to verify the installation and test with sample data
"""

import sys
from pathlib import Path

def test_ii_analysis():
    """Test the I&I analysis tool with sample data."""
    
    print("=" * 80)
    print("TESTING I&I ANALYSIS TOOL")
    print("=" * 80)
    
    # Check if sample files exist
    files_needed = {
        'flow': 'sample_flow_data.csv',
        'rain': 'sample_rainfall_data.csv',
        'asset': 'sample_asset_data.csv'
    }
    
    print("\n1. Checking sample data files...")
    for name, filename in files_needed.items():
        if Path(filename).exists():
            print(f"   ✓ {filename} found")
        else:
            print(f"   ✗ {filename} NOT FOUND")
            print(f"   Please ensure sample files are in the current directory")
            return False
    
    # Import the module
    print("\n2. Importing ii_analysis module...")
    try:
        from ii_analysis import run_analysis
        print("   ✓ Module imported successfully")
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        print("   Please install required packages: pip install -r requirements.txt")
        return False
    
    # Run analysis
    print("\n3. Running analysis on sample data...")
    try:
        results = run_analysis(
            flow_file='sample_flow_data.csv',
            rain_file='sample_rainfall_data.csv',
            asset_file='sample_asset_data.csv',
            flow_unit='gpm',
            rain_unit='in',
            output_dir='./test_output',
            resample_interval='1H'
        )
        print("   ✓ Analysis completed successfully!")
    except Exception as e:
        print(f"   ✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check outputs
    print("\n4. Verifying outputs...")
    output_dir = Path('./test_output')
    
    if output_dir.exists():
        files = list(output_dir.glob('*'))
        print(f"   ✓ Output directory created with {len(files)} files")
        
        # Check for key files
        has_excel = any(f.suffix == '.xlsx' for f in files)
        has_plots = any(f.suffix == '.png' for f in files)
        
        if has_excel:
            print("   ✓ Excel report generated")
        else:
            print("   ✗ Excel report missing")
            
        if has_plots:
            print(f"   ✓ {sum(1 for f in files if f.suffix == '.png')} plot(s) generated")
        else:
            print("   ✗ Plots missing")
    else:
        print("   ✗ Output directory not created")
        return False
    
    # Display results summary
    print("\n5. Results Summary:")
    print(f"   - Events detected: {len(results['events'])}")
    print(f"   - Data points: {len(results['dataframe'])}")
    
    if results['events']:
        total_rain = sum(e['total_rainfall_in'] for e in results['events'])
        total_ii = sum(e.get('total_ii_volume_gal', 0) for e in results['events'])
        print(f"   - Total rainfall: {total_rain:.2f} inches")
        print(f"   - Total I&I volume: {total_ii:,.0f} gallons")
        
        if total_rain > 0:
            print(f"   - Average I&I rate: {total_ii/total_rain:,.0f} gal/inch")
    
    if not results['basin_priority'].empty:
        print(f"   - Basins analyzed: {len(results['basin_priority'])}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY! ✓")
    print("=" * 80)
    print(f"\nCheck the ./test_output/ directory for results")
    print("\nNext steps:")
    print("  1. Review the Excel report and plots in ./test_output/")
    print("  2. Run with your own data using the command line or Streamlit interface")
    print("  3. Read README.md for detailed documentation")
    
    return True


if __name__ == '__main__':
    success = test_ii_analysis()
    sys.exit(0 if success else 1)
