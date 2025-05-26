# =========================================================================
# CORE SARIMA STUDENT ENROLLMENT FORECASTING FOR DATABRICKS
# (Data loading handled separately by user)
# 
# EXPECTED DATA STRUCTURE:
# - Reporting_Year: 2021, 2022, 2023, 2024
# - Reporting_Month: "January", "Feb", "March", etc.
# - Student_Counts: numerical values (e.g., 600, 1000, 37000)
# =========================================================================

# Install required packages
%pip install pmdarima

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Try to import auto_arima
try:
    from pmdarima import auto_arima
    AUTO_ARIMA_AVAILABLE = True
    print("✅ pmdarima imported successfully")
except ImportError:
    print("❌ pmdarima not available")
    AUTO_ARIMA_AVAILABLE = False

# Enable inline plotting for Databricks
%matplotlib inline

# =========================================================================
# CORE SARIMA ANALYSIS FUNCTIONS
# =========================================================================

def prepare_student_data(df):
    """Prepare student enrollment data for time series analysis"""
    print("📊 Data Preview:")
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    
    # Check data types
    print(f"\nData types:")
    print(f"Reporting_Year: {df['Reporting_Year'].dtype}")
    print(f"Reporting_Month: {df['Reporting_Month'].dtype}")
    
    # Handle month names (convert to numbers)
    month_mapping = {
        'January': 1, 'Jan': 1,
        'February': 2, 'Feb': 2,
        'March': 3, 'Mar': 3,
        'April': 4, 'Apr': 4,
        'May': 5,
        'June': 6, 'Jun': 6,
        'July': 7, 'Jul': 7,
        'August': 8, 'Aug': 8,
        'September': 9, 'Sep': 9, 'Sept': 9,
        'October': 10, 'Oct': 10,
        'November': 11, 'Nov': 11,
        'December': 12, 'Dec': 12
    }
    
    # Convert month names to numbers if needed
    if df['Reporting_Month'].dtype == 'object':  # If month column contains text
        print("🔄 Converting month names to numbers...")
        df['Month_Num'] = df['Reporting_Month'].map(month_mapping)
        
        # Check for any unmapped months
        unmapped = df[df['Month_Num'].isna()]['Reporting_Month'].unique()
        if len(unmapped) > 0:
            print(f"⚠️ Warning: Could not map these months: {unmapped}")
            print("Available mappings:", list(month_mapping.keys()))
            # Fill unmapped values with a default or show error
            print("First few unmapped values:")
            print(df[df['Month_Num'].isna()][['Reporting_Year', 'Reporting_Month']].head())
        
        # Use the numeric month for datetime creation
        month_col = 'Month_Num'
    else:
        month_col = 'Reporting_Month'
        df['Month_Num'] = df['Reporting_Month']  # Copy for consistency
    
    # Debug: Check what we're working with
    print(f"\nUsing month column: {month_col}")
    print(f"Sample values:")
    print(f"Years: {df['Reporting_Year'].head().tolist()}")
    print(f"Months: {df[month_col].head().tolist()}")
    
    # More robust datetime creation
    try:
        # Method 1: Using pd.to_datetime with explicit format
        df['Date'] = pd.to_datetime({
            'year': df['Reporting_Year'],
            'month': df[month_col],
            'day': 1
        })
        print("✅ DateTime created successfully using method 1")
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
        try:
            # Method 2: String concatenation approach
            df['date_string'] = df['Reporting_Year'].astype(str) + '-' + df[month_col].astype(str) + '-01'
            df['Date'] = pd.to_datetime(df['date_string'], format='%Y-%m-%d')
            print("✅ DateTime created successfully using method 2")
            
        except Exception as e2:
            print(f"❌ Method 2 also failed: {e2}")
            print("Showing sample data for debugging:")
            print(df[['Reporting_Year', 'Reporting_Month', month_col]].head(10))
            raise ValueError("Could not create datetime column. Please check your data format.")
    
    # Sort by date and reset index
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Create time series
    ts = df.set_index('Date')['Student_Counts']
    
    print(f"\n📅 Time series created: {len(ts)} observations")
    print(f"From: {ts.index[0]} to {ts.index[-1]}")
    
    return ts, df

def explore_enrollment_patterns(ts, df):
    """Explore enrollment patterns with focus on seasonal jumps"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time series plot
    axes[0,0].plot(ts.index, ts.values, marker='o', linewidth=2)
    axes[0,0].set_title('Student Enrollment Over Time', fontsize=12, fontweight='bold')
    axes[0,0].set_ylabel('Student Count')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Monthly patterns
    df_temp = df[['Date', 'Student_Counts']].copy()
    df_temp['Month_Num'] = df_temp['Date'].dt.month
    df_temp['Month_Name'] = df_temp['Date'].dt.strftime('%b')
    
    monthly_avg = df_temp.groupby(['Month_Num', 'Month_Name'])['Student_Counts'].mean().reset_index()
    monthly_avg = monthly_avg.sort_values('Month_Num')
    
    axes[0,1].bar(monthly_avg['Month_Name'], monthly_avg['Student_Counts'], 
                  color='skyblue', edgecolor='navy')
    axes[0,1].set_title('Average Monthly Enrollment', fontsize=12, fontweight='bold')
    axes[0,1].set_ylabel('Average Student Count')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # Year over year comparison
    pivot_data = df.pivot_table(index=df['Date'].dt.month, 
                               columns=df['Date'].dt.year, 
                               values='Student_Counts', 
                               aggfunc='mean')
    
    for year in pivot_data.columns:
        axes[1,0].plot(pivot_data.index, pivot_data[year], marker='o', label=f'{year}')
    
    axes[1,0].set_title('Year-over-Year Monthly Comparison', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Month')
    axes[1,0].set_ylabel('Student Count')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xticks(range(1, 13))
    axes[1,0].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                              'Jul','Aug','Sep','Oct','Nov','Dec'])
    
    # Seasonal decomposition
    try:
        decomp = seasonal_decompose(ts, model='multiplicative', period=12)
        axes[1,1].plot(decomp.seasonal[:12], marker='o', linewidth=2)
        axes[1,1].set_title('Seasonal Pattern', fontsize=12, fontweight='bold')
        axes[1,1].set_xlabel('Month')
        axes[1,1].set_xticks(range(12))
        axes[1,1].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                                  'Jul','Aug','Sep','Oct','Nov','Dec'])
        axes[1,1].grid(True, alpha=0.3)
    except:
        axes[1,1].text(0.5, 0.5, 'Seasonal decomposition\nnot available', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
    
    plt.tight_layout()
    plt.show()
    
    # Print key statistics
    print("\n📈 KEY ENROLLMENT INSIGHTS:")
    print("="*50)
    
    # Monthly statistics
    monthly_stats = df_temp.groupby(['Month_Num', 'Month_Name'])['Student_Counts'].agg(['mean', 'min', 'max']).round(0)
    monthly_stats = monthly_stats.sort_values('Month_Num')
    
    print("Monthly Statistics:")
    for (month_num, month_name), row in monthly_stats.iterrows():
        print(f"  {month_name}: Avg={row['mean']:>8.0f}, Min={row['min']:>8.0f}, Max={row['max']:>8.0f}")
    
    # Identify dramatic changes
    changes = ts.diff().abs()
    if len(changes.dropna()) > 0:
        threshold = changes.quantile(0.8)
        large_changes = changes[changes > threshold].dropna()
        
        if len(large_changes) > 0:
            print(f"\n🚨 Largest Month-to-Month Changes:")
            for date, change in large_changes.head(5).items():
                prev_date = date - pd.DateOffset(months=1)
                if prev_date in ts.index:
                    print(f"  {prev_date.strftime('%b %Y')} → {date.strftime('%b %Y')}: "
                          f"{ts[prev_date]:>6.0f} → {ts[date]:>6.0f} (+{change:>6.0f})")

def handle_extreme_seasonality(ts):
    """Handle extreme seasonal variations with transformations"""
    print("\n🔄 Testing transformations for extreme seasonality...")
    
    # Test different transformations
    ts_log = np.log(ts + 1)
    ts_sqrt = np.sqrt(ts)
    
    # Calculate coefficient of variation for each
    cv_original = ts.std() / ts.mean()
    cv_log = ts_log.std() / ts_log.mean()
    cv_sqrt = ts_sqrt.std() / ts_sqrt.mean()
    
    print(f"Coefficient of Variation:")
    print(f"  Original: {cv_original:.3f}")
    print(f"  Log:      {cv_log:.3f}")
    print(f"  Sqrt:     {cv_sqrt:.3f}")
    
    # Choose best transformation
    if cv_log < cv_original and cv_log < cv_sqrt:
        print("✅ Using LOG transformation")
        return ts_log, 'log'
    elif cv_sqrt < cv_original:
        print("✅ Using SQRT transformation")
        return ts_sqrt, 'sqrt'
    else:
        print("✅ Using ORIGINAL data")
        return ts, 'none'

def fit_sarima_model(ts_transformed, transformation='none'):
    """Fit SARIMA model for academic enrollment data"""
    print("\n🎯 Fitting SARIMA model...")
    
    best_model = None
    best_aic = float('inf')
    
    # Try different SARIMA configurations
    configs = [
        ((1,1,1), (1,1,1,12)),
        ((2,1,1), (1,1,1,12)),
        ((1,1,2), (1,1,1,12)),
        ((0,1,1), (0,1,1,12)),
        ((1,1,0), (1,1,0,12)),
    ]
    
    for order, seasonal_order in configs:
        try:
            model = ARIMA(ts_transformed, order=order, seasonal_order=seasonal_order)
            fitted = model.fit()
            
            print(f"SARIMA{order}x{seasonal_order}[12] - AIC: {fitted.aic:.2f}")
            
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_model = fitted
                best_config = (order, seasonal_order)
        except Exception as e:
            print(f"SARIMA{order}x{seasonal_order}[12] - Failed")
    
    if best_model:
        print(f"\n✅ Best model: SARIMA{best_config[0]}x{best_config[1]}[12]")
        print(f"   AIC: {best_aic:.2f}")
        best_model.transformation = transformation
        
    return best_model

def forecast_2025_enrollment(model, ts_original):
    """Generate 2025 enrollment forecasts"""
    print("\n🔮 Generating 2025 forecasts...")
    
    # Generate forecast
    forecast = model.forecast(steps=12)
    conf_int = model.get_forecast(steps=12).conf_int()
    
    # Transform back to original scale if needed
    if hasattr(model, 'transformation'):
        if model.transformation == 'log':
            forecast = np.exp(forecast) - 1
            conf_int = np.exp(conf_int) - 1
        elif model.transformation == 'sqrt':
            forecast = forecast ** 2
            conf_int = conf_int ** 2
    
    # Create forecast DataFrame
    dates_2025 = pd.date_range('2025-01-01', '2025-12-01', freq='MS')
    
    forecast_df = pd.DataFrame({
        'Date': dates_2025,
        'Month': dates_2025.strftime('%B'),
        'Forecast': forecast.round(0),
        'Lower_CI': conf_int.iloc[:, 0].round(0),
        'Upper_CI': conf_int.iloc[:, 1].round(0)
    })
    
    # Plot results
    plt.figure(figsize=(15, 8))
    
    # Historical data
    plt.plot(ts_original.index, ts_original.values, 
             label='Historical Data', linewidth=2, marker='o')
    
    # Forecast
    plt.plot(dates_2025, forecast, 
             label='2025 Forecast', linewidth=3, color='red', marker='s', markersize=8)
    
    # Confidence intervals
    plt.fill_between(dates_2025, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                    alpha=0.3, color='red', label='95% Confidence Interval')
    
    plt.title('Student Enrollment Forecast for 2025', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Student Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("\n🎓 2025 MONTHLY ENROLLMENT FORECASTS:")
    print("="*60)
    
    for _, row in forecast_df.iterrows():
        print(f"{row['Month']:>10}: {row['Forecast']:>8.0f} students "
              f"(Range: {row['Lower_CI']:>6.0f} - {row['Upper_CI']:>6.0f})")
    
    total_forecast = forecast_df['Forecast'].sum()
    print("="*60)
    print(f"{'Total 2025':>10}: {total_forecast:>8.0f} students")
    
    return forecast_df

def run_complete_analysis(base_data):
    """Run complete SARIMA analysis"""
    
    print("🚀 Starting Complete SARIMA Analysis for Student Enrollment")
    print("="*70)
    
    # Step 1: Prepare data
    ts, df = prepare_student_data(base_data)
    
    # Step 2: Explore patterns
    explore_enrollment_patterns(ts, df)
    
    # Step 3: Handle extreme seasonality
    ts_transformed, transformation = handle_extreme_seasonality(ts)
    
    # Step 4: Fit SARIMA model
    sarima_model = fit_sarima_model(ts_transformed, transformation)
    
    if sarima_model:
        # Step 5: Generate forecasts
        forecast_2025 = forecast_2025_enrollment(sarima_model, ts)
        
        print(f"\n🎉 Analysis Complete!")
        print(f"📊 Model: SARIMA with {transformation} transformation")
        
        return forecast_2025, sarima_model
    else:
        print("❌ Model fitting failed")
        return None, None

# =========================================================================
# EXECUTION WITH SPARK DATAFRAME
# =========================================================================

# Your data loading method (Spark DataFrame)
file_path = "abfs://folder2@printxpp.dfs.core.windows.net/scratch/opts/data.csv"
base_data_spark = spark.read.option("header","true").option("inferSchema","true").csv(file_path)

# Convert Spark DataFrame to Pandas DataFrame for SARIMA analysis
print("🔄 Converting Spark DataFrame to Pandas...")
base_data = base_data_spark.toPandas()

print("✅ Data conversion complete!")
print(f"📊 Data shape: {base_data.shape}")
print(f"📋 Columns: {list(base_data.columns)}")

# Verify expected columns exist
expected_columns = ['Reporting_Year', 'Reporting_Month', 'Student_Counts']
missing_columns = [col for col in expected_columns if col not in base_data.columns]

if missing_columns:
    print(f"⚠️ Warning: Missing expected columns: {missing_columns}")
    print("📋 Available columns:", list(base_data.columns))
    print("💡 Update column names in the prepare_student_data function if needed")
else:
    print("✅ All expected columns found!")
    
    # Show sample of your data structure
    print(f"\n📊 Sample of your data:")
    print(base_data[['Reporting_Year', 'Reporting_Month', 'Student_Counts']].head())

# Run complete SARIMA analysis
if not missing_columns:
    forecast_results, model = run_complete_analysis(base_data)
    
    # Show results (CSV export removed)
    if forecast_results is not None:
        print("✅ Analysis completed successfully!")
        
        # Show key results for your academic calendar pattern
        print("\n🎯 KEY RESULTS FOR YOUR ENROLLMENT PATTERN:")
        print("="*55)
        
        jan_forecast = forecast_results[forecast_results['Month'] == 'January']['Forecast'].iloc[0]
        feb_forecast = forecast_results[forecast_results['Month'] == 'February']['Forecast'].iloc[0]
        mar_forecast = forecast_results[forecast_results['Month'] == 'March']['Forecast'].iloc[0]
        
        print(f"📅 January 2025:  {jan_forecast:>8.0f} students")
        print(f"📅 February 2025: {feb_forecast:>8.0f} students")
        print(f"📅 March 2025:    {mar_forecast:>8.0f} students")
        print(f"📈 Feb→Mar surge: {((mar_forecast - feb_forecast) / feb_forecast * 100):>7.0f}%")
        
        # Show forecast results DataFrame
        print(f"\n📊 Complete 2025 Forecast Results:")
        print(forecast_results)
        
else:
    print("❌ Cannot run analysis - please check column names")

print("🎉 SARIMA Analysis Complete!")
