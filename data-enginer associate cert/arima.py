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

def create_complete_timeseries_2021_2025(ts_original, forecast_df):
    """Create complete time series from 2021 to 2025 (historical + forecasted)"""
    print("\n📈 Creating Complete 2021-2025 Time Series...")
    
    # Historical data (2021-2024)
    historical_df = pd.DataFrame({
        'Date': ts_original.index,
        'Student_Counts': ts_original.values,
        'Type': 'Historical',
        'Lower_CI': ts_original.values,  # Historical data has no uncertainty
        'Upper_CI': ts_original.values
    })
    
    # Forecasted data (2025)
    forecast_clean = pd.DataFrame({
        'Date': forecast_df['Date'],
        'Student_Counts': forecast_df['Forecast'],
        'Type': 'Forecasted',
        'Lower_CI': forecast_df['Lower_CI'],
        'Upper_CI': forecast_df['Upper_CI']
    })
    
    # Combine historical and forecasted data
    complete_ts = pd.concat([historical_df, forecast_clean], ignore_index=True)
    complete_ts = complete_ts.sort_values('Date').reset_index(drop=True)
    
    # Add additional columns for analysis
    complete_ts['Year'] = complete_ts['Date'].dt.year
    complete_ts['Month'] = complete_ts['Date'].dt.month
    complete_ts['Month_Name'] = complete_ts['Date'].dt.strftime('%B')
    
    print(f"✅ Complete time series created: {len(complete_ts)} data points")
    print(f"📊 Historical data: {len(historical_df)} points (2021-2024)")
    print(f"🔮 Forecasted data: {len(forecast_clean)} points (2025)")
    
    return complete_ts

def create_prediction_table_with_confidence(forecast_df):
    """Create a clean prediction table with confidence ratings"""
    print("\n📋 Creating Prediction Table with Confidence Analysis...")
    
    # Calculate confidence metrics
    prediction_table = forecast_df.copy()
    
    # Calculate confidence interval width
    prediction_table['CI_Width'] = prediction_table['Upper_CI'] - prediction_table['Lower_CI']
    
    # Calculate relative confidence (smaller CI width = higher confidence)
    prediction_table['Relative_CI_Width'] = (prediction_table['CI_Width'] / prediction_table['Forecast']) * 100
    
    # Assign confidence ratings based on relative CI width
    def assign_confidence_rating(rel_width):
        if rel_width <= 10:
            return "Very High"
        elif rel_width <= 20:
            return "High" 
        elif rel_width <= 35:
            return "Medium"
        elif rel_width <= 50:
            return "Low"
        else:
            return "Very Low"
    
    prediction_table['Confidence_Rating'] = prediction_table['Relative_CI_Width'].apply(assign_confidence_rating)
    
    # Create clean display table
    display_table = pd.DataFrame({
        'Month': prediction_table['Month'],
        'Predicted_Students': prediction_table['Forecast'].round(0).astype(int),
        'Lower_Bound': prediction_table['Lower_CI'].round(0).astype(int),
        'Upper_Bound': prediction_table['Upper_CI'].round(0).astype(int),
        'Confidence_Rating': prediction_table['Confidence_Rating'],
        'Uncertainty_Range': prediction_table['CI_Width'].round(0).astype(int),
        'Uncertainty_Percent': prediction_table['Relative_CI_Width'].round(1)
    })
    
    print("✅ Prediction table created with confidence analysis")
    
    return display_table, prediction_table

def visualize_complete_timeline_2021_2025(complete_ts):
    """Create comprehensive visualization of 2021-2025 timeline"""
    print("\n📊 Creating 2021-2025 Timeline Visualization...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 15))
    
    # Plot 1: Complete timeline with historical + forecasted data
    historical_data = complete_ts[complete_ts['Type'] == 'Historical']
    forecast_data = complete_ts[complete_ts['Type'] == 'Forecasted']
    
    axes[0].plot(historical_data['Date'], historical_data['Student_Counts'], 
                 marker='o', linewidth=2, label='Historical Data (2021-2024)', color='blue')
    
    axes[0].plot(forecast_data['Date'], forecast_data['Student_Counts'], 
                 marker='s', linewidth=3, label='Forecasted Data (2025)', color='red', markersize=8)
    
    # Add confidence intervals for 2025
    axes[0].fill_between(forecast_data['Date'], 
                        forecast_data['Lower_CI'], 
                        forecast_data['Upper_CI'],
                        alpha=0.3, color='red', label='95% Confidence Interval')
    
    axes[0].set_title('Complete Student Enrollment Timeline (2021-2025)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Student Count')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add vertical line to separate historical from forecasted
    transition_date = forecast_data['Date'].iloc[0]
    axes[0].axvline(x=transition_date, color='gray', linestyle='--', alpha=0.7, 
                   label='Historical/Forecast Boundary')
    
    # Plot 2: Year-over-year comparison (2021-2025)
    pivot_complete = complete_ts.pivot_table(index='Month', columns='Year', 
                                           values='Student_Counts', aggfunc='mean')
    
    colors = ['blue', 'green', 'orange', 'purple', 'red']
    for i, year in enumerate(pivot_complete.columns):
        line_style = '-' if year <= 2024 else '--'  # Solid for historical, dashed for forecast
        line_width = 2 if year <= 2024 else 3
        axes[1].plot(pivot_complete.index, pivot_complete[year], 
                    marker='o', label=f'{year}', color=colors[i % len(colors)],
                    linestyle=line_style, linewidth=line_width, markersize=6)
    
    axes[1].set_title('Year-over-Year Enrollment Comparison (2021-2025)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Student Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                            'Jul','Aug','Sep','Oct','Nov','Dec'])
    
    # Plot 3: Focus on the dramatic seasonal pattern (all years)
    march_data = complete_ts[complete_ts['Month'] == 3]
    jan_feb_data = complete_ts[complete_ts['Month'].isin([1, 2])]
    
    # Bar chart showing March vs Jan-Feb average by year
    years = sorted(complete_ts['Year'].unique())
    march_values = []
    jan_feb_avg = []
    
    for year in years:
        march_val = complete_ts[(complete_ts['Year'] == year) & (complete_ts['Month'] == 3)]['Student_Counts'].iloc[0]
        jan_feb_val = complete_ts[(complete_ts['Year'] == year) & (complete_ts['Month'].isin([1, 2]))]['Student_Counts'].mean()
        march_values.append(march_val)
        jan_feb_avg.append(jan_feb_val)
    
    x_pos = np.arange(len(years))
    width = 0.35
    
    bars1 = axes[2].bar(x_pos - width/2, jan_feb_avg, width, label='Jan-Feb Average', 
                       color='lightblue', edgecolor='navy')
    bars2 = axes[2].bar(x_pos + width/2, march_values, width, label='March Enrollment', 
                       color='red', alpha=0.7, edgecolor='darkred')
    
    # Add value labels on bars
    for i, (jan_feb, march) in enumerate(zip(jan_feb_avg, march_values)):
        axes[2].text(i - width/2, jan_feb + 500, f'{jan_feb:.0f}', ha='center', va='bottom', fontsize=8)
        axes[2].text(i + width/2, march + 1000, f'{march:.0f}', ha='center', va='bottom', fontsize=8)
    
    axes[2].set_title('Seasonal Surge Analysis: Jan-Feb vs March (2021-2025)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Year')
    axes[2].set_ylabel('Student Count')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(years)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add forecast indicator for 2025
    axes[2].text(len(years)-1, max(march_values) * 0.8, 'FORECAST', 
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Complete timeline visualization created!")

def display_summary_statistics(complete_ts, display_table):
    """Display comprehensive summary statistics"""
    print("\n📊 COMPREHENSIVE ENROLLMENT ANALYSIS SUMMARY")
    print("="*70)
    
    # Historical vs Forecasted summary
    historical_data = complete_ts[complete_ts['Type'] == 'Historical']
    forecast_data = complete_ts[complete_ts['Type'] == 'Forecasted']
    
    print(f"\n🔍 HISTORICAL DATA SUMMARY (2021-2024):")
    print(f"   Total observations: {len(historical_data)}")
    print(f"   Average monthly enrollment: {historical_data['Student_Counts'].mean():,.0f}")
    print(f"   Minimum enrollment: {historical_data['Student_Counts'].min():,.0f}")
    print(f"   Maximum enrollment: {historical_data['Student_Counts'].max():,.0f}")
    print(f"   Standard deviation: {historical_data['Student_Counts'].std():,.0f}")
    
    print(f"\n🔮 FORECASTED DATA SUMMARY (2025):")
    print(f"   Predicted total annual enrollment: {forecast_data['Student_Counts'].sum():,.0f}")
    print(f"   Average monthly enrollment: {forecast_data['Student_Counts'].mean():,.0f}")
    print(f"   Predicted minimum month: {forecast_data['Student_Counts'].min():,.0f}")
    print(f"   Predicted maximum month: {forecast_data['Student_Counts'].max():,.0f}")
    
    # Seasonal pattern analysis
    historical_march = historical_data[historical_data['Month'] == 3]['Student_Counts'].mean()
    forecast_march = forecast_data[forecast_data['Month'] == 3]['Student_Counts'].iloc[0]
    
    historical_jan_feb = historical_data[historical_data['Month'].isin([1, 2])]['Student_Counts'].mean()
    forecast_jan_feb = forecast_data[forecast_data['Month'].isin([1, 2])]['Student_Counts'].mean()
    
    print(f"\n🌊 SEASONAL PATTERN ANALYSIS:")
    print(f"   Historical March average: {historical_march:,.0f}")
    print(f"   Forecasted March 2025: {forecast_march:,.0f}")
    print(f"   Historical Jan-Feb average: {historical_jan_feb:,.0f}")
    print(f"   Forecasted Jan-Feb 2025: {forecast_jan_feb:,.0f}")
    print(f"   Historical seasonal surge: {((historical_march - historical_jan_feb) / historical_jan_feb * 100):.0f}%")
    print(f"   Forecasted seasonal surge: {((forecast_march - forecast_jan_feb) / forecast_jan_feb * 100):.0f}%")
    
    # Confidence analysis
    high_confidence = len(display_table[display_table['Confidence_Rating'].isin(['Very High', 'High'])])
    medium_confidence = len(display_table[display_table['Confidence_Rating'] == 'Medium'])
    low_confidence = len(display_table[display_table['Confidence_Rating'].isin(['Low', 'Very Low'])])
    
    print(f"\n🎯 FORECAST CONFIDENCE ANALYSIS:")
    print(f"   High confidence predictions: {high_confidence}/12 months")
    print(f"   Medium confidence predictions: {medium_confidence}/12 months")
    print(f"   Low confidence predictions: {low_confidence}/12 months")
    print(f"   Average uncertainty range: ±{display_table['Uncertainty_Range'].mean():,.0f} students")
    print(f"   Average uncertainty percentage: ±{display_table['Uncertainty_Percent'].mean():.1f}%")
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
    try:
        forecast_results, model, complete_timeseries, prediction_table = run_complete_analysis(base_data)
        
        # Show results
        if forecast_results is not None:
            print("✅ Analysis completed successfully!")
            
            # ALWAYS display prediction table - even if the fancy one fails
            print("\n📋 2025 ENROLLMENT PREDICTIONS:")
            print("="*80)
            
            if prediction_table is not None:
                print("Using detailed prediction table with confidence analysis:")
                print(prediction_table.to_string(index=False))
            else:
                # Fallback: create simple prediction table
                print("Using basic prediction table:")
                simple_table = pd.DataFrame({
                    'Month': forecast_results['Month'],
                    'Predicted_Students': forecast_results['Forecast'].round(0).astype(int),
                    'Lower_Bound': forecast_results['Lower_CI'].round(0).astype(int),
                    'Upper_Bound': forecast_results['Upper_CI'].round(0).astype(int)
                })
                print(simple_table.to_string(index=False))
            
            # ALWAYS display timeseries data - even if visualization fails
            print(f"\n📊 COMPLETE 2021-2025 TIME SERIES DATA:")
            print("="*60)
            
            if complete_timeseries is not None:
                print(f"Complete dataset with {len(complete_timeseries)} data points:")
                # Show first few historical points
                print("\nFirst 10 data points (Historical):")
                print(complete_timeseries[['Date', 'Student_Counts', 'Type', 'Year']].head(10).to_string(index=False))
                
                # Show last few forecast points
                print("\nLast 12 data points (2025 Forecasts):")
                print(complete_timeseries[['Date', 'Student_Counts', 'Type', 'Year']].tail(12).to_string(index=False))
                
            else:
                # Fallback: create simple timeseries from available data
                print("Creating basic timeseries from available data:")
                
                # Get historical data from base_data
                try:
                    # Create date column for historical data
                    base_data_copy = base_data.copy()
                    
                    # Convert month names to numbers for date creation
                    month_mapping = {
                        'January': 1, 'Jan': 1, 'February': 2, 'Feb': 2,
                        'March': 3, 'Mar': 3, 'April': 4, 'Apr': 4,
                        'May': 5, 'June': 6, 'Jun': 6,
                        'July': 7, 'Jul': 7, 'August': 8, 'Aug': 8,
                        'September': 9, 'Sep': 9, 'October': 10, 'Oct': 10,
                        'November': 11, 'Nov': 11, 'December': 12, 'Dec': 12
                    }
                    
                    if base_data_copy['Reporting_Month'].dtype == 'object':
                        base_data_copy['Month_Num'] = base_data_copy['Reporting_Month'].map(month_mapping)
                    else:
                        base_data_copy['Month_Num'] = base_data_copy['Reporting_Month']
                    
                    base_data_copy['Date'] = pd.to_datetime({
                        'year': base_data_copy['Reporting_Year'],
                        'month': base_data_copy['Month_Num'],
                        'day': 1
                    })
                    
                    historical_df = pd.DataFrame({
                        'Date': base_data_copy['Date'],
                        'Student_Counts': base_data_copy['Student_Counts'],
                        'Type': 'Historical'
                    })
                    
                    # Add forecast data
                    forecast_df = pd.DataFrame({
                        'Date': forecast_results['Date'],
                        'Student_Counts': forecast_results['Forecast'],
                        'Type': 'Forecasted'
                    })
                    
                    # Combine
                    simple_timeseries = pd.concat([historical_df, forecast_df], ignore_index=True)
                    simple_timeseries = simple_timeseries.sort_values('Date').reset_index(drop=True)
                    
                    print(f"Basic timeseries with {len(simple_timeseries)} data points:")
                    print("First 10 points (Historical):")
                    print(simple_timeseries.head(10).to_string(index=False))
                    print("\nLast 12 points (2025 Forecasts):")
                    print(simple_timeseries.tail(12).to_string(index=False))
                    
                except Exception as e:
                    print(f"Could not create fallback timeseries: {e}")
                    print("Showing just the forecast data:")
                    print(forecast_results[['Date', 'Month', 'Forecast', 'Lower_CI', 'Upper_CI']].to_string(index=False))
            
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
            
            print(f"\n💾 AVAILABLE DATA OBJECTS:")
            if complete_timeseries is not None:
                print(f"   📈 complete_timeseries: Full 2021-2025 timeline ({len(complete_timeseries)} data points)")
            if prediction_table is not None:
                print(f"   📋 prediction_table: Clean 2025 predictions with confidence ratings")
            print(f"   🔮 forecast_results: Detailed 2025 forecasts with confidence intervals")
            if model is not None:
                print(f"   🤖 model: Fitted SARIMA model for further analysis")
                
        else:
            print("❌ Analysis failed - model could not be fitted")
            
    except ValueError as e:
        print(f"❌ Error during analysis: {e}")
        print("🔍 This might be due to data format issues. Let's check your data:")
        print("\nFirst 5 rows of your data:")
        print(base_data.head())
        print("\nData types:")
        print(base_data.dtypes)
        print("\nUnique values in Reporting_Month:")
        print(base_data['Reporting_Month'].unique())
        
else:
    print("❌ Cannot run analysis - please check column names")

print("🎉 SARIMA Analysis Complete!")
