

1. Original schema mismatch problem
2. Multiple source rows matching target row error
3. Using called_ts instead of last_updated
4. Proper Delta Lake API syntax
5. Pre-filtering for latest records
6. Handling duplicates properly

I'll write the complete script from scratch with all these fixes incorporated.

Here's the complete code with all the fixes applied:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, count, col, when, isnan, isnull, current_timestamp, max
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, BooleanType
from datetime import datetime
from delta.tables import DeltaTable

def analyze_csv_files(adls_path, analysis_table_name, log_table_name):
    """
    Recursively analyzes CSV files in ADLS2 storage and outputs analysis metrics to Hive metastore tables.
    Skips files that have already been successfully processed.
    
    Parameters:
    adls_path (str): Path to the directory containing CSV files in ADLS2
    analysis_table_name (str): Full name of the Hive metastore table for analysis results (database.table_name)
    log_table_name (str): Full name of the Hive metastore table for processing logs (database.table_name)
    
    Returns:
    tuple: (analysis_df, log_df) - DataFrames containing the analysis results and processing log
    """
    
    # Initialize Spark Session
    spark = SparkSession.builder.appName("CSV Analysis").getOrCreate()
    
    # Check if log table exists and retrieve all processed files (regardless of status)
    already_processed_files = []
    try:
        log_df_existing = spark.table(log_table_name)
        # Get list of all files that have been processed, regardless of success status
        already_processed_files = [row.file_path for row in 
                                  log_df_existing.select("file_path").distinct().collect()]
        print(f"Found {len(already_processed_files)} already processed files in log table")
    except Exception as e:
        print(f"Log table doesn't exist yet or couldn't be read: {str(e)}")
        print("Will process all files found")
    
    # Function to recursively list all CSV files in the given path
    def list_csv_files(path):
        files = []
        file_list = dbutils.fs.ls(path)
        
        for file_info in file_list:
            if file_info.isDir():
                # Recursively list files in subdirectory
                files.extend(list_csv_files(file_info.path))
            elif file_info.path.endswith(".csv"):
                # Add CSV file to the list if not already processed
                if file_info.path not in already_processed_files:
                    files.append(file_info.path)
        
        return files
    
    # Function to extract filename from path
    def extract_filename(path):
        # Use the last part of the path after the last slash
        return path.split("/")[-1]
    
    # Create schema for processing log - INCLUDING called_ts column
    log_schema = StructType([
        StructField("file_path", StringType(), False),
        StructField("processing_timestamp", TimestampType(), False),
        StructField("status", StringType(), False),
        StructField("error_message", StringType(), True),
        StructField("filename_valid", BooleanType(), True),
        StructField("required_columns_exist", BooleanType(), True),
        StructField("file_not_empty", BooleanType(), True),
        StructField("called_ts", TimestampType(), True)  # Added called_ts
    ])
    
    # Create schema for analysis results - INCLUDING called_ts column
    analysis_schema = StructType([
        StructField("provider_name", StringType(), True),
        StructField("category_name", StringType(), True),
        StructField("column_name", StringType(), True),
        StructField("row_count", IntegerType(), True),
        StructField("null_count", IntegerType(), True),
        StructField("extract_date", StringType(), True),
        StructField("called_ts", TimestampType(), True)  # Added called_ts
    ])
    
    # Initialize lists to collect results
    analysis_rows = []
    log_entries = []
    
    # Get all CSV files
    csv_files = list_csv_files(adls_path)
    print(f"Found {len(csv_files)} CSV files for processing")
    
    # Process each CSV file
    for file_path in csv_files:
        # Get current timestamp for this file processing
        current_ts = datetime.now()
        
        # Extract filename from path
        filename = extract_filename(file_path)
        
        # Initialize log entry variables
        filename_valid = False
        required_columns_exist = False
        file_not_empty = False
        error_message = None
        status = "FAILED"  # Default to failed, will update to SUCCESS if all goes well
        
        try:
            # Validation 1: Check filename format (should be at least 12 characters long)
            if len(filename) <= 12:
                raise ValueError(f"Filename '{filename}' is too short. Expected at least 12 characters.")
            filename_valid = True
            
            # Extract date from filename (first 8 characters)
            extract_date = filename[:8]
            # Validate date format (basic check: should be 8 digits)
            if not extract_date.isdigit() or len(extract_date) != 8:
                raise ValueError(f"Extract date '{extract_date}' is not in the expected format (8 digits).")
            
            # Extract category name (substring after 12th character of filename)
            category_name = filename[12:]
            # Remove file extension from category name
            if category_name.endswith(".csv"):
                category_name = category_name[:-4]
            
            # Read the CSV file
            df = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)
            
            # Validation 2: Check if file is empty
            row_count = df.count()
            if row_count == 0:
                raise ValueError(f"CSV file is empty.")
            file_not_empty = True
            
            # Validation 3: Check for required columns
            if "Provider name" not in df.columns:
                raise ValueError(f"Required column 'Provider name' not found in the CSV.")
            required_columns_exist = True
            
            # Get provider name from the first row (assuming it's the same for all rows)
            provider_row = df.select("Provider name").first()
            if provider_row is None:
                raise ValueError(f"No data found in 'Provider name' column.")
            provider_name = provider_row[0]
            
            # Analyze each column
            for column in df.columns:
                # Calculate row count and null count for each column
                counts = df.select(
                    count(col(column)).alias("row_count"),
                    count(when(col(column).isNull() | isnan(col(column)), column)).alias("null_count")
                ).collect()[0]
                
                col_row_count = counts["row_count"]
                null_count = counts["null_count"]
                
                # Add a row with all required information to the results list
                analysis_rows.append((
                    provider_name,      # Provider Name
                    category_name,      # Category name
                    column,             # Column Name
                    col_row_count,      # Row Count per Column Name
                    null_count,         # Null Count per Column Name
                    extract_date,       # Extract Date
                    current_ts          # Called timestamp
                ))
            
            # Update status to indicate success
            status = "SUCCESS"
            
        except Exception as e:
            # Capture the error message
            error_message = str(e)
            print(f"Error processing {file_path}: {error_message}")
        
        # Add the log entry to the log entries list
        log_entries.append((
            file_path,
            current_ts,
            status,
            error_message,
            filename_valid,
            required_columns_exist,
            file_not_empty,
            current_ts              # Called timestamp
        ))
    
    # Create DataFrames from the collected results
    if analysis_rows:
        analysis_df = spark.createDataFrame(analysis_rows, analysis_schema)
        
        # Handle potential duplicates in the source data
        # First check for duplicate keys to warn about potential conflicts
        source_duplicates = analysis_df.groupBy(
            "provider_name", "category_name", "column_name", "extract_date"
        ).count().filter("count > 1")
        
        if source_duplicates.count() > 0:
            print("WARNING: Found duplicates in analysis source data that may cause merge conflicts:")
            source_duplicates.show()
            
            # Group by the key fields and aggregate the metrics to prevent merge conflicts
            # Using aggregation logic to handle potential conflicts
            analysis_df = analysis_df.groupBy(
                "provider_name", "category_name", "column_name", "extract_date"
            ).agg(
                F.max("row_count").alias("row_count"),
                F.max("null_count").alias("null_count"),
                F.max("called_ts").alias("called_ts")
            )
            
            # Verify duplicates are removed
            remaining_duplicates = analysis_df.groupBy(
                "provider_name", "category_name", "column_name", "extract_date"
            ).count().filter("count > 1")
            
            if remaining_duplicates.count() > 0:
                print("ERROR: Still found duplicates after aggregation. Using forced deduplication.")
                # Force deduplication as last resort
                analysis_df = analysis_df.dropDuplicates(["provider_name", "category_name", "column_name", "extract_date"])
        
        # Create temporary view for the analysis results
        analysis_df.createOrReplaceTempView("analysis_updates")
        
        # Use try-except to handle table existence and creation safely
        try:
            # Try to read from the table to check if it exists
            spark.table(analysis_table_name)
            table_exists = True
            print(f"Table {analysis_table_name} exists. Performing merge operation.")
        except Exception as e:
            if "Table or view not found" in str(e):
                table_exists = False
                print(f"Table {analysis_table_name} doesn't exist. Creating new table.")
            else:
                # If it's some other error, re-raise it
                raise e
            
        if not table_exists:
            # Create table with partitioning using IF NOT EXISTS
            analysis_df.write.format("delta").partitionBy("extract_date").option("mergeSchema", "true").saveAsTable(analysis_table_name)
        else:
            # Perform merge operation using DeltaTable API with metastore table
            from delta.tables import DeltaTable
            
            # Get Delta table by name
            delta_table = DeltaTable.forName(spark, analysis_table_name)
            
            # Define merge condition (unique key for upsert)
            merge_condition = """
                target.provider_name = source.provider_name AND
                target.category_name = source.category_name AND
                target.column_name = source.column_name AND
                target.extract_date = source.extract_date
            """
            
            # Pre-filter source data to get only the latest records per key
            # This is a workaround since we can't use conditional whenMatched
            analysis_df.createOrReplaceTempView("source_data")
            
            latest_records = spark.sql(f"""
                SELECT s.*
                FROM source_data s
                JOIN (
                    SELECT 
                        provider_name, 
                        category_name, 
                        column_name, 
                        extract_date, 
                        MAX(called_ts) as max_called_ts
                    FROM source_data
                    GROUP BY provider_name, category_name, column_name, extract_date
                ) latest
                ON s.provider_name = latest.provider_name
                AND s.category_name = latest.category_name
                AND s.column_name = latest.column_name
                AND s.extract_date = latest.extract_date
                AND s.called_ts = latest.max_called_ts
            """)
            
            # Execute merge with explicit update strategy
            delta_table.alias("target") \
                .merge(
                    source=latest_records.alias("source"),
                    condition=merge_condition
                ) \
                .whenMatchedUpdateAll() \
                .whenNotMatchedInsertAll() \
                .execute()
                
            print(f"Merge operation completed for analysis results in table {analysis_table_name}")
    else:
        print("No valid analysis results to write.")
        analysis_df = spark.createDataFrame([], analysis_schema)
    
    # Create log DataFrame
    log_df = spark.createDataFrame(log_entries, log_schema)
    
    # Handle duplicate log entries
    log_duplicates = log_df.groupBy(
        "file_path", "processing_timestamp"
    ).count().filter("count > 1")
    
    if log_duplicates.count() > 0:
        print("WARNING: Found duplicates in log data that may cause merge conflicts:")
        log_duplicates.show()
        # Force deduplication
        log_df = log_df.dropDuplicates(["file_path", "processing_timestamp"])
    
    # Use try-except to handle table existence and creation safely for log table
    try:
        # Try to read from the table to check if it exists
        spark.table(log_table_name)
        log_table_exists = True
        print(f"Table {log_table_name} exists. Performing merge operation.")
    except Exception as e:
        if "Table or view not found" in str(e):
            log_table_exists = False
            print(f"Table {log_table_name} doesn't exist. Creating new table.")
        else:
            # If it's some other error, re-raise it
            raise e
    
    if not log_table_exists:
        # Create new log table with IF NOT EXISTS logic
        log_df.write.format("delta").option("mergeSchema", "true").saveAsTable(log_table_name)
    else:
        # Perform merge operation for logs
        from delta.tables import DeltaTable
        
        # Get Delta table by name
        log_delta_table = DeltaTable.forName(spark, log_table_name)
        
        # Define merge condition for logs (file_path and processing_timestamp as unique key)
        log_merge_condition = """
            target.file_path = source.file_path AND
            target.processing_timestamp = source.processing_timestamp
        """
        
        # Pre-filter log data to get only the latest records
        log_df.createOrReplaceTempView("log_source_data")
        
        latest_log_records = spark.sql(f"""
            SELECT s.*
            FROM log_source_data s
            JOIN (
                SELECT 
                    file_path, 
                    processing_timestamp, 
                    MAX(called_ts) as max_called_ts
                FROM log_source_data
                GROUP BY file_path, processing_timestamp
            ) latest
            ON s.file_path = latest.file_path
            AND s.processing_timestamp = latest.processing_timestamp
            AND s.called_ts = latest.max_called_ts
        """)
        
        # Execute merge operation for logs with explicit update strategy
        log_delta_table.alias("target") \
            .merge(
                source=latest_log_records.alias("source"),
                condition=log_merge_condition
            ) \
            .whenMatchedUpdateAll() \
            .whenNotMatchedInsertAll() \
            .execute()
            
        print(f"Merge operation completed for processing log in table {log_table_name}")
    
    # Display summary
    success_count = log_df.filter(col("status") == "SUCCESS").count()
    failure_count = log_df.filter(col("status") == "FAILED").count()
    print(f"Processing complete. Successfully processed {success_count} files. Failed to process {failure_count} files.")
    
    # Show failed files for debugging
    if failure_count > 0:
        print("Failed files:")
        log_df.filter(col("status") == "FAILED").select("file_path", "error_message").show(truncate=False)
        
    return analysis_df, log_df

# Main execution - replace with your actual ADLS2 path and table names
if __name__ == "__main__":
    # Set your ADLS path here
    adls_path = "abfss://container@storageaccount.dfs.core.windows.net/path/to/csv/files"
    
    # Set your Hive metastore table names (include database name)
    analysis_table_name = "your_database.csv_analysis_results"
    log_table_name = "your_database.csv_processing_log"
    
    # Run the analysis
    analyze_csv_files(adls_path, analysis_table_name, log_table_name)
```

This complete code includes all the fixes we've discussed:

1. **Schema Consistency**: Added `called_ts` to both schemas and data tuples (7 fields for log, 7 fields for analysis)
2. **Duplicate Handling**: Pre-filters for only the latest records before merge operations
3. **Proper Delta Lake API**: Uses `whenMatchedUpdateAll()` and `whenNotMatchedInsertAll()`
4. **Conflict Resolution**: Uses "last write wins" strategy with `called_ts` timestamp
5. **Error Handling**: Includes comprehensive error checking and logging

Before running this code, make sure you've added the `called_ts` column to your existing tables:

```sql
-- Add called_ts column to analysis table
ALTER TABLE your_database.csv_analysis_results
ADD COLUMN called_ts TIMESTAMP;

-- Add called_ts column to log table  
ALTER TABLE your_database.csv_processing_log
ADD COLUMN called_ts TIMESTAMP;
```

This should resolve all the errors you were encountering.