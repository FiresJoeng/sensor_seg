import pandas as pd
import numpy as np
import os

# --- Configuration ---

# !! IMPORTANT !!: Update this path to the exact location of your Excel file
EXCEL_FILE_PATH = '/home/kk/pythonProject/ai_seg/sensor_seg/libs/semantic (副本).xlsx' 

# !! IMPORTANT !!: Update these sheet names if they are different in your Excel file
SHEET_NAMES = [
    '变送器部分', 
    '传感器部分', 
    '保护管部分'
]

# Output file for the processed data (optional, set to None to disable saving)
OUTPUT_CSV_PATH = 'prepared_semantic_data.csv' 

# Define key columns (using original names from previous examples)
actual_values_col = '实际参数值（多个值用|隔开）一体化'
standard_value_col = '规格书代码的说明（多个值用|隔开）'
standard_code_col = '对应代码'
standard_param_col = '标准参数'
component_part_col = '元器件部位' # Used for context/metadata

# --- Main Preprocessing Function ---

def preprocess_semantic_excel(excel_path, sheet_names_list):
    """
    Reads data from specified sheets in an Excel file, combines them,
    and processes the 'actual parameter values' column for embedding preparation.

    Args:
        excel_path (str): Path to the input Excel file.
        sheet_names_list (list): A list of sheet names to read from the Excel file.

    Returns:
        pandas.DataFrame: The processed DataFrame ready for embedding, 
                          or None if an error occurs.
    """
    print(f"--- Starting Preprocessing ---")
    print(f"Reading Excel file: {excel_path}")

    if not os.path.exists(excel_path):
        print(f"Error: Excel file not found at '{excel_path}'")
        return None

    all_dfs = []
    try:
        for sheet in sheet_names_list:
            print(f"  Reading sheet: '{sheet}'...")
            try:
                df_sheet = pd.read_excel(excel_path, sheet_name=sheet)
                all_dfs.append(df_sheet)
            except Exception as sheet_error:
                # Handle case where a specific sheet might be missing or unreadable
                print(f"  Warning: Could not read sheet '{sheet}'. Error: {sheet_error}. Skipping this sheet.")
        
        if not all_dfs:
             print("Error: No sheets were successfully read from the Excel file.")
             return None

        df_combined = pd.concat(all_dfs, ignore_index=True)
        print(f"\nCombined data from {len(all_dfs)} sheets. Shape: {df_combined.shape}")
        print(f"Initial columns: {df_combined.columns.tolist()}")

    except Exception as e:
        print(f"Error reading or combining sheets from Excel file: {e}")
        return None

    # --- Apply the same processing logic as before ---

    # Select relevant columns and handle potential missing ones
    relevant_cols = [
        actual_values_col, standard_value_col, standard_code_col,
        standard_param_col, component_part_col
    ]
    print(f"\nSelecting relevant columns: {relevant_cols}")
    # Ensure columns exist, add empty if not (avoids errors later)
    for col in relevant_cols:
        if col not in df_combined.columns:
            # Use original column name in warning, as df_combined still has them
            print(f"  Warning: Column '{col}' not found in combined data. Added as empty column.")
            df_combined[col] = np.nan # Add column if missing
            
    df_prepared = df_combined[relevant_cols].copy()

    # Handle missing values in the column to be split
    print(f"Processing column: '{actual_values_col}'")
    if actual_values_col not in df_prepared.columns:
        print(f"  Error: Crucial column '{actual_values_col}' is missing after selection. Cannot proceed.")
        return None
        
    df_prepared[actual_values_col] = df_prepared[actual_values_col].fillna('')

    # Split and explode
    print("  Splitting values by '|'...")
    df_prepared[actual_values_col] = df_prepared[actual_values_col].str.split('|')
    
    print("  Exploding rows based on split values...")
    df_exploded = df_prepared.explode(actual_values_col)

    # Clean up and rename the exploded column for clarity
    exploded_col_name = 'Actual_Value_Variation'
    df_exploded.rename(columns={actual_values_col: exploded_col_name}, inplace=True)
    df_exploded[exploded_col_name] = df_exploded[exploded_col_name].str.strip()
    print(f"  Renamed exploded column to '{exploded_col_name}' and stripped whitespace.")

    # Filter out empty variations and rows with missing standard info
    print("  Filtering out empty variations and rows with missing standard values/codes...")
    original_rows = len(df_exploded)
    
    # Ensure the standard columns exist before filtering
    if standard_value_col not in df_exploded.columns or standard_code_col not in df_exploded.columns:
         print(f"  Error: Standard columns ('{standard_value_col}', '{standard_code_col}') missing before filtering.")
         return None
         
    df_final = df_exploded[
        (df_exploded[exploded_col_name] != '') &
        (df_exploded[standard_value_col].notna()) &
        (df_exploded[standard_code_col].notna())
    ].copy()
    
    removed_rows = original_rows - len(df_final)
    print(f"  Removed {removed_rows} rows during filtering.")

    # Reset index for the final dataframe
    df_final.reset_index(drop=True, inplace=True)
    print(f"\nPreprocessing complete. Final DataFrame shape: {df_final.shape}")
    print(f"Final columns: {df_final.columns.tolist()}")

    return df_final

# --- Execution ---

if __name__ == "__main__":
    # Run the preprocessing function
    processed_data = preprocess_semantic_excel(EXCEL_FILE_PATH, SHEET_NAMES)

    # Optional: Save the processed data
    if processed_data is not None and OUTPUT_CSV_PATH:
        try:
            processed_data.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
            print(f"\nProcessed data successfully saved to: {OUTPUT_CSV_PATH}")
        except Exception as e:
            print(f"\nError saving processed data to CSV: {e}")
    elif processed_data is None:
         print("\nPreprocessing failed. No data to save.")
         
    if processed_data is not None:
        print("\nFirst 5 rows of processed data:")
        print(processed_data.head().to_markdown(index=False))