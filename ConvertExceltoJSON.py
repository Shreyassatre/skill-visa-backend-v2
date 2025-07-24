import json
import pandas as pd
import numpy as np
import sys # Import sys to use sys.exit()

# --- CONFIGURATION ---
# !! REPLACE 'your_file_name.xlsx' WITH THE ACTUAL PATH TO YOUR FILE !!
excel_file_path = r'Occupations List (21 July).xlsx'
# Specify the sheet name if it's not the first one (optional)
# sheet_name = 'YourSheetName' # Uncomment and change if needed
sheet_name = 0 # Use 0 for the first sheet by index

# Output JSON file path
output_path = "aus_visa_occupations_only.json" # Save in the same directory

# --- READ EXCEL FILE ---
try:
    # Read the Excel file:
    # - header=0: Headers are in the first row (index 0)
    # - skiprows=[1]: Skip the second row (index 1), which might be a totals/summary row.
    # - dtype={'ANZSCO': str}: Read ANZSCO codes as strings to preserve leading zeros.
    df = pd.read_excel(
        excel_file_path,
        sheet_name=sheet_name,
        header=0,
        skiprows=[1],
        dtype={'ANZSCO': str} # Treat ANZSCO as text
        )
    print(f"Successfully loaded Excel file: {excel_file_path} (Sheet: {sheet_name})")

except FileNotFoundError:
    print(f"Error: The file '{excel_file_path}' was not found.")
    print("Please make sure the file path is correct and the file exists.")
    sys.exit(1) # Exit the script with an error code
except ValueError as e:
    # Handle potential sheet name errors
     print(f"Error reading sheet '{sheet_name}': {e}")
     print(f"Please ensure the sheet name or index is correct in the file '{excel_file_path}'.")
     sys.exit(1)
except Exception as e:
    print(f"An error occurred while reading the Excel file: {e}")
    print("Potential issues: File corruption, incorrect format, or password protection.")
    sys.exit(1) # Exit on other potential errors

# --- COLUMN MAPPING (Updated for new headers) ---
# This map links the desired JSON structure to the exact column names in your new Excel file.
column_map = {
    "eligibility_lists": ["MLTSSL", "STSOL", "ROL", "CSOL"],
    "visa_subclasses": ["SC 482-SID", "SC 494", "ENS-SC 186", "SC 189", "SC 190", "SC 491 State", "SC 491 Family", "SC 485", "SC 407"],
    "states": {
        "NSW": ["NSW - Skills List", "NSW - Regional Skills List"],
        "ACT": ["ACT-190", "ACT-491"],
        "VIC": ["VIC -190", "VIC 491"],
        "SA": ["SA 190", "491(SA Grad)", "SA 491(Skilled)", "SA 491(Outer)", "SA 491(Offshore)"],
        "WA": ["WA 190 WS-1", "WA 190 WS-2", "WA 190 - Grad", "WA 190", "WA 491 - WS-1", "WA 491 WS-2"],
        "QLD": ["QLD 190", "QLD 491"],
        "TAS": ["TAS 190", "TAS 491"],
        "NT": ["NT 190", "NT 491", "NT 491 - Offshore"],
    },
    "DAMA": [
        "DAMA-Adelaide", "DAMA-Reg SA", "DAMA-East Kim", "DAMA-Far North QLD",
        "DAMA-GV", "DAMA-GSC", "DAMA-NT", "DAMA-Orana", "DAMA-Pilbara",
        "DAMA-South West", "DAMA-Goldfields", "DAMA-Townsville", "DAMA-WA"
    ]
}

# This map defines the JSON keys for the state-specific columns.
# The order of keys for each state MUST match the order of columns in column_map["states"].
state_key_map = {
    "NSW": ["Skills_List", "Regional_Skills_List"],
    "ACT": ["190", "491"],
    "VIC": ["190", "491"],
    "SA": ["190", "491_SA_Grad", "491_Skilled", "491_Outer", "491_Offshore"],
    "WA": ["190_WS1", "190_WS2", "190_Grad", "190_General", "491_WS1", "491_WS2"],
    "QLD": ["190", "491"],
    "TAS": ["190", "491"],
    "NT": ["190", "491_Onshore", "491_Offshore"],
}

# --- DATA PREPARATION ---
# Replace empty strings and pure whitespace cells with NaN (Not a Number)
df_data = df.replace(r'^\s*$', np.nan, regex=True)
# Fill remaining NaN values with None for consistent JSON output
df_data = df_data.fillna(value=np.nan)


# --- BUILD JSON ---
full_json = []
processed_rows = 0
skipped_rows_missing_anzsco = 0

print("\nProcessing rows...")

for index, row in df_data.iterrows():
    # Skip rows where ANZSCO is missing, as it's a primary identifier
    anzsco_code = row.get("ANZSCO")
    if pd.isna(anzsco_code) or str(anzsco_code).strip() == "":
        skipped_rows_missing_anzsco += 1
        continue # Skip to the next row

    # Function to safely convert values to 1, 0, or None
    def parse_flag(value):
        if pd.isna(value):
            return None
        str_val = str(value).strip()
        if str_val == '1' or str_val.lower() == 'yes' or str_val.lower() == 'true':
            return 1
        if str_val == '0' or str_val.lower() == 'no' or str_val.lower() == 'false':
             return 0
        try:
            float_val = float(str_val)
            if float_val == 1.0:
                return 1
            if float_val == 0.0:
                return 0
        except ValueError:
            pass # Not a float
        return None # If not clearly 1 or 0, treat as None


    entry = {
        # Use .get with a default for safety, and trim whitespace
        "ANZSCO": str(row.get("ANZSCO", "")).strip() or None,
        "Occupation": str(row.get("Occupation", "")).strip() or None,
        "Assessing_Authority": str(row.get("Assessing Authority", "")).strip() or None,
        "eligibility_lists": {},
        "visa_subclasses": {},
        "State": {},
        "DAMA": {}
    }

    # Eligibility Lists
    for list_key in column_map["eligibility_lists"]:
        entry["eligibility_lists"][list_key] = parse_flag(row.get(list_key))

    # Visa Subclasses
    for visa_key in column_map["visa_subclasses"]:
        field_key = visa_key.replace(" ", "_").replace("-", "_") # Convert to snake_case for JSON key
        entry["visa_subclasses"][field_key] = parse_flag(row.get(visa_key))

    # States
    for state, cols in column_map["states"].items():
        state_data = {}
        if state in state_key_map:
            keys_for_state = state_key_map[state]
            if len(cols) != len(keys_for_state):
                print(f"Warning: Mismatch between defined columns ({len(cols)}) and keys ({len(keys_for_state)}) for state '{state}'. Check your mapping dictionaries.")
            for i, col_name in enumerate(cols):
                 if i < len(keys_for_state):
                    state_key = keys_for_state[i]
                    if col_name in df_data.columns:
                        state_data[state_key] = parse_flag(row.get(col_name))
                    else:
                         state_data[state_key] = None # Column in map but not in Excel
                         print(f"Warning: Column '{col_name}' for state '{state}' not found in Excel file.")
        else:
             print(f"Warning: State '{state}' is not defined in state_key_map. Skipping.")
        entry["State"][state] = state_data


    # DAMA (Dynamic Processing)
    for dama_col in column_map["DAMA"]:
        if dama_col in df_data.columns:
            # Create a clean key for the JSON output
            # e.g., "DAMA-Far North QLD" becomes "Far_North_QLD"
            dama_key = dama_col.replace("DAMA-", "").replace(" ", "_")
            entry["DAMA"][dama_key] = parse_flag(row.get(dama_col))

    full_json.append(entry)
    processed_rows += 1

print(f"Processing complete. Processed {processed_rows} data rows.")
if skipped_rows_missing_anzsco > 0:
    print(f"Skipped {skipped_rows_missing_anzsco} rows due to missing ANZSCO code.")

# --- SAVE TO JSON FILE ---
try:
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False)
    print(f"\nSuccessfully saved JSON data to: {output_path}")
except Exception as e:
    print(f"\nAn error occurred while saving the JSON file: {e}")