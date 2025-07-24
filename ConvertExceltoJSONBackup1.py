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
    # - skiprows=[1]: Skip the second row (index 1, which contains "TOTALS")
    # - dtype={'ANZSCO': str}: Read ANZSCO codes as strings to preserve leading zeros (if any) and avoid number formatting issues.
    df = pd.read_excel(
        excel_file_path,
        sheet_name=sheet_name,
        header=0,
        skiprows=[1],
        dtype={'ANZSCO': str} # Treat ANZSCO as text
        )
    print(f"Successfully loaded Excel file: {excel_file_path} (Sheet: {sheet_name})")
    # Optional: Print first few rows and column names to verify
    # print("First 5 rows of data:")
    # print(df.head().to_markdown(index=False)) # Use to_markdown for better console readability
    # print("\nColumn Names:")
    # print(df.columns.tolist())

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

# --- COLUMN MAPPING (Verify these match your Excel headers *exactly*) ---
# Make sure the names like "SC 482-SID", "ACT", "Unnamed: 17" etc.
# precisely match what's in the first row of your Excel file.
column_map = {
    "eligibility_lists": ["MLTSSL", "STSOL", "ROL", "CSOL"],
    "visa_subclasses": ["SC 482-SID", "SC 494", "ENS-SC 186", "SC 189", "SC 190", "SC 491 State", "SC 491 Family", "SC 485", "SC 407"],
    "states": {
        # The "Unnamed: X" columns depend heavily on the exact file structure.
        # Verify these correspond to the correct columns in your specific file.
        # You might need to open the file in Excel and see which columns lack a header in row 1.
        "ACT": ["ACT", "Unnamed: 17"],  # Assuming ACT 190 is 'ACT', ACT 491 is 'Unnamed: 17'
        "VIC": ["VIC", "Unnamed: 19"],  # Assuming VIC 190 is 'VIC', VIC 491 is 'Unnamed: 19'
        "SA": ["SA", "Unnamed: 21", "Unnamed: 22", "Unnamed: 23", "Unnamed: 24"], # SA 190, 491(Grad), 491(Skilled), 491(Outer), 491(Offshore)
        "WA": ["Unnamed: 25", "Unnamed: 26", "Unnamed: 27"], # WA 190 WS-1, 190 WS-2, 190 Grad - Check source for exact WA mapping
        "QLD": ["QLD", "Unnamed: 29", "Unnamed: 30"], # QLD 190, 491 WS-1, 491 WS-2 - Check source
        "TAS": ["TAS", "Unnamed: 32"], # TAS 190, 491
        "NT": ["NT", "Unnamed: 34", "Unnamed: 35"], # NT 190, 491, 491 Offshore? - Check source
    },
    "DAMA": ["DAMA-Adelaide", "DAMA-Reg SA"] # Assuming these columns exist
}

# Define the keys corresponding to the state columns for clarity
# Make sure the order matches the columns listed in column_map["states"] for each state!
state_key_map = {
    "ACT": ["190", "491"],
    "VIC": ["190", "491"],
    "SA": ["190", "491_SA_Grad", "491_Skilled", "491_Outer", "491_Offshore"], # Verify order matches "SA", "Unnamed: 21", etc.
    "WA": ["190_WS1", "190_WS2", "190_Grad"], # Check source for WA mapping & column names
    "QLD": ["190", "491_Onshore", "491_Offshore"], # Check source for QLD mapping & column names
    "TAS": ["190", "491"],
    "NT": ["190", "491_Onshore", "491_Offshore"], # Check source for NT mapping & column names
}

# --- DATA PREPARATION ---
# Replace empty strings and pure whitespace cells with NaN (Not a Number)
df_data = df.replace(r'^\s*$', np.nan, regex=True)
# Fill remaining NaN values in the *entire* DataFrame with None for consistent JSON output
# Alternatively, you could fill specific columns if needed
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

    # Function to safely convert value to 1, 0, or None
    def parse_flag(value):
        if pd.isna(value):
            return None
        str_val = str(value).strip()
        if str_val == '1' or str_val.lower() == 'yes' or str_val.lower() == 'true': # Handle common boolean text
            return 1
        if str_val == '0' or str_val.lower() == 'no' or str_val.lower() == 'false':
             return 0
        # Handle cases like '1.0' coming from Excel numbers
        try:
            float_val = float(str_val)
            if float_val == 1.0:
                return 1
            if float_val == 0.0:
                return 0
        except ValueError:
            pass # Not a float
        return None # If it's not clearly 1 or 0, treat as None


    entry = {
        # Use .get with a default of None for safety, trim whitespace
        "ANZSCO": str(row.get("ANZSCO", "")).strip() or None,
        "Occupation": str(row.get("Occupation", "")).strip() or None,
        "Assessing_Authority": str(row.get("Assessing Authority", "")).strip() or None, # Changed key to snake_case
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
        field_key = visa_key.replace(" ", "_").replace("-", "_") # Convert to snake_case
        entry["visa_subclasses"][field_key] = parse_flag(row.get(visa_key))

    # States
    for state, cols in column_map["states"].items():
        state_data = {}
        if state in state_key_map:
            keys_for_state = state_key_map[state]
            if len(cols) != len(keys_for_state):
                print(f"Warning: Mismatch between defined columns ({len(cols)}) and keys ({len(keys_for_state)}) for state '{state}'. Check column_map and state_key_map.")
            for i, col in enumerate(cols):
                 if i < len(keys_for_state): # Process only if a key is defined
                    state_key = keys_for_state[i]
                    # Check if the column actually exists in the DataFrame before getting
                    if col in df_data.columns:
                        state_data[state_key] = parse_flag(row.get(col))
                    else:
                         state_data[state_key] = None # Column defined in map but not found in Excel
                         # Optional: Add a warning here if a mapped column is missing
                         # print(f"Warning: Column '{col}' for state '{state}' not found in Excel file.")
                 # else: (Handled by the length check above)
        else:
             print(f"Warning: State '{state}' found in column_map['states'] but not defined in state_key_map. Skipping state.")
        entry["State"][state] = state_data


    # DAMA
    # Check if DAMA columns exist before trying to access them
    adelaide_val = row.get("DAMA-Adelaide") if "DAMA-Adelaide" in df_data.columns else None
    reg_sa_val = row.get("DAMA-Reg SA") if "DAMA-Reg SA" in df_data.columns else None # Note space in 'Reg SA'

    entry["DAMA"]["Adelaide"] = parse_flag(adelaide_val)
    entry["DAMA"]["Regional_SA"] = parse_flag(reg_sa_val) # Key uses underscore

    full_json.append(entry)
    processed_rows += 1

print(f"Processing complete. Processed {processed_rows} data rows.")
if skipped_rows_missing_anzsco > 0:
    print(f"Skipped {skipped_rows_missing_anzsco} rows due to missing ANZSCO code.")

# --- SAVE TO JSON FILE ---
try:
    with open(output_path, "w", encoding='utf-8') as f: # Specify encoding
        json.dump(full_json, f, indent=2, ensure_ascii=False) # ensure_ascii=False for non-Latin characters
    print(f"\nSuccessfully saved JSON data to: {output_path}")
except Exception as e:
    print(f"\nAn error occurred while saving the JSON file: {e}")