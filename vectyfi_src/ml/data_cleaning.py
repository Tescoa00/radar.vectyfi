import pandas as pd

def clean_ted_data(input_filepath, output_filepath):
    print("1. Loading raw TED data...")
    # low_memory=False prevents pandas warnings when dealing with mixed data types in massive CSVs
    df_raw = pd.read_csv(input_filepath, low_memory=False)

    print("2. Deduplicating by ID_AWARD...")
    # Essential to preserve the true market structure. The flat CSV format duplicates 
    # overarching notice information for every single contract award [2].
    df_unique = df_raw.drop_duplicates(subset=['ID_AWARD']).copy()

    print("3. Creating the Target Variable (TARGET_NOT_AWARDED)...")
    # According to the TED codebook, INFO_ON_NON_AWARD is empty if the contract was awarded.
    # It contains strings like 'PROCUREMENT_UNSUCCESSFUL' if it failed [3]. 
    df_unique['TARGET_NOT_AWARDED'] = df_unique['INFO_ON_NON_AWARD'].notna().astype(int)

    print("4. Balancing the Dataset (250k Awarded / 250k Failed)...")
    df_awarded = df_unique[df_unique['TARGET_NOT_AWARDED'] == 0]
    df_failed = df_unique[df_unique['TARGET_NOT_AWARDED'] == 1]

    # Random sampling across the entire multi-year pool to prevent geographical/temporal bias
    df_awarded_sample = df_awarded.sample(n=250000, random_state=42)
    df_failed_sample = df_failed.sample(n=250000, random_state=42)
    
    # Concatenate the balanced classes and shuffle them thoroughly (frac=1)
    df_balanced = pd.concat([df_awarded_sample, df_failed_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

    print("5. Selecting the 14 verified safe pre-award features...")
    columns_to_load = [
        'ID_AWARD', 'INFO_ON_NON_AWARD', # We load these now but drop them at the very end
        'B_MULTIPLE_CAE', 'B_EU_FUNDS', 'TOP_TYPE', 'ISO_COUNTRY_CODE', 
        'B_FRA_AGREEMENT', 'B_GPA', 'YEAR', 'TYPE_OF_CONTRACT', 
        'CAE_TYPE', 'CRIT_CODE', 'B_ACCELERATED', 'MAIN_ACTIVITY', 
        'CRIT_PRICE_WEIGHT', 'LOTS_NUMBER', 'TARGET_NOT_AWARDED'
    ]
    
    # Safely load only the columns that actually exist in the dataframe
    existing_cols = [c for c in columns_to_load if c in df_balanced.columns]
    df_ultimate = df_balanced[existing_cols].copy()

    print("6. Categorical Clean Up...")
    # Fill missing country codes to prevent the One-Hot Encoder from crashing
    if 'ISO_COUNTRY_CODE' in df_ultimate.columns:
        df_ultimate['ISO_COUNTRY_CODE'] = df_ultimate['ISO_COUNTRY_CODE'].fillna('UNKNOWN')

    print("7. Dropping Identifiers to guarantee Zero Data Leakage...")
    # ID_AWARD and INFO_ON_NON_AWARD must be deleted before machine learning.
    # Keeping them would allow the model to cheat by looking directly at the award result [3].
    cols_to_drop = ['ID_AWARD', 'INFO_ON_NON_AWARD']
    existing_drops = [c for c in cols_to_drop if c in df_ultimate.columns]
    df_ultimate = df_ultimate.drop(columns=existing_drops)

    num_rows, num_cols = df_ultimate.shape
    print(f"\nFinal Dataset Shape: {num_rows} rows, {num_cols} columns.")
    
    df_ultimate.to_csv(output_filepath, index=False)
    print(f"Success! Cleaned data saved to: {output_filepath}")

# Execute the script
if __name__ == "__main__":
    # Ensure you replace 'ted_raw_data.csv' with your actual input file name
    clean_ted_data('/Users/edu/Edu/testproject/tenderpilot_data/data/export_CAN_2023_2018.csv','raw_data/f_balanced_500k.csv')