import pandas as pd
import time
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline

RANDOM_STATE = 42
RAW_DATA_PATH='./raw_data/'

def clean_ted_data(input_filepath, output_filepath=None):
    print(f"1. Loading raw TED data '{input_filepath}'...")
    # low_memory=False prevents pandas warnings when dealing with mixed data types in massive CSVs
    df_raw = pd.read_csv(input_filepath, low_memory=False)

    print("2. Creating balanced data set and shuffle...")
    values = {'INFO_ON_NON_AWARD': 'awarded'}
    df_raw_balanced = df_raw.fillna(value=values)
    # create max data set, many rows will be removed in duplicate removal below
    n_awarded      = 380_000
    n_unsuccessful = 190_000
    n_discontinued = 190_000
    grp_awarded      = df_raw_balanced[df_raw_balanced["INFO_ON_NON_AWARD"] == "awarded"]
    grp_unsuccessful = df_raw_balanced[df_raw_balanced["INFO_ON_NON_AWARD"] == "PROCUREMENT_UNSUCCESSFUL"]
    grp_discontinued = df_raw_balanced[df_raw_balanced["INFO_ON_NON_AWARD"] == "PROCUREMENT_DISCONTINUED"]
    df_raw_all_balanced = (
        pd.concat([
            grp_awarded.sample(n=n_awarded, random_state=RANDOM_STATE),
            grp_unsuccessful.sample(n=n_unsuccessful, random_state=RANDOM_STATE),
            grp_discontinued.sample(n=n_discontinued, random_state=RANDOM_STATE),
            ])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
        )

    print("3. Deduplicating by ID_NOTICE_CAN...")
    df_unique = df_raw_all_balanced.drop_duplicates(subset=['ID_NOTICE_CAN'], keep='first')

    print("4. Selecting the 14 verified safe pre-award features + target...")
    # TARGET_NOT_AWARDED: 1 = not awarded, 0 = awarded (inverted from 'awarded' label)
    df_unique['TARGET_NOT_AWARDED'] = (df_unique['INFO_ON_NON_AWARD'] != 'awarded').astype(int)

    columns_to_keep = [
        'B_MULTIPLE_CAE', 'B_EU_FUNDS', 'TOP_TYPE', 'ISO_COUNTRY_CODE',
        'B_FRA_AGREEMENT', 'B_GPA', 'YEAR', 'TYPE_OF_CONTRACT',
        'CAE_TYPE', 'CRIT_CODE', 'B_ACCELERATED', 'MAIN_ACTIVITY',
        'CRIT_PRICE_WEIGHT', 'LOTS_NUMBER', 'TARGET_NOT_AWARDED'
    ]
    df_non_na = df_unique[columns_to_keep].copy()

    print("5. Taking care of missing values...")
    df_non_na['B_ACCELERATED'] = df_non_na['B_ACCELERATED'].fillna(0).replace('Y', 1)
    # CRIT_PRICE_WEIGHT: strip " %", replace EU comma decimal, take first number
    df_non_na['CRIT_PRICE_WEIGHT'] = (
        df_non_na['CRIT_PRICE_WEIGHT']
        .astype(str)
        .str.replace(r'\s*%', '', regex=True)
        .str.replace(',', '.', regex=False)
        .str.extract(r'(\d+\.?\d*)')[0]
        .pipe(pd.to_numeric, errors='coerce')
        .fillna(0)
    )
    df_non_na['ISO_COUNTRY_CODE'] = df_non_na['ISO_COUNTRY_CODE'].fillna('UNKNOWN')

    preproc_num = make_pipeline(SimpleImputer(strategy="mean"))
    preproc_cat = make_pipeline(SimpleImputer(strategy="most_frequent"))

    # Exclude target column from imputation
    feature_cols = [c for c in columns_to_keep if c != 'TARGET_NOT_AWARDED']
    preproc_transformer = make_column_transformer(
        (preproc_num, make_column_selector(dtype_include=["number"])),
        (preproc_cat, make_column_selector(dtype_include=["object"])),
        remainder="passthrough"
    ).set_output(transform="pandas")
    df_out = preproc_transformer.fit_transform(df_non_na[feature_cols])
    df_out['TARGET_NOT_AWARDED'] = df_non_na['TARGET_NOT_AWARDED']

    num_rows, num_cols = df_out.shape
    print(f"\nFinal Dataset Shape: {num_rows} rows, {num_cols} columns.")

    #TODO optional: timestamp = time.strftime("%Y%m%d-%H%M%S")
    if output_filepath is None:
        output_filepath = RAW_DATA_PATH + 'balanced_cleaned_' + str(round(num_rows, -3)).rstrip('0') + 'k.csv'
    df_out.to_csv(output_filepath, index=False)
    print(f"Success! Cleaned data saved to: {output_filepath}")

# Execute the script
if __name__ == "__main__":
    clean_ted_data(RAW_DATA_PATH + 'export_CAN_2023_2018.csv')