import pandas as pd
import numpy as np
import csv

def auto_format_for_jadbio(test_file_path, jadbio_csv_path, output_path="ready_for_jadbio.csv"):
    """
    Aligns features and strictly quotes categorical variables so JadBio 
    automatically recognizes them as Categorical without manual UI clicks.
    """
    print("Loading datasets...")
    # Load feature definitions
    jadbio_features = pd.read_csv(jadbio_csv_path).drop_duplicates(subset=['Feature_Name'])
    type_mapping = dict(zip(jadbio_features['Feature_Name'], jadbio_features['Assigned_Type']))
    expected_cols = list(jadbio_features['Feature_Name'])
    
    # Load your test data
    test_df = pd.read_csv(test_file_path)
    
    print("Aligning features...")
    # Add missing columns with NaN
    for col in expected_cols:
        if col not in test_df.columns:
            test_df[col] = np.nan
            
    test_df = test_df[expected_cols]
    
    print("Applying strict typing and quoting for JadBio...")
    for col in expected_cols:
        expected_type = str(type_mapping[col]).strip().lower()
        
        if expected_type == 'numerical':
            # Force numbers to be floats. Pandas will not quote floats.
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
            
        elif expected_type == 'categorical':
            # Force categorical variables to be strings. 
            # Pandas will put quotes around these when we export.
            # 1. Temporarily replace NaN so they don't become the word "nan"
            test_df[col] = test_df[col].fillna('__MISSING__')
            # 2. Convert to string
            test_df[col] = test_df[col].astype(str)
            # 3. Put actual nulls back
            test_df[col] = test_df[col].replace('__MISSING__', np.nan)

    # Save using QUOTE_NONNUMERIC
    # This tells Pandas: "Put double quotes around anything that is a string, but leave numbers alone."
    test_df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    
    print(f"Success! Automated file saved as: {output_path}")
    print("You can now upload this directly to JadBio. Categorical features are safely quoted!")

if __name__ == "__main__":
    # Specify your file names here
    TEST_DATA_FILE = r"C:\Users\hp\Desktop\stage etis\open_dataset\jadbio_csvs\jadbio_animal_musk_test_real.csv"
    JADBIO_FEATURE_FILE = r"C:\Users\hp\Desktop\stage etis\jadbio_all_feature_types.csv"
    
    auto_format_for_jadbio(TEST_DATA_FILE, JADBIO_FEATURE_FILE)