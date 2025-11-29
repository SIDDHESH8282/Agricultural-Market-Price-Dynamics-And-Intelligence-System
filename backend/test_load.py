import pandas as pd
import time

print("Testing CSV load...")
start = time.time()

try:
    df = pd.read_csv('../preprocessed_final_2014_2025_FULL_FEATURES_corrected_SORTED_NO_NULLS.csv', low_memory=False)
    print(f"SUCCESS: Loaded {len(df)} rows in {time.time()-start:.2f} seconds")
    print(f"Columns: {df.columns.tolist()[:10]}")
    print(f"Data types: {df.dtypes.value_counts()}")
except Exception as e:
    print(f"ERROR: {e}")
