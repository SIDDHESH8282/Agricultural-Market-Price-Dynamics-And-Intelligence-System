import pandas as pd

try:
    df = pd.read_csv('preprocessed_no_leakage_sorted_2014_2024_CLEANED (1).csv')
    with open('eda_summary.txt', 'w') as f:
        f.write(f"Columns: {df.columns.tolist()}\n")
        f.write(f"Shape: {df.shape}\n")
        
        # Identify date column
        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            f.write(f"Date Range: {df[date_col].min()} to {df[date_col].max()}\n")
        else:
            f.write("Date column not found.\n")

        # Identify target (assuming 'price' or similar)
        f.write("Sample Row:\n")
        f.write(str(df.iloc[0].to_dict()) + "\n")
    
except Exception as e:
    print(f"Error: {e}")
