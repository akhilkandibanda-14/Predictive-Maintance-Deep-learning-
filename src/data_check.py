import os
import pandas as pd
import glob

def main():
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    
    # Verify if directory exists
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory '{dataset_dir}' does not exist.")
        return
        
    # Automatically detect the CSV file
    csv_files = glob.glob(os.path.join(dataset_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV file found in '{dataset_dir}'. Please place the dataset here.")
        return
        
    dataset_path = csv_files[0]
    print(f"Dataset found: {dataset_path}")
    
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    print("\n--- Dataset Shape ---")
    print(df.shape)
    
    print("\n--- Dataset Columns ---")
    print(df.columns.tolist())
    
    print("\n--- Data Types ---")
    print(df.dtypes)
    
    print("\n--- First 5 Rows ---")
    print(df.head())

if __name__ == '__main__':
    main()
