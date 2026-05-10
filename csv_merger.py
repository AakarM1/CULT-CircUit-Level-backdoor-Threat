import os
import pandas as pd

def combine_csv_files(directory, output_file='combined2.csv'):
    # List to hold all dataframes
    all_dataframes = []

    # Walk through the directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):  # Check if the file is a CSV
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                print(file)  # Read the CSV file into a dataframe
                iid = file.split("_")[3]
                mix = file.split("_")[2]
                # print(iid, mix)
                df['iid'] = iid
                df['Device Type'] = mix
                # print(df.head())
                all_dataframes.append(df)  # Add the dataframe to the list

    # Combine all dataframes into one
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Write the combined dataframe to an output CSV
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved as {output_file}")

# Example usage
directory = 'C:/Users/Admin/Desktop/Aakar/QFL/Backdoor Attacks/Quantum-Federated-Learning/runs/grid_20250924_152319_full'#input("Enter the directory location: ")
combine_csv_files(directory)