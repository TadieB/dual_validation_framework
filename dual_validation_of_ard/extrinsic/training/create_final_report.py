import pandas as pd
import argparse
import os

# OUT_OF_ME+ , even 25G, 8 CPU cannot run this task!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def main(args):
    print("--- Generating Final Stratified Report ---")

    # --- 1. Load all individual results files ---
    print("Loading all inference and baseline results...")
    models_to_load = {
        '3D U-Net': args.unet_csv,
        'Prithvi (Zero-Shot)': args.prithvi_zeroshot_csv,
        'Prithvi (Frozen)': args.prithvi_frozen_csv,
        'Prithvi (Partial FT)': args.prithvi_partial_csv,
        'Least Cloudy': args.least_cloudy_csv,
        'Mosaicking': args.mosaicking_csv
    }
    
    all_dfs = []
    for model_name, path in models_to_load.items():
        try:
            df = pd.read_csv(path)
            df['model'] = model_name
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: Could not find results file for {model_name} at {path}. Skipping.")
    
    if not all_dfs:
        print("Error: No results files found. Exiting.")
        return
        
    all_results = pd.concat(all_dfs)

    # --- 2. Load difficulty map and merge ---
    print("Loading difficulty map and merging with results...")
    df_difficulty = pd.read_csv(args.difficulty_csv)
    
    merged_df = pd.merge(all_results, df_difficulty, on=['y_coord', 'x_coord'])
    
    if merged_df.empty:
        print("\nERROR: Merge failed. No matching pixel coordinates were found.")
        return

    # ==================== OPTIMIZATION START ====================
    # Reduce memory usage by downcasting data types before pivoting
    print("Optimizing DataFrame memory usage...")
    for col in merged_df.select_dtypes(include=['float64']).columns:
        merged_df[col] = merged_df[col].astype('float32')
    for col in merged_df.select_dtypes(include=['int64']).columns:
        merged_df[col] = merged_df[col].astype('int32')
    # ===================== OPTIMIZATION END =====================

    # --- 3. Create the Stratum Column ---
    print("Assigning difficulty strata...")
    merged_df['stratum'] = pd.qcut(
        merged_df['difficulty_index'].rank(method='first'),
        q=3,
        labels=['Low', 'Medium', 'High']
    )

    # --- 4. Create and Save the Final Master CSV File ---
    print("Pivoting data to create final report...")
    final_report_df = merged_df.pivot_table(
        index=['y_coord', 'x_coord', 'stratum', 'difficulty_index', 'heterogeneity', 'phenology_variability', 'cloud_persistence'],
        columns='model',
        values=['rmse', 'psnr', 'ssim', 'sam']
    )
    final_report_df.columns = ['_'.join(col).strip() for col in final_report_df.columns.values]
    final_report_df.reset_index(inplace=True)
    
    output_path = os.path.join(args.output_dir, "final_stratified_results_lookup.csv")
    final_report_df.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("SUCCESS: Final report generated!")
    print(f"Master CSV file saved to: {output_path}")
    print("="*60)
    print("\nThis file is your lookup table. You can now open it to find the coordinates")
    print("of samples in the 'Low', 'Medium', and 'High' strata for your visualizations.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a final, master report for all model results.")
    
    parser.add_argument('--unet_csv', type=str, required=True)
    parser.add_argument('--prithvi_zeroshot_csv', type=str, required=True)
    parser.add_argument('--prithvi_frozen_csv', type=str, required=True)
    parser.add_argument('--prithvi_partial_csv', type=str, required=True)
    parser.add_argument('--least_cloudy_csv', type=str, required=True)
    parser.add_argument('--mosaicking_csv', type=str, required=True)
    parser.add_argument('--difficulty_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='.')
    
    args = parser.parse_args()
    main(args)
