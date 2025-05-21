import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
import json 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_burst_id_from_json_string(json_str):
    """Helper function to extract burst_id from a JSON string."""
    try:
        return json.loads(json_str).get('burst_id')
    except Exception as e:
        logging.error(f"Error parsing JSON string: {json_str[:100]}... Error: {e}")
        return None

def main(args):
    original_csv_path = Path(args.input_csv_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # Create output directory if it doesn't exist

    train_output_filename = args.train_output_name
    test_output_filename = args.test_output_name

    train_output_path = output_dir / train_output_filename
    test_output_path = output_dir / test_output_filename

    logging.info(f"Loading original dataset from: {original_csv_path}")
    try:
        # Load the entire original CSV.
        df_original = pd.read_csv(original_csv_path, sep=';')
    except FileNotFoundError:
        logging.error(f"Original CSV file not found at {original_csv_path}. Exiting.")
        return
    except Exception as e:
        logging.error(f"Error loading original CSV file: {e}. Exiting.")
        return

    logging.info(f"Successfully loaded {len(df_original)} rows from {original_csv_path}.")

    # --- Extract burst_ids ---
    # The burst_id is inside the 'json' column. We need to parse it to perform the split.
    logging.info("Extracting burst_ids from the 'json' column...")
    try:
        # Using tqdm for progress if the file is large
        from tqdm import tqdm
        tqdm.pandas(desc="Parsing JSON for burst_ids")
        df_original['temp_burst_id'] = df_original['json'].progress_apply(get_burst_id_from_json_string)
    except ImportError:
        logging.warning("tqdm not found. Progress bar will not be shown for JSON parsing.")
        df_original['temp_burst_id'] = df_original['json'].apply(get_burst_id_from_json_string)
    
    # Handle cases where burst_id might not have been parsed (e.g., malformed JSON)
    if df_original['temp_burst_id'].isnull().any():
        num_null_burst_ids = df_original['temp_burst_id'].isnull().sum()
        logging.warning(f"Found {num_null_burst_ids} rows where burst_id could not be parsed from JSON. These rows will be excluded from splitting if they persist.")
        df_original.dropna(subset=['temp_burst_id'], inplace=True) # Drop rows where burst_id is None
        if df_original.empty:
            logging.error("No valid burst_ids found after parsing. Cannot proceed with splitting.")
            return
            
    df_original['temp_burst_id'] = df_original['temp_burst_id'].astype(int) # Ensure burst_id is int for unique
    all_unique_burst_ids = np.array(df_original['temp_burst_id'].unique())
    
    if len(all_unique_burst_ids) == 0:
        logging.error("No unique burst_ids found in the dataset. Cannot split. Please check your data.")
        return

    logging.info(f"Found {len(all_unique_burst_ids)} unique burst_ids.")

    # --- Shuffle and Split burst_ids ---
    if args.seed is not None:
        logging.info(f"Using random seed: {args.seed} for shuffling burst_ids.")
        rng = np.random.default_rng(args.seed)
        rng.shuffle(all_unique_burst_ids)
    else:
        logging.info("No random seed provided. Shuffling burst_ids randomly.")
        np.random.shuffle(all_unique_burst_ids) # In-place shuffle

    num_total_bursts = len(all_unique_burst_ids)
    # Calculate number of test bursts first
    num_test_bursts = int(args.test_split_ratio * num_total_bursts)
    if num_test_bursts == 0 and num_total_bursts > 0 and args.test_split_ratio > 0:
        num_test_bursts = 1 # Ensure at least one test sample if ratio > 0 and data exists
        logging.warning(f"Calculated num_test_bursts is 0 with ratio {args.test_split_ratio}. Setting to 1.")

    num_train_bursts = num_total_bursts - num_test_bursts

    if num_train_bursts <= 0 or num_test_bursts < 0 : # num_test_bursts can be 0 if dataset is very small
        logging.error(f"Invalid split sizes: Train={num_train_bursts}, Test={num_test_bursts}. "
                      "Check split ratio or dataset size. Exiting.")
        return


    logging.info(f"Splitting bursts: Train={num_train_bursts} (~{(num_train_bursts/num_total_bursts)*100 if num_total_bursts > 0 else 0:.1f}%), "
                 f"Test={num_test_bursts} ({args.test_split_ratio*100:.1f}%)")

    test_burst_ids_set = set(all_unique_burst_ids[:num_test_bursts])
    train_burst_ids_set = set(all_unique_burst_ids[num_test_bursts:])
    
    # --- Filter Original DataFrame based on split burst_ids ---
    # We use the 'temp_burst_id' column we created for filtering.
    df_train = df_original[df_original['temp_burst_id'].isin(train_burst_ids_set)].copy()
    df_test = df_original[df_original['temp_burst_id'].isin(test_burst_ids_set)].copy()

    # Drop the temporary burst_id column before saving
    df_train.drop(columns=['temp_burst_id'], inplace=True)
    df_test.drop(columns=['temp_burst_id'], inplace=True)

    # --- Save the new CSV files ---
    # These will retain the original columns: 'rec_time' and 'json'
    if not df_train.empty:
        logging.info(f"Saving training data ({len(df_train)} rows) to: {train_output_path}")
        df_train.to_csv(train_output_path, sep=';', index=False)
    else:
        logging.warning(f"Training split resulted in an empty DataFrame. No {train_output_filename} was saved.")

    if not df_test.empty:
        logging.info(f"Saving test data ({len(df_test)} rows) to: {test_output_path}")
        df_test.to_csv(test_output_path, sep=';', index=False)
    else:
        logging.warning(f"Test split resulted in an empty DataFrame. No {test_output_filename} was saved.")
        
    logging.info("Dataset splitting complete.")
    logging.info(f"Training data: {train_output_path} ({len(train_burst_ids_set)} unique bursts)")
    logging.info(f"Test data: {test_output_path} ({len(test_burst_ids_set)} unique bursts)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split an original CSV dataset into training and test sets based on burst_id.")
    
    parser.add_argument('--input-csv-file', type=str, required=True,
                        help="Path to the original input CSV file (e.g., dataset/training.csv).")
    parser.add_argument('--output-dir', type=str, default='./dataset_split',
                        help="Directory where the split CSV files will be saved (default: ./dataset_split).")
    parser.add_argument('--train-output-name', type=str, default='training_NLOS.csv',
                        help="Filename for the output training CSV (default: training_nlos.csv).")
    parser.add_argument('--test-output-name', type=str, default='test_NLOS.csv',
                        help="Filename for the output test CSV (default: test_NLOS.csv).")
    parser.add_argument('--test-split-ratio', type=float, default=0.1,
                        help="Fraction of the dataset to allocate to the test set (e.g., 0.1 for 10%%). "
                             "The rest will be training data.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducible shuffling and splitting (default: None for random).")

    args = parser.parse_args()

    if not (0 < args.test_split_ratio < 1):
        logging.error("test_split_ratio must be between 0 and 1 (exclusive). Exiting.")
    else:
        main(args)
