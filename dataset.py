import pandas as pd
import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm 
import logging
import collections

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_anchors(anchor_file_path: Path) -> pd.DataFrame:
    """Loads anchor positions from the specified txt file."""
    try:
        anchors_df = pd.read_csv(anchor_file_path, sep=',', index_col='ID')
        anchors_df.index = anchors_df.index.astype(str)
        logging.info(f"Loaded anchor data (raw):\n{anchors_df}")
        anchors_df_2d = anchors_df[['p_x', 'p_y']].astype(np.float32)
        logging.info(f"Using 2D anchor data:\n{anchors_df_2d}")
        return anchors_df_2d
    except FileNotFoundError:
        logging.error(f"Anchor file not found at {anchor_file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading anchor file {anchor_file_path}: {e}")
        raise

def load_and_preprocess_data(csv_path: Path, anchor_df: pd.DataFrame, is_training=True) -> pd.DataFrame:
    """Loads CSV, parses JSON, and merges anchor info."""
    logging.info(f"Starting data loading from {csv_path}...")
    try:
        # Read CSV with specified separator
        data = pd.read_csv(csv_path, sep=';')
        logging.info(f"Read {len(data)} rows from {csv_path}.")

        # Keep original index if needed, otherwise reset
        data.reset_index(drop=True, inplace=True)

        # Parse JSON string in the 'json' column
        logging.info("Parsing JSON data (this may take a while)...")
        # Use tqdm to show progress for the apply step
        tqdm.pandas(desc="Parsing JSON")
        json_data = data['json'].progress_apply(json.loads)

        # Expand the JSON data into separate columns
        json_df = pd.json_normalize(json_data)
        logging.info("JSON parsing complete.")

        # Ensure required columns exist after parsing
        required_cols = ['burst_id', 'anch_id', 'csi_real', 'csi_imag', 'rec_time']
        if is_training:
            required_cols.extend(['ref_x', 'ref_y'])

        missing_cols = [col for col in required_cols if col not in json_df.columns]
        if missing_cols:
             raise ValueError(f"Missing required columns after JSON parsing: {missing_cols}")

        # Combine original rec_time and the parsed JSON data
        # Drop the original 'json' column as it's now expanded
        processed_df = pd.concat([data[['rec_time']].rename(columns={'rec_time':'orig_rec_time'}), json_df], axis=1)

        # Convert data types
        processed_df['burst_id'] = processed_df['burst_id'].astype(int)
        processed_df['anch_id'] = processed_df['anch_id'].astype(str)
        processed_df['rec_time'] = processed_df['rec_time'].astype(float)

        # Convert CSI lists to numpy arrays
        logging.info("Converting CSI lists to numpy arrays...")
        # Wrap in try-except to catch potential errors during conversion if lists are malformed
        try:
            processed_df['csi_real'] = processed_df['csi_real'].apply(lambda x: np.array(x, dtype=np.float32))
            processed_df['csi_imag'] = processed_df['csi_imag'].apply(lambda x: np.array(x, dtype=np.float32))
        except Exception as e:
            logging.error(f"Error converting CSI lists to numpy arrays: {e}. Check data format.")
            raise e
        logging.info("CSI conversion complete.")


        if is_training:
            processed_df['ref_x'] = processed_df['ref_x'].astype(np.float32)
            processed_df['ref_y'] = processed_df['ref_y'].astype(np.float32)

        logging.info(f"Processed DataFrame columns: {processed_df.columns.tolist()}")

        # Merge anchor positions (p_x, p_y from anchor_df)
        processed_df = processed_df.merge(anchor_df, left_on='anch_id', right_index=True, how='left')

        # Check for NaNs introduced by merge (anch_id in data not in anchors.txt)
        rows_before_drop = len(processed_df)
        nan_mask = processed_df[['p_x', 'p_y']].isnull().any(axis=1)
        nan_rows = nan_mask.sum()
        if nan_rows > 0:
            logging.warning(f"Found {nan_rows} rows with anch_id not present in anchors.txt after merge. Dropping these rows.")
            processed_df = processed_df[~nan_mask].copy() # Drop rows where p_x or p_y is NaN
            logging.info(f"Rows remaining after dropping NaNs: {len(processed_df)}")

        logging.info(f"Data loading and initial preprocessing complete for {csv_path}. Shape: {processed_df.shape}")
        return processed_df

    except FileNotFoundError:
        logging.error(f"Data file not found at {csv_path}")
        raise
    except Exception as e:
        logging.error(f"Error processing data file {csv_path}: {e}", exc_info=True)
        raise

class MultiAnchorCIRDataset(Dataset):
    """
    Dataset class.
    Groups CIR measurements by burst_id from multiple anchors.
    """
    def __init__(self, data_df: pd.DataFrame, anchor_df: pd.DataFrame, cir_len=128, num_anchors=8, is_training=True):
        """
        Args:
            data_df (pd.DataFrame): Preprocessed dataframe containing CIR data and positions.
            anchor_df (pd.DataFrame): Dataframe with anchor IDs and 2D positions (p_x, p_y).
            cir_len (int): Expected length of the CIR signal (128 samples).
            num_anchors (int): Total number of anchors in the system (8).
            is_training (bool): Flag indicating if loading training data (requires ref_x, ref_y).
        """
        self.data_df = data_df
        self.anchor_df = anchor_df
        self.cir_len = cir_len
        self.num_anchors = num_anchors
        self.is_training = is_training

        # Get a sorted list of unique anchor IDs from the anchor file
        self.anchor_ids_ordered = sorted(anchor_df.index.tolist())
        # Create a mapping from anchor ID to its index (0 to num_anchors-1)
        self.anchor_id_to_index = {anchor_id: i for i, anchor_id in enumerate(self.anchor_ids_ordered)}

        logging.info("Grouping data by burst_id...")
        # Group data by burst_id
        self.grouped_data = data_df.groupby('burst_id')

        # Get a list of unique burst_ids to determine the dataset size
        self.burst_ids = list(self.grouped_data.groups.keys())

        logging.info(f"Dataset initialized with {len(self.burst_ids)} unique bursts.")

        # Store anchor positions in a tensor for easy access
        # Order according to anchor_ids_ordered
        anchor_pos_list = [anchor_df.loc[anchor_id].values for anchor_id in self.anchor_ids_ordered]
        self.anchor_positions = torch.tensor(np.array(anchor_pos_list), dtype=torch.float32) # Shape: [num_anchors, 2]

    def __len__(self):
        """Returns the total number of unique bursts."""
        return len(self.burst_ids)


    def __getitem__(self, idx):
        burst_id = self.burst_ids[idx]
        try:
            burst_group = self.grouped_data.get_group(burst_id)

            cir_tensor = torch.zeros((self.num_anchors, 2, self.cir_len), dtype=torch.float32)
            anchor_mask = torch.zeros((self.num_anchors,), dtype=torch.float32)
            tdoa_vector = torch.zeros((self.num_anchors,), dtype=torch.float32)

            toas = [row['TOA'] for _, row in burst_group.iterrows()]
            toa_dict = {row['anch_id']: row['TOA'] for _, row in burst_group.iterrows()}
            min_toa = min(toa_dict.values())

            gt_position = None

            for _, row in burst_group.iterrows():
                anchor_id = row['anch_id']
                if anchor_id in self.anchor_id_to_index:
                    anchor_idx = self.anchor_id_to_index[anchor_id]

                    csi_real = np.array(row['csi_real'], dtype=np.float32)
                    csi_imag = np.array(row['csi_imag'], dtype=np.float32)

                    if csi_real.shape[0] != self.cir_len or csi_imag.shape[0] != self.cir_len:
                        continue

                    cir_combined = np.stack([csi_real, csi_imag], axis=0)
                    cir_combined_tensor = torch.from_numpy(cir_combined)
                    mean = torch.mean(cir_combined_tensor.float())
                    std = torch.std(cir_combined_tensor.float())
                    cir_normalized = (cir_combined_tensor - mean) / (std + 1e-8)

                    cir_tensor[anchor_idx] = cir_normalized.type(torch.float32)
                    anchor_mask[anchor_idx] = 1.0

                    # --------------------- TDoA calculation ---------------------
                    tdoa = row['TOA'] - min_toa
                    tdoa_vector[anchor_idx] = tdoa  # <--- Fill aligned TDoA vector

                    if self.is_training:
                        current_pos = torch.tensor([row['ref_x'], row['ref_y']], dtype=torch.float32)
                        if gt_position is None:
                            gt_position = current_pos
                        elif not torch.allclose(gt_position, current_pos, atol=1e-6):
                            logging.warning(f"GT mismatch in burst {burst_id}")

                else:
                    logging.warning(f"Unknown anchor ID {anchor_id} for burst {burst_id}")

            sample = {
                'cir_data': cir_tensor,  # [8, 2, 128]
                'tdoa': tdoa_vector,     # [8]
                'anchor_mask': anchor_mask,
                'anchor_positions': self.anchor_positions,  # [8, 2]
                'burst_id': burst_id
            }

            if self.is_training:
                if gt_position is None:
                    logging.error(f"No valid GT position in burst {burst_id}")
                    sample['position'] = torch.zeros((2,), dtype=torch.float32)
                else:
                    sample['position'] = gt_position

            return sample

        except Exception as e:
            logging.error(f"Error with burst {burst_id}: {e}", exc_info=True)
            return {
                'cir_data': torch.zeros((self.num_anchors, 2, self.cir_len)),
                'tdoa': torch.zeros((self.num_anchors,)),
                'anchor_mask': torch.zeros((self.num_anchors,)),
                'anchor_positions': self.anchor_positions,
                'burst_id': -1,
                'position': torch.zeros((2,), dtype=torch.float32) if self.is_training else None
            }


# --- Example Usage ---
if __name__ == '__main__':
    # Assumes 'dataset' folder is in the same directory as the script
    base_path = Path("./dataset")
    if not base_path.exists():
        # If running from a different directory, adjust the path
        script_dir = Path(__file__).parent
        base_path = script_dir / "dataset"
        logging.info(f"Script dir: {script_dir}, trying dataset path: {base_path}")
        if not base_path.exists():
             raise FileNotFoundError(f"Dataset directory not found at {base_path}. Please adjust 'base_path' or run from the correct directory.")

    anchor_file = base_path / "anchors.txt"
    training_file = base_path / "training.csv"
    experimental_file = base_path / "experimental_trial.csv"

    # --- Load Anchors ---
    anchors = load_anchors(anchor_file)
    num_anchors = len(anchors)
    if num_anchors != 8:
        logging.warning(f"Expected 8 anchors, but found {num_anchors} in {anchor_file}")

    # --- Load and Preprocess Training Data ---
    training_df = load_and_preprocess_data(training_file, anchors, is_training=True)

    # --- Create Dataset ---
    if training_df is not None and not training_df.empty:
        train_dataset = MultiAnchorCIRDataset(training_df, anchors, num_anchors=num_anchors, is_training=True)

        # --- Create DataLoader ---
        batch_size = 16 # Example batch size
        # Set num_workers > 0 for parallel loading (start with 0 for debugging)
        # persistent_workers=True can speed up epoch starts if num_workers > 0
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=0, persistent_workers=False,
                                      pin_memory=True if torch.cuda.is_available() else False) # Pin memory if using GPU

        logging.info(f"\n--- Example Batch ---")
        # Fetch one batch to test
        try:
            example_batch = next(iter(train_dataloader))

            logging.info(f"Batch CIR data shape: {example_batch['cir_data'].shape}")
            logging.info(f"Batch anchor positions shape: {example_batch['anchor_positions'].shape}")
            logging.info(f"Batch anchor mask shape: {example_batch['anchor_mask'].shape}")
            if 'position' in example_batch:
                 logging.info(f"Batch position shape: {example_batch['position'].shape}")
                 # Test the GT consistency check with allclose on a real batch item
                 first_item_burst = example_batch['burst_id'][0].item()
                 burst_group = training_df[training_df['burst_id'] == first_item_burst]
                 if len(burst_group) > 1:
                     pos1 = torch.tensor([burst_group.iloc[0]['ref_x'], burst_group.iloc[0]['ref_y']], dtype=torch.float32)
                     pos2 = torch.tensor([burst_group.iloc[1]['ref_x'], burst_group.iloc[1]['ref_y']], dtype=torch.float32)
                     logging.info(f"Testing GT consistency for burst {first_item_burst}: pos1={pos1}, pos2={pos2}")
                     logging.info(f"torch.equal: {torch.equal(pos1, pos2)}")
                     logging.info(f"torch.allclose(atol=1e-6): {torch.allclose(pos1, pos2, atol=1e-6)}")


            logging.info(f"Batch burst_ids: {example_batch['burst_id']}")


            # Check mask usage:
            if example_batch['anchor_mask'].numel() > 0: # Check if mask is not empty
                print(f"Example anchor mask for first item in batch:\n{example_batch['anchor_mask'][0]}")
            else:
                logging.warning("Anchor mask is empty in the example batch.")

        except StopIteration:
            logging.error("DataLoader yielded no batches. Check dataset size and preprocessing.")
        except Exception as e:
           logging.error(f"Error fetching batch from DataLoader: {e}", exc_info=True)

    else:
        logging.error("Training DataFrame is empty or None. Cannot create dataset.")
