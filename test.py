import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import json
import time
import os

# Import custom modules
from dataset import load_anchors, load_and_preprocess_data, MultiAnchorCIRDataset
from model import LocalizationModel 

log_dir = Path("new_checkpoints/checkpoints_TDOA_anchor_no_attentation") 
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "test_LOS-NLOS.log"

# Clear existing logging handlers to avoid duplication or inheritance
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'), # Save logs to file (overwrite each run)
        logging.StreamHandler()  # Also print to console
    ]
)

try:
    from train import calculate_metrics 
except ImportError:
    logging.warning("Could not import 'calculate_metrics' from train.py. Using fallback definition.")
    def calculate_metrics(predictions, targets):
        """Calculates Euclidean distance metrics."""
        if not isinstance(predictions, torch.Tensor): predictions = torch.tensor(predictions)
        if not isinstance(targets, torch.Tensor): targets = torch.tensor(targets)
        predictions = predictions.to(targets.device)
        if predictions.shape != targets.shape: raise ValueError(f"Shape mismatch: predictions {predictions.shape}, targets {targets.shape}")
        if predictions.shape[1] != 2: raise ValueError(f"Predictions/Targets must have 2 columns (x, y), got shape {predictions.shape}")
        
        distances = torch.sqrt(torch.sum((predictions - targets)**2, dim=1))
        distances = distances[~torch.isnan(distances)]
        
        if distances.numel() == 0: return {'mean': float('nan'), 'median': float('nan'), 'p75': float('nan'), 'p90': float('nan'), 'count': 0}

        mean_dist = torch.mean(distances).item()
        # Use correct median access for 1D tensor
        median_val_tensor = torch.median(distances)
        # Check if median returns a tuple (older torch) or just the value tensor
        if isinstance(median_val_tensor, tuple):
            median_dist = median_val_tensor[0].item() # Older torch: (values, indices)
        else:
             median_dist = median_val_tensor.item() # Newer torch: values tensor
             
        p75_dist = torch.quantile(distances, 0.75).item()
        p90_dist = torch.quantile(distances, 0.90).item()
        
        return {'mean': mean_dist, 'median': median_dist, 'p75': p75_dist, 'p90': p90_dist, 'count': distances.numel()}

def main(args):
    """Main testing script."""
    start_time = time.time()

    # --- Setup ---
    if not Path(args.checkpoint).is_file():
        logging.error(f"Checkpoint file not found: {args.checkpoint}")
        return

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # logging.info(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Checkpoint and Training Args ---
    logging.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if 'args' not in checkpoint:
        logging.error("Checkpoint does not contain training arguments ('args'). Cannot guarantee model consistency. Exiting.")
        return
    else:
        train_args = argparse.Namespace(**checkpoint['args'])
        logging.info("Loaded training arguments from checkpoint.")
        logging.info(f"  - Model Type: LocalizationModel (assumed)")
        logging.info(f"  - CNN Feat Dim: {getattr(train_args, 'cnn_feat_dim', 'Default')}")
        logging.info(f"  - Use Anchor Pos: {not getattr(train_args, 'no_anchor_pos', 'DefaultFalse')}")

    # --- Load Anchors ---
    logging.info("Loading anchor data...")
    base_path = Path(args.data_dir)
    anchor_file = base_path / "anchors.txt"
    anchors = load_anchors(anchor_file)
    num_anchors = len(anchors)

    # --- Initialize Model ---
    logging.info("Initializing model architecture based on training args...")
    model = LocalizationModel(
        cir_len=128,
        num_anchors=num_anchors,
        anchor_pos_dim=2,
        cnn_feature_dim=train_args.cnn_feat_dim,
        use_anchor_pos=not train_args.no_anchor_pos,
        # use_anchor_pos=train_args.use_anchor_pos,
        pos_embed_dim=train_args.pos_embed_dim,
        use_tdoa=not args.no_tdoa,
        agg_attention_dim=train_args.attn_dim,
        num_heads=8,
        mlp_hidden_dim=train_args.mlp_hidden_dim,
        dropout_rate=train_args.dropout,
        use_attention=not args.no_attention
    ).to(device)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model loaded with {num_params:,} parameters.")

    # --- Prepare Test Data ---
    test_files = {
        "score1": "test_NLOS.csv",
        "score2": "test_LOS-NLOS.csv"
    }
    if args.test_set not in test_files:
        logging.error(f"Invalid test_set argument: {args.test_set}. Choose from {list(test_files.keys())}")
        return

    test_file_name = test_files[args.test_set]
    test_file_path = base_path / test_file_name
    logging.info(f"Loading and preprocessing test data: {test_file_path}")

    test_df = load_and_preprocess_data(test_file_path, anchors, is_training=True) 

    if test_df is None or test_df.empty:
        logging.error(f"Failed to load or preprocess test data from {test_file_path}. Exiting.")
        return

    # === Check if ground truth is actually available ===
    has_ground_truth = 'ref_x' in test_df.columns and 'ref_y' in test_df.columns
    if has_ground_truth:
        logging.info("Ground truth columns (ref_x, ref_y) found in test data. Metrics will be calculated.")
        # Create Dataset with is_training=True to load GT
        test_dataset = MultiAnchorCIRDataset(test_df, anchors, num_anchors=num_anchors, is_training=True)
    else:
        logging.info("Ground truth columns not found in test data. Only predictions will be generated.")
         # Create Dataset with is_training=False
        test_dataset = MultiAnchorCIRDataset(test_df, anchors, num_anchors=num_anchors, is_training=False)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True if device == torch.device('cuda') else False) # Corrected device check

    # --- Run Inference ---
    logging.info(f"Running inference on {args.test_set} dataset...")
    all_predictions = []
    all_burst_ids = []
    all_targets = [] # List to store ground truth if available

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Testing on {args.test_set}")
        for batch in pbar:
            cir_data = batch['cir_data'].to(device)
            anchor_positions = batch['anchor_positions'].to(device)
            anchor_mask = batch['anchor_mask'].to(device)
            tdoa = batch['tdoa'].to(device)
            burst_ids = batch['burst_id']

            model_input = {
                'cir_data': cir_data,
                'anchor_positions': anchor_positions,
                'anchor_mask': anchor_mask,
                'tdoa': tdoa
            }

            predictions, _ = model(model_input)

            all_predictions.append(predictions.cpu())
            all_burst_ids.append(burst_ids)

            # Store ground truth if it exists in the batch
            if has_ground_truth and 'position' in batch:
                all_targets.append(batch['position'].cpu())

    # --- Format and Save Predictions ---
    logging.info("Inference complete. Formatting and saving predictions...")

    all_predictions_cat = torch.cat(all_predictions, dim=0).numpy()
    # Ensure burst IDs are collected correctly
    if isinstance(all_burst_ids[0], torch.Tensor):
        all_burst_ids_cat = torch.cat([b.cpu() for b in all_burst_ids], dim=0).numpy() 
    else: # Handle case where burst_ids might be lists/numpy arrays already
         all_burst_ids_cat = np.concatenate([np.array(b) for b in all_burst_ids])

    results_df = pd.DataFrame({
        'burst_id': all_burst_ids_cat.astype(int),
        'pred_x': all_predictions_cat[:, 0],
        'pred_y': all_predictions_cat[:, 1]
    })

    output_filename = f"predictions_{args.test_set}.csv"
    output_filepath = output_dir / output_filename
    results_df.to_csv(output_filepath, index=False)
    logging.info(f"Predictions saved to: {output_filepath}")

    # --- Calculate and Log Metrics (if GT available) ---
    if has_ground_truth:
        if not all_targets:
             logging.warning("Ground truth was expected but not collected from batches. Cannot calculate metrics.")
        else:
            logging.info("Calculating performance metrics...")
            all_targets_cat = torch.cat(all_targets, dim=0)

            # Ensure shapes match before calculating metrics
            if all_predictions_cat.shape[0] != all_targets_cat.shape[0]:
                logging.error(f"Prediction count ({all_predictions_cat.shape[0]}) does not match target count ({all_targets_cat.shape[0]}). Cannot calculate metrics accurately.")
            else:
                # Pass numpy array predictions and torch tensor targets to function
                metrics = calculate_metrics(all_predictions_cat, all_targets_cat)
                logging.info(f"--- Performance Metrics ({args.test_set}) ---")
                logging.info(f"  Mean Error:      {metrics['mean']:.3f} m")
                logging.info(f"  Median Error:    {metrics['median']:.3f} m")
                logging.info(f"  P75 Error (3Q):  {metrics['p75']:.3f} m")
                logging.info(f"  P90 Error:       {metrics['p90']:.3f} m")
                logging.info(f"  Sample Count:    {metrics['count']}")
                logging.info(f"------------------------------------")
                
                # Optionally save metrics to a file
                metrics_filename = f"metrics_{args.test_set}.json"
                metrics_filepath = output_dir / metrics_filename
                with open(metrics_filepath, 'w') as f:
                     json.dump(metrics, f, indent=4)
                logging.info(f"Metrics saved to: {metrics_filepath}")
    else:
        logging.info("No ground truth available for metric calculation.")


    # --- Finish ---
    test_duration = time.time() - start_time
    logging.info(f"Testing finished in {test_duration:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Localization Model and Calculate Metrics if GT Available")

    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.pth.tar file)')
    parser.add_argument('--test-set', type=str, required=True, choices=['score1', 'score2'],
                        help='Which test set to evaluate (score1, score2)')
    parser.add_argument('--data-dir', type=str, default='./dataset', help='Directory containing dataset files')
    parser.add_argument('--output-dir', type=str, default='predictions/LOS-NLOS/predictions_TDOA_anchor_no_attentation', help='Directory to save prediction & metrics files')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of dataloader workers')
    parser.add_argument('--no-tdoa', action='store_true', help='Disable using tdoa as features')
    parser.add_argument('--no-anchor-pos', action='store_true', help='Disable using anchor positions as features')
    # parser.add_argument('--use-anchor-pos', action='store_true', help='Enable using anchor positions as features')
    parser.add_argument('--no-attention', action='store_true', help='Disable attention and use mean fusion instead')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use (default: 0)')

    args = parser.parse_args()

    logging.info("----- Configuration -----")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    logging.info("-------------------------")

    main(args)
