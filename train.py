import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

# --- Configuration ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_metrics(predictions, targets):
    """Calculates Euclidean distance metrics."""
    # Ensure inputs are torch tensors on the same device
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    
    predictions = predictions.to(targets.device) # Ensure device match
    
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape}, targets {targets.shape}")
    if predictions.shape[1] != 2:
         raise ValueError(f"Predictions/Targets must have 2 columns (x, y), got shape {predictions.shape}")

    distances = torch.sqrt(torch.sum((predictions - targets)**2, dim=1))
    
    # Handle potential NaN distances if predictions/targets were NaN
    distances = distances[~torch.isnan(distances)] 
    
    if distances.numel() == 0: # If all were NaN or empty input
        return {'mean': float('nan'), 'median': float('nan'), 'p75': float('nan'), 'p90': float('nan'), 'count': 0}

    mean_dist = torch.mean(distances).item()
    median_dist = torch.median(distances).item() # .values needed for torch >= 1.7
    p75_dist = torch.quantile(distances, 0.75).item() # 3rd Quartile
    p90_dist = torch.quantile(distances, 0.90).item()
    
    return {
        'mean': mean_dist, 
        'median': median_dist, 
        'p75': p75_dist, 
        'p90': p90_dist,
        'count': distances.numel() # Number of valid distances computed
        }

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar', checkpoint_dir='checkpoints'):
    """Saves model and training parameters."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = checkpoint_dir / filename
    torch.save(state, filepath)
    logging.info(f"Checkpoint saved to {filepath}")
    if is_best:
        best_filepath = checkpoint_dir / best_filename
        torch.save(state, best_filepath)
        logging.info(f"Best model saved to {best_filepath}")


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs):
    """Runs one training epoch."""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", leave=False)
    for batch in pbar:
        cir_data = batch['cir_data'].to(device)
        anchor_positions = batch['anchor_positions'].to(device)
        anchor_mask = batch['anchor_mask'].to(device)
        tdoa = batch['tdoa'].to(device) 
        targets = batch['position'].to(device)

        model_input = {
            'cir_data': cir_data,
            'anchor_positions': anchor_positions,
            'anchor_mask': anchor_mask,
            'tdoa': tdoa
        }

        predictions, _ = model(model_input)
        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_predictions.append(predictions.detach().cpu())
        all_targets.append(targets.detach().cpu())
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    all_predictions_cat = torch.cat(all_predictions, dim=0)
    all_targets_cat = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_predictions_cat, all_targets_cat)

    logging.info(f"Epoch {epoch+1}/{total_epochs} [Train] Avg Loss: {avg_loss:.4f}, "
                 f"MeanDist: {metrics['mean']:.3f}m, P75: {metrics['p75']:.3f}m")
    return avg_loss, metrics


def validate(model, dataloader, criterion, device):
    """Runs validation."""
    model.eval()  # Set model to evaluation mode
    
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad(): 
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for batch in pbar:
            # Move batch to device
            cir_data = batch['cir_data'].to(device)
            anchor_positions = batch['anchor_positions'].to(device)
            anchor_mask = batch['anchor_mask'].to(device)
            targets = batch['position'].to(device)

            model_input = {
                'cir_data': cir_data,
                'anchor_positions': anchor_positions,
                'anchor_mask': anchor_mask,
                'tdoa': batch['tdoa'].to(device)  
            }

            # --- Forward Pass ---
            predictions, _ = model(model_input)

            # --- Calculate Loss ---
            loss = criterion(predictions, targets)

            # --- Accumulate Loss and Results ---
            total_loss += loss.item()
            all_predictions.append(predictions.cpu()) # Move to CPU
            all_targets.append(targets.cpu())

    avg_loss = total_loss / len(dataloader)
    
    # Concatenate results from all batches
    all_predictions_cat = torch.cat(all_predictions, dim=0)
    all_targets_cat = torch.cat(all_targets, dim=0)
    
    # Calculate metrics on validation results
    metrics = calculate_metrics(all_predictions_cat, all_targets_cat)
    
    logging.info(f"[Validation] Avg Loss: {avg_loss:.4f}, MeanDist: {metrics['mean']:.3f}m, " 
                 f"MedianDist: {metrics['median']:.3f}m, P75: {metrics['p75']:.3f}m, "
                 f"P90: {metrics['p90']:.3f}m (Count: {metrics['count']})")
                 
    return avg_loss, metrics


def main(args):
    """Main training script."""
    
    # --- Setup ---
    start_time = time.time()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        logging.info(f"Using random seed: {args.seed}")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # logging.info(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Data ---
    logging.info("Loading anchor data...")
    base_path = Path(args.data_dir)
    anchor_file = base_path / "anchors.txt"
    anchors = load_anchors(anchor_file)
    num_anchors = len(anchors)

    logging.info("Loading and preprocessing training data...")
    training_file = base_path / "training_NLOS.csv"
    # Load the entire dataset first
    full_processed_df = load_and_preprocess_data(training_file, anchors, is_training=True)

    if full_processed_df is None or full_processed_df.empty:
        logging.error("Failed to load or preprocess training data. Exiting.")
        return

    # Split the preprocessed data (indices or the dataset object)
    total_size = len(full_processed_df['burst_id'].unique()) # Get unique bursts from preprocessed df
    val_split = args.val_split
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    logging.info(f"Total unique bursts: {total_size}")
    logging.info(f"Splitting data: Train={train_size}, Validation={val_size} ({val_split*100:.1f}%)")
    
    # Create the full dataset object
    full_dataset = MultiAnchorCIRDataset(full_processed_df, anchors, num_anchors=num_anchors, is_training=True)

    # Perform random split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                             generator=torch.Generator().manual_seed(args.seed or 42)) # Use seed for reproducible split

    logging.info("Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, # Often use larger batch size for validation
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)

    # --- Model ---
    logging.info("Initializing model...")
    model = LocalizationModel(
        cir_len=128,
        num_anchors=num_anchors,
        anchor_pos_dim=2,
        cnn_feature_dim=args.cnn_feat_dim,
        use_anchor_pos=not args.no_anchor_pos,
        # use_anchor_pos=args.use_anchor_pos,  
        pos_embed_dim=args.pos_embed_dim,
        use_tdoa=not args.no_tdoa, 
        agg_attention_dim=args.attn_dim,
        num_heads=8,
        mlp_hidden_dim=args.mlp_hidden_dim,
        dropout_rate=args.dropout,
        use_attention=not args.no_attention  
    ).to(device)
    
    # Log model structure and parameter count
    # print(model) 
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model initialized with {num_params:,} trainable parameters.")


    # --- Loss Function ---
    criterion = nn.SmoothL1Loss().to(device) 

    # --- Optimizer ---
    # AdamW is often preferred over Adam
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Scheduler ---
    # Reduce LR when validation metric (P75 error) plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)

    # --- Training Loop ---
    best_val_metric = float('inf') # We want to minimize P75 distance
    start_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_p75': [], 'lr': []}

    patience = args.patiencee
    early_stop_counter = 0
    
    # Optional: Resume from checkpoint
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_file():
            logging.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler']) # Load scheduler state
            start_epoch = checkpoint['epoch']
            best_val_metric = checkpoint['best_val_metric']
            history = checkpoint.get('history', history) # Load history if available
            logging.info(f"Resumed from epoch {start_epoch}, best val P75: {best_val_metric:.3f}m")
        else:
            logging.warning(f"Resume checkpoint not found at {args.resume}. Starting from scratch.")


    logging.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, args.epochs)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Update history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_p75'].append(val_metrics['p75'])
        history['lr'].append(current_lr)
        
        # Update learning rate scheduler based on validation P75 error
        scheduler.step(val_metrics['p75'])

        # Save checkpoint
        is_best = val_metrics['p75'] < best_val_metric
        if is_best:
            best_val_metric = val_metrics['p75']
            logging.info(f"*** New best validation P75: {best_val_metric:.3f}m at epoch {epoch+1} ***")
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), # Save scheduler state
            'best_val_metric': best_val_metric,
            'history': history, # Save training history
            'args': vars(args) # Save arguments used for this run
        }, is_best, checkpoint_dir=checkpoint_dir)
        
        epoch_duration = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_duration:.2f}s. LR: {current_lr:.1e}")
        if not is_best:
            early_stop_counter += 1
            logging.info(f"Validation P75 did not improve. Early stop counter: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                logging.info("Early stopping triggered. Stopping training early.")
                break
        else:
            early_stop_counter = 0

    # --- Finish ---
    training_duration = time.time() - start_time
    logging.info(f"Training finished in {training_duration / 3600:.2f} hours.")
    logging.info(f"Best validation P75 (3rd Quartile) distance achieved: {best_val_metric:.3f}m")
    
    # Save history to a file
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    logging.info(f"Training history saved to {history_path}")


if __name__ == "__main__":

    log_dir = Path("new_checkpoints/checkpoints_TDOA_no_anchor_attentation")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"

    # Clear existing logging handlers if needed
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()  # Console output
        ]
    )
    
    parser = argparse.ArgumentParser(description="Train Localization Model")

    # Data args
    parser.add_argument('--data-dir', type=str, default='./dataset', help='Directory containing dataset files')
    parser.add_argument('--val-split', type=float, default=0.1, help='Fraction of data to use for validation (default: 0.1)')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of dataloader workers (set > 0 for parallel loading)')
    
    # Model args (match defaults in model.py or provide overrides)
    parser.add_argument('--cnn-feat-dim', type=int, default=128, help='Feature dimension from CNN')
    parser.add_argument('--pos-embed-dim', type=int, default=16, help='Dimension for anchor position embedding')
    parser.add_argument('--attn-dim', type=int, default=64, help='Hidden dimension for attention network')
    parser.add_argument('--mlp-hidden-dim', type=int, default=256, help='Hidden dimension for final MLP')
    parser.add_argument('--no-tdoa', action='store_true', help='Disable using tdoa as features')
    parser.add_argument('--no-anchor-pos', action='store_true', help='Disable using anchor positions as features')
    # parser.add_argument('--use-anchor-pos', action='store_true', help='Enable using anchor positions as features')
    parser.add_argument('--no-attention', action='store_true', help='Disable attention and use mean fusion instead')

    # Training args
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--patiencee', type=int, default=20, help='Number of epochs to wait for improvement before early stopping')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay (AdamW)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate in MLP')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use (default: 0)')

    # Checkpoint args
    parser.add_argument('--checkpoint-dir', type=str, default='new_checkpoints/checkpoints_TDOA_no_anchor_attentation', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    
    # Misc args
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()
    
    # Print arguments used
    logging.info("----- Configuration -----")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    logging.info("-------------------------")

    main(args)
