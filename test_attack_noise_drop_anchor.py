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
import time # Ensure time is imported
import os

# Import custom modules
from dataset import load_anchors, load_and_preprocess_data, MultiAnchorCIRDataset
from model import LocalizationModel

# --- Constants for Attacks ---
SPEED_OF_LIGHT = 299792458.0
CSI_SAMPLING_RATE_HZ = 184.32e6
SPOOFING_DISTANCE_CLOSER_M = 5.0
SPOOFING_AMPLITUDE_SCALE = 0.5

# --- Logging Configuration ---
log_dir_test = Path("./test_results_final")
log_dir_test.mkdir(parents=True, exist_ok=True)
log_file_test = log_dir_test / "test.log"

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_test, mode='w'),
        logging.StreamHandler()
    ]
)

def inject_spoofing_signal(csi_real_np: np.ndarray, csi_imag_np: np.ndarray,
                           sampling_rate_hz: float, distance_closer_m: float,
                           amplitude_scale: float, cir_len: int = 128):
    if len(csi_real_np) != cir_len or len(csi_imag_np) != cir_len:
        logging.warning(f"CIR length mismatch during spoofing. Expected {cir_len}, got {len(csi_real_np)}. Skipping spoofing for this CIR.")
        return csi_real_np, csi_imag_np
    cir_complex = csi_real_np + 1j * csi_imag_np
    if np.all(np.abs(cir_complex) < 1e-9):
        logging.debug("CIR is all zeros or near-zeros. Skipping spoofing injection.")
        return csi_real_np, csi_imag_np
    cir_magnitude = np.abs(cir_complex)
    main_peak_idx = np.argmax(cir_magnitude)
    main_peak_value_complex = cir_complex[main_peak_idx]
    if np.abs(main_peak_value_complex) < 1e-9:
        logging.debug("Main peak is zero in CIR, spoofing signal will also be zero. No effective change.")
        return csi_real_np, csi_imag_np
    time_delay_change = distance_closer_m / SPEED_OF_LIGHT
    sample_shift = int(round(time_delay_change * sampling_rate_hz))
    spoofed_peak_idx = main_peak_idx - sample_shift
    modified_cir_complex = cir_complex.copy()
    if 0 <= spoofed_peak_idx < cir_len:
        spoofed_signal_component = main_peak_value_complex * amplitude_scale
        modified_cir_complex[spoofed_peak_idx] += spoofed_signal_component
        logging.debug(f"Spoofing: Original main peak at index {main_peak_idx}. "
                      f"Spoofed peak (amplitude {np.abs(spoofed_signal_component):.2f}) "
                      f"injected at index {spoofed_peak_idx} (shift of {sample_shift} samples).")
    else:
        logging.debug(f"Spoofed peak index {spoofed_peak_idx} is out of bounds (0-{cir_len-1}). Original peak at {main_peak_idx}. No spoofing injection for this CIR.")
    return np.real(modified_cir_complex), np.imag(modified_cir_complex)

def inject_gaussian_noise(csi_real_np: np.ndarray, csi_imag_np: np.ndarray,
                          sigma: float, cir_len: int = 128):
    if len(csi_real_np) != cir_len or len(csi_imag_np) != cir_len:
        logging.warning(f"CIR length mismatch during noise injection. Expected {cir_len}, got {len(csi_real_np)}. Returning original.")
        return csi_real_np, csi_imag_np
    noise_real = np.random.normal(0, sigma, cir_len)
    noise_imag = np.random.normal(0, sigma, cir_len)
    noisy_csi_real = csi_real_np + noise_real
    noisy_csi_imag = csi_imag_np + noise_imag
    logging.debug(f"Noise injection: Added Gaussian noise with sigma={sigma} to CIR.")
    return noisy_csi_real, noisy_csi_imag


try:
    from train import calculate_metrics
except ImportError:
    logging.warning("Could not import 'calculate_metrics' from train.py. Using fallback definition in test.py.")
    def calculate_metrics(predictions, targets):
        if not isinstance(predictions, torch.Tensor): predictions = torch.tensor(predictions, dtype=torch.float32)
        if not isinstance(targets, torch.Tensor): targets = torch.tensor(targets, dtype=torch.float32)
        predictions = predictions.to(targets.device)
        if predictions.shape != targets.shape: raise ValueError(f"Shape mismatch: predictions {predictions.shape}, targets {targets.shape}")
        if predictions.ndim == 1 and targets.ndim == 1:
            predictions = predictions.unsqueeze(0)
            targets = targets.unsqueeze(0)
        if predictions.shape[1] != 2: raise ValueError(f"Predictions/Targets must have 2 columns (x, y), got shape {predictions.shape}")
        distances = torch.sqrt(torch.sum((predictions - targets)**2, dim=1))
        distances = distances[~torch.isnan(distances)]
        if distances.numel() == 0: return {'mean': float('nan'), 'median': float('nan'), 'p75': float('nan'), 'p90': float('nan'), 'count': 0}
        mean_dist = torch.mean(distances).item()
        median_val_tensor = torch.median(distances)
        median_dist = median_val_tensor[0].item() if isinstance(median_val_tensor, tuple) else median_val_tensor.item()
        p75_dist = torch.quantile(distances, 0.75).item()
        p90_dist = torch.quantile(distances, 0.90).item()
        return {'mean': mean_dist, 'median': median_dist, 'p75': p75_dist, 'p90': p90_dist, 'count': distances.numel()}

def main(args):
    script_start_time = time.time()
    if not Path(args.checkpoint).is_file():
        logging.error(f"Main checkpoint file not found: {args.checkpoint}")
        return

    if torch.cuda.is_available() and args.gpu >= 0 :
        if args.gpu >= torch.cuda.device_count():
            logging.warning(f"GPU ID {args.gpu} is invalid, using GPU 0 instead.")
            args.gpu = 0
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading main checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'args' not in checkpoint:
        logging.error("Main checkpoint does not contain 'args'. Exiting.")
        return
    train_args = argparse.Namespace(**checkpoint['args'])
    logging.info("Loaded training arguments for main model (model under test):")
    for arg_name, value in vars(train_args).items(): logging.info(f"  - {arg_name}: {value}")

    logging.info("Loading anchor data...")
    base_path = Path(args.data_dir)
    anchor_file = base_path / "anchors.txt"
    anchors = load_anchors(anchor_file)
    num_anchors = len(anchors)
    cir_len_from_dataset = 128

    logging.info("Initializing main model (model under test) architecture...")
    model = LocalizationModel(
        cir_len=cir_len_from_dataset, num_anchors=num_anchors, anchor_pos_dim=2,
        cnn_feature_dim=getattr(train_args, 'cnn_feat_dim', 128),
        use_anchor_pos=not getattr(train_args, 'no_anchor_pos', False),
        pos_embed_dim=getattr(train_args, 'pos_embed_dim', 16),
        use_tdoa=not getattr(train_args, 'no_tdoa', False),
        tdoa_embed_dim=getattr(train_args, 'tdoa_embed_dim', 16),
        agg_attention_dim=getattr(train_args, 'attn_dim', 64),
        num_heads=getattr(train_args, 'num_heads', 8),
        mlp_hidden_dim=getattr(train_args, 'mlp_hidden_dim', 256),
        dropout_rate=getattr(train_args, 'dropout', 0.3),
        use_attention=not getattr(train_args, 'no_attention', False)
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    logging.info(f"Main model loaded with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    test_files = {"score1": "test_NLOS.csv", "score2": "test_LOS-NLOS.csv"}
    if args.test_set not in test_files:
        logging.error(f"Invalid test_set: {args.test_set}. Choose from {list(test_files.keys())}")
        return
    test_file_path = base_path / test_files[args.test_set]
    logging.info(f"Loading and preprocessing test data: {test_file_path}")
    test_df = load_and_preprocess_data(test_file_path, anchors, is_training=True)
    if test_df is None or test_df.empty:
        logging.error(f"Failed to load or preprocess test data from {test_file_path}. Exiting.")
        return
    has_ground_truth = 'ref_x' in test_df.columns and 'ref_y' in test_df.columns
    test_dataset = MultiAnchorCIRDataset(test_df, anchors, num_anchors=num_anchors, cir_len=cir_len_from_dataset, is_training=has_ground_truth)
    # Pin memory only if on CUDA and num_workers > 0 for best effect
    pin_memory_flag = device.type == 'cuda' and args.num_workers > 0
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_memory_flag, 
                             persistent_workers=args.num_workers > 0 if args.num_workers > 0 else False)


    active_attack_type = None
    num_attack_flags = sum([args.spoofing_attack, args.noise_attack, args.drop_influential_anchor_attack, args.drop_random_anchor_attack])
    if num_attack_flags > 1:
        logging.error("Multiple attack types specified. Please choose only one. Exiting.")
        return
    elif args.spoofing_attack: active_attack_type = "spoofing"
    elif args.noise_attack: active_attack_type = "noise"
    elif args.drop_influential_anchor_attack: active_attack_type = "drop_influential"
    elif args.drop_random_anchor_attack: active_attack_type = "drop_random"

    # Variables for inference time calculation
    total_inference_time_ms = 0.0
    total_samples_processed_for_inf_time = 0 # Use a different counter for samples included in inf time

    if active_attack_type:
        logging.info(f"--- Running Simulated Attack: {active_attack_type.upper()} ---")
        attack_strategy_for_selection = None 
        model_for_attention_guidance = None 
        main_model_uses_attention = not getattr(train_args, 'no_attention', False)

        if active_attack_type == "drop_random":
            attack_strategy_for_selection = 'random_selection_for_drop'
            logging.info("Attack strategy: Dropping a randomly selected ACTIVE anchor for each sample.")
        elif main_model_uses_attention:
            attack_strategy_for_selection = 'attention_main'
            model_for_attention_guidance = model
            logging.info(f"Main model ({args.checkpoint}) uses attention. It will guide target anchor selection.")
        elif args.guiding_attention_checkpoint:
            attack_strategy_for_selection = 'attention_guiding'
            logging.info(f"Main model non-attentional. Checking guiding model: {args.guiding_attention_checkpoint}")
            if not Path(args.guiding_attention_checkpoint).is_file(): logging.error(f"Guiding checkpoint not found: {args.guiding_attention_checkpoint}. Exiting."); return
            guiding_checkpoint = torch.load(args.guiding_attention_checkpoint, map_location=device)
            if 'args' not in guiding_checkpoint: logging.error("Guiding checkpoint missing 'args'. Exiting."); return
            guiding_train_args = argparse.Namespace(**guiding_checkpoint['args'])
            if getattr(guiding_train_args, 'no_attention', False): logging.error(f"Guiding model ({args.guiding_attention_checkpoint}) also non-attentional. Cannot guide. Exiting."); return
            logging.info("Initializing guiding attention model architecture...")
            guiding_model = LocalizationModel(cir_len=cir_len_from_dataset, num_anchors=num_anchors, anchor_pos_dim=2, cnn_feature_dim=getattr(guiding_train_args, 'cnn_feat_dim', 128), use_anchor_pos=not getattr(guiding_train_args, 'no_anchor_pos', False), pos_embed_dim=getattr(guiding_train_args, 'pos_embed_dim', 16), use_tdoa=not getattr(guiding_train_args, 'no_tdoa', False), tdoa_embed_dim=getattr(guiding_train_args, 'tdoa_embed_dim', 16), agg_attention_dim=getattr(guiding_train_args, 'attn_dim', 64), num_heads=getattr(guiding_train_args, 'num_heads', 8), mlp_hidden_dim=getattr(guiding_train_args, 'mlp_hidden_dim', 256), dropout_rate=getattr(guiding_train_args, 'dropout', 0.3), use_attention=True).to(device)
            guiding_model.load_state_dict(guiding_checkpoint['state_dict'])
            guiding_model.eval()
            model_for_attention_guidance = guiding_model
            logging.info(f"Guiding model ({args.guiding_attention_checkpoint}) will guide target anchor selection.")
        elif active_attack_type == "drop_influential":
            logging.error(f"Attack 'drop_influential' needs attention guidance, but main model non-attentional and no guiding checkpoint. Exiting.")
            return
        else: 
            attack_strategy_for_selection = 'random_selection_for_cir_mod'
            logging.info(f"Main model non-attentional, no guiding model. Target for {active_attack_type} will be RANDOMLY selected ACTIVE anchor.")

        all_attacked_predictions_list, all_attacked_burst_ids_list, all_attacked_targets_list = [], [], []
        pbar_desc = f"{active_attack_type.replace('_', ' ').capitalize()} ({attack_strategy_for_selection.replace('_', ' ')}) on {args.test_set}"
        if active_attack_type == "noise": pbar_desc += f" sigma={args.noise_sigma}"
        
        with torch.no_grad():
            pbar_attack = tqdm(test_loader, desc=pbar_desc)
            for batch_idx, batch in enumerate(pbar_attack):
                current_batch_size = batch['cir_data'].size(0)
                attacked_cir_data_cpu = batch['cir_data'].clone()
                attacked_anchor_mask_cpu = batch['anchor_mask'].clone()
                target_anchor_indices_batch = torch.full((current_batch_size,), -1, dtype=torch.long, device='cpu')

                if attack_strategy_for_selection in ['attention_main', 'attention_guiding']:
                    # Note: Time taken by the guiding model (if different from main model) is NOT part of the main model's inference time.
                    # We only time the final forward pass of the 'model' under test.
                    input_for_attention = {'cir_data': batch['cir_data'].to(device), 'anchor_positions': batch['anchor_positions'].to(device), 'anchor_mask': batch['anchor_mask'].to(device), 'tdoa': batch['tdoa'].to(device)}
                    _, attn_weights_normal = model_for_attention_guidance(input_for_attention) # This uses guiding model or main model
                    if attn_weights_normal is None: logging.error(f"Attention guidance model returned None for weights in batch {batch_idx}. Skipping."); continue
                    anchor_influence = attn_weights_normal.sum(dim=1)
                    target_anchor_indices_batch_gpu = torch.argmax(anchor_influence, dim=1)
                    target_anchor_indices_batch = target_anchor_indices_batch_gpu.cpu()
                elif attack_strategy_for_selection in ['random_selection_for_drop', 'random_selection_for_cir_mod']:
                    for i in range(current_batch_size):
                        active_anchors_sample = (batch['anchor_mask'][i] == 1.0).nonzero(as_tuple=True)[0]
                        if active_anchors_sample.numel() > 0:
                            rand_idx = torch.randint(0, active_anchors_sample.numel(), (1,)).item()
                            target_anchor_indices_batch[i] = active_anchors_sample[rand_idx].item()
                
                for i in range(current_batch_size):
                    target_idx_sample = target_anchor_indices_batch[i].item()
                    if target_idx_sample == -1: continue 
                    if batch['anchor_mask'][i, target_idx_sample].item() == 0: logging.debug(f"Target anchor {target_idx_sample} for attack on sample {i} is already masked. No action."); continue
                    if active_attack_type == "spoofing":
                        real_orig, imag_orig = attacked_cir_data_cpu[i,target_idx_sample,0,:].numpy(), attacked_cir_data_cpu[i,target_idx_sample,1,:].numpy()
                        mod_r, mod_i = inject_spoofing_signal(real_orig, imag_orig, CSI_SAMPLING_RATE_HZ, SPOOFING_DISTANCE_CLOSER_M, SPOOFING_AMPLITUDE_SCALE, cir_len_from_dataset)
                        attacked_cir_data_cpu[i,target_idx_sample,0,:] = torch.from_numpy(mod_r); attacked_cir_data_cpu[i,target_idx_sample,1,:] = torch.from_numpy(mod_i)
                    elif active_attack_type == "noise":
                        real_orig, imag_orig = attacked_cir_data_cpu[i,target_idx_sample,0,:].numpy(), attacked_cir_data_cpu[i,target_idx_sample,1,:].numpy()
                        mod_r, mod_i = inject_gaussian_noise(real_orig, imag_orig, args.noise_sigma, cir_len_from_dataset)
                        attacked_cir_data_cpu[i,target_idx_sample,0,:] = torch.from_numpy(mod_r); attacked_cir_data_cpu[i,target_idx_sample,1,:] = torch.from_numpy(mod_i)
                    elif active_attack_type in ["drop_influential", "drop_random"]:
                        attacked_anchor_mask_cpu[i, target_idx_sample] = 0.0
                        logging.debug(f"Dropped anchor {target_idx_sample} for sample {i} (Burst ID: {batch['burst_id'][i].item() if isinstance(batch['burst_id'], torch.Tensor) else batch['burst_id'][i]})")
                
                input_for_main_model_attacked = {'cir_data': attacked_cir_data_cpu.to(device), 'anchor_positions': batch['anchor_positions'].to(device), 'anchor_mask': attacked_anchor_mask_cpu.to(device), 'tdoa': batch['tdoa'].to(device)}
                
                # Measure inference time for the main model
                torch.cuda.synchronize(device=device) if device.type == 'cuda' else None # Ensure previous GPU ops are done
                start_inf_time = time.perf_counter()
                predictions_attacked, _ = model(input_for_main_model_attacked)
                torch.cuda.synchronize(device=device) if device.type == 'cuda' else None # Ensure model forward pass is done
                end_inf_time = time.perf_counter()
                
                batch_inference_time_ms = (end_inf_time - start_inf_time) * 1000
                total_inference_time_ms += batch_inference_time_ms
                total_samples_processed_for_inf_time += predictions_attacked.size(0) # current_batch_size
                
                all_attacked_predictions_list.append(predictions_attacked.cpu())
                all_attacked_burst_ids_list.append(batch['burst_id'])
                if has_ground_truth and 'position' in batch: all_attacked_targets_list.append(batch['position'].cpu())

            logging.info(f"{active_attack_type.replace('_', ' ').capitalize()} attack simulation complete. Formatting results...")
            avg_inference_time_per_sample_ms = (total_inference_time_ms / total_samples_processed_for_inf_time) if total_samples_processed_for_inf_time > 0 else float('nan')
            logging.info(f"Average inference time per sample (attacked data): {avg_inference_time_per_sample_ms:.3f} ms")

            if all_attacked_predictions_list:
                preds_cat = torch.cat(all_attacked_predictions_list, dim=0).numpy()
                bursts_cat = torch.cat([b.cpu() for b in all_attacked_burst_ids_list if isinstance(b, torch.Tensor)], dim=0).numpy() if isinstance(all_attacked_burst_ids_list[0], torch.Tensor) else np.concatenate([np.array(b) for b in all_attacked_burst_ids_list])
                results_df_attacked = pd.DataFrame({'burst_id': bursts_cat.astype(int), 'pred_x': preds_cat[:,0], 'pred_y': preds_cat[:,1]})
                attack_params_suffix = f"_sigma{args.noise_sigma}" if active_attack_type == "noise" else ""
                fname_attacked = f"predictions_{args.test_set}_{active_attack_type.replace('_','-')}_{attack_strategy_for_selection.replace('_','-')}{attack_params_suffix}.csv"
                results_df_attacked.to_csv(output_dir / fname_attacked, index=False)
                logging.info(f"Attacked ({active_attack_type}) predictions saved to: {output_dir / fname_attacked}")
                if has_ground_truth and all_attacked_targets_list:
                    targets_cat = torch.cat(all_attacked_targets_list, dim=0)
                    if preds_cat.shape[0] == targets_cat.shape[0]:
                        metrics_attacked = calculate_metrics(preds_cat, targets_cat)
                        metrics_attacked['avg_inference_time_ms'] = avg_inference_time_per_sample_ms
                        logging.info(f"--- Metrics under {active_attack_type.replace('_', ' ').upper()} ({args.test_set}, Strategy: {attack_strategy_for_selection.replace('_','-')}{attack_params_suffix}) ---")
                        for k, v in metrics_attacked.items(): logging.info(f"  {k.capitalize().replace('_', ' ')}: {v:.3f}" if isinstance(v,float) else f"  {k.capitalize().replace('_', ' ')}: {v}")
                        metric_fname_attacked = f"metrics_{args.test_set}_{active_attack_type.replace('_','-')}_{attack_strategy_for_selection.replace('_','-')}{attack_params_suffix}.json"
                        with open(output_dir / metric_fname_attacked, 'w') as f: json.dump(metrics_attacked, f, indent=4)
                        logging.info(f"Attacked ({active_attack_type}) metrics saved to: {output_dir / metric_fname_attacked}")
                    else: logging.error(f"Attacked prediction count ({preds_cat.shape[0]}) != target count ({targets_cat.shape[0]}). Metrics not calculated.")
                elif has_ground_truth: logging.warning("GT available but no attacked targets collected.")
            else: logging.warning(f"No {active_attack_type} predictions were generated.")
    else: # NORMAL TESTING (NO ATTACK)
        logging.info(f"Running NORMAL inference on {args.test_set} dataset...")
        all_predictions, all_burst_ids, all_targets = [], [], []
        total_inference_time_ms_normal = 0.0 # Separate for normal run
        total_samples_processed_normal = 0

        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Testing on {args.test_set}")
            for batch in pbar:
                current_batch_size_normal = batch['cir_data'].size(0)
                input_data = {'cir_data': batch['cir_data'].to(device), 'anchor_positions': batch['anchor_positions'].to(device), 'anchor_mask': batch['anchor_mask'].to(device), 'tdoa': batch['tdoa'].to(device)}
                
                torch.cuda.synchronize(device=device) if device.type == 'cuda' else None
                start_inf_time_normal = time.perf_counter()
                predictions, _ = model(input_data)
                torch.cuda.synchronize(device=device) if device.type == 'cuda' else None
                end_inf_time_normal = time.perf_counter()

                batch_inference_time_ms_normal = (end_inf_time_normal - start_inf_time_normal) * 1000
                total_inference_time_ms_normal += batch_inference_time_ms_normal
                total_samples_processed_normal += predictions.size(0) # current_batch_size_normal

                all_predictions.append(predictions.cpu())
                all_burst_ids.append(batch['burst_id'])
                if has_ground_truth and 'position' in batch: all_targets.append(batch['position'].cpu())
        
        avg_inference_time_per_sample_ms_normal = (total_inference_time_ms_normal / total_samples_processed_normal) if total_samples_processed_normal > 0 else float('nan')
        logging.info(f"Average inference time per sample (normal data): {avg_inference_time_per_sample_ms_normal:.3f} ms")
        
        logging.info("Normal inference complete. Formatting and saving results...")
        if all_predictions:
            preds_cat = torch.cat(all_predictions, dim=0).numpy()
            bursts_cat = torch.cat([b.cpu() for b in all_burst_ids if isinstance(b, torch.Tensor)], dim=0).numpy() if isinstance(all_burst_ids[0], torch.Tensor) else np.concatenate([np.array(b) for b in all_burst_ids])
            results_df_normal = pd.DataFrame({'burst_id': bursts_cat.astype(int), 'pred_x': preds_cat[:,0], 'pred_y': preds_cat[:,1]})
            fname_normal = f"predictions_{args.test_set}_normal.csv"
            results_df_normal.to_csv(output_dir / fname_normal, index=False)
            logging.info(f"Normal predictions saved to: {output_dir / fname_normal}")
            if has_ground_truth and all_targets:
                targets_cat = torch.cat(all_targets, dim=0)
                if preds_cat.shape[0] == targets_cat.shape[0]:
                    metrics_normal = calculate_metrics(preds_cat, targets_cat)
                    metrics_normal['avg_inference_time_ms'] = avg_inference_time_per_sample_ms_normal
                    logging.info(f"--- Performance Metrics (Normal - {args.test_set}) ---")
                    for k, v in metrics_normal.items(): logging.info(f"  {k.capitalize().replace('_', ' ')}: {v:.3f}" if isinstance(v,float) else f"  {k.capitalize().replace('_', ' ')}: {v}")
                    metric_fname_normal = f"metrics_{args.test_set}_normal.json"
                    with open(output_dir / metric_fname_normal, 'w') as f: json.dump(metrics_normal, f, indent=4)
                    logging.info(f"Normal metrics saved to: {output_dir / metric_fname_normal}")
                else: logging.error(f"Normal prediction count ({preds_cat.shape[0]}) != target count ({targets_cat.shape[0]}). Metrics not calculated.")
            elif has_ground_truth: logging.warning("GT available but no normal targets collected.")
        else: logging.warning("No normal predictions were generated.")

    logging.info(f"Testing process finished in {time.time() - script_start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Localization Model and Simulate Various Attacks")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the main model checkpoint (.pth.tar file)')
    parser.add_argument('--test-set', type=str, required=True, choices=['score1', 'score2'], help='Which test set to evaluate')
    parser.add_argument('--data-dir', type=str, default='./dataset', help='Directory containing dataset files')
    parser.add_argument('--output-dir', type=str, default='./test_results_final', help='Directory to save prediction & metrics files')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use (-1 for CPU)')
    
    parser.add_argument('--spoofing-attack', action='store_true', help='Enable simulated spoofing attack on one anchor.')
    parser.add_argument('--noise-attack', action='store_true', help='Enable simulated Gaussian noise attack on one anchor’s CIR.')
    parser.add_argument('--drop-influential-anchor-attack', action='store_true', help='Enable attack by dropping the most influential anchor.')
    parser.add_argument('--drop-random-anchor-attack', action='store_true', help='Enable attack by dropping a random active anchor.')
    
    parser.add_argument('--noise-sigma', type=float, default=0.5, help='Std deviation (sigma) of Gaussian noise for noise-attack (default: 0.5).')
    
    parser.add_argument('--guiding-attention-checkpoint', type=str, default=None,
                        help='Optional: Path to an ATTENTION-BASED model checkpoint to guide anchor selection for certain attacks '
                             'if the main checkpoint model does NOT use attention.')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logging.info("----- Test Script Configuration -----")
    for arg, value in vars(args).items(): logging.info(f"{arg}: {value}")
    if args.spoofing_attack: logging.info(f"Spoofing Params: Distance Closer={SPOOFING_DISTANCE_CLOSER_M}m, Amplitude Scale={SPOOFING_AMPLITUDE_SCALE}")
    if args.noise_attack: logging.info(f"Noise Attack Param: Sigma={args.noise_sigma}")
    logging.info(f"CSI Sampling Rate (for spoofing time shift): {CSI_SAMPLING_RATE_HZ} Hz")
    logging.info("-----------------------------------")
    main(args)
