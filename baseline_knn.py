import argparse
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from tqdm import tqdm
import json
from scipy.stats import entropy as scipy_entropy
from joblib import Parallel, delayed
import time
from dataset import load_anchors, load_and_preprocess_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_features(df: pd.DataFrame, n_anchor_system: int = 8, n_jobs: int = -1, use_parallel: bool = True) -> pd.DataFrame:
    """
    Extracts features from the preprocessed DataFrame.
    Each row in the output DataFrame corresponds to one burst_id,
    with features from all anchors pivoted into columns.
    """
    logging.info("Starting feature extraction...")
    
    # Log initial anchor stats per burst_id from the input df
    # This df contains one row per anchor per burst
    initial_anchor_counts = df.groupby('burst_id')['anch_id'].nunique()
    logging.info(f"Input data: Total unique burst_ids: {initial_anchor_counts.shape[0]}")
    logging.info(f"Input data: Anchor counts per burst_id: min={initial_anchor_counts.min()}, max={initial_anchor_counts.max()}, mean={initial_anchor_counts.mean():.2f}")

    # Group by both burst_id and anch_id to process each anchor's data for a burst
    # This assumes that for a given (burst_id, anch_id) pair, all relevant data (csi_real, csi_imag) is present in any row of that group.
    grouped_for_features = df.groupby(['burst_id', 'anch_id'])

    # Define the number of features extracted per anchor (peak, energy, delay, std, mean_phase, std_phase, phase_slope, entropy).
    num_features_per_anchor = 8 

    def compute_features_for_anchor_burst(key_group_tuple):
        key, group = key_group_tuple
        burst_id, anch_id = key
        
        # Take the first row for this anchor in this burst.
        row = group.iloc[0] 
        csi_real = np.array(row['csi_real'])
        csi_imag = np.array(row['csi_imag'])
        cir = csi_real + 1j * csi_imag

        mag = np.abs(cir)
        phase = np.angle(cir)

        # Handle all-zero CIR (e.g., from a truly missing/silent anchor)
        if np.all(mag < 1e-9): # If magnitude is effectively zero
            peak = 0.0
            energy = 0.0
            delay = 0.0 # Or some other placeholder like -1 or np.nan if preferred before fillna
            std = 0.0
            mean_phase = 0.0
            std_phase = 0.0
            phase_slope = 0.0
            mag_entropy = 0.0 # Or a high entropy value
        else:
            energy = np.sum(mag ** 2)
            peak = np.max(mag)
            # Weighted average delay (mean delay)
            delay = np.sum(np.arange(len(mag)) * mag) / (np.sum(mag) + 1e-8) 
            std = np.std(mag)
            mean_phase = np.mean(phase)
            std_phase = np.std(phase)
            unwrapped_phase = np.unwrap(phase)
            # Fit line to unwrapped phase to get slope; handle short/flat phases
            if len(unwrapped_phase) > 1:
                phase_slope = np.polyfit(np.arange(len(unwrapped_phase)), unwrapped_phase, 1)[0]
            else:
                phase_slope = 0.0 
            # Calculate magnitude entropy
            hist_mag, _ = np.histogram(mag, bins=min(32, len(np.unique(mag)) if len(np.unique(mag)) > 1 else 2), density=True)
            mag_entropy = scipy_entropy(hist_mag + 1e-9) # Add small epsilon for stability

        # Features are NOT prefixed with anch_id here; pivot will handle that.
        return {
            'burst_id': burst_id,
            'anchor_id': anch_id, 
            'peak': peak,
            'energy': energy,
            'delay': delay,
            'std_dev_mag': std, 
            'mean_phase': mean_phase,
            'std_dev_phase': std_phase, 
            'phase_slope': phase_slope,
            'mag_entropy': mag_entropy,
        }

    feature_list_of_dicts = []
    if use_parallel:
        logging.info(f"Extracting features in parallel with n_jobs={n_jobs}...")
        # Convert groupby object to list of (key, group) tuples for Parallel
        group_items = list(grouped_for_features)
        feature_list_of_dicts = Parallel(n_jobs=n_jobs)(
            delayed(compute_features_for_anchor_burst)(item) for item in tqdm(group_items, desc="Extracting features")
        )
    else:
        logging.info("Extracting features serially...")
        for key, group in tqdm(grouped_for_features, desc="Extracting features (serial)"):
            feature_list_of_dicts.append(compute_features_for_anchor_burst((key, group)))

    if not feature_list_of_dicts:
        logging.error("No features were extracted. Check input data and grouping.")
        # Return an empty DataFrame with expected columns if possible, or raise error.
        return pd.DataFrame()

    features_per_anchor_df = pd.DataFrame(feature_list_of_dicts)

    # Pivot the table to have one row per burst_id and features from each anchor as columns.
    # List all feature columns (excluding 'burst_id', 'anchor_id').
    value_columns = ['peak', 'energy', 'delay', 'std_dev_mag', 'mean_phase', 'std_dev_phase', 'phase_slope', 'mag_entropy']
    
    try:
        pivoted_features_df = features_per_anchor_df.pivot(
            index='burst_id', 
            columns='anchor_id', 
            values=value_columns
        )
    except Exception as e:
        logging.error(f"Error during pivoting: {e}. This might happen if 'features_per_anchor_df' is empty or has unexpected structure.")
        logging.error(f"Shape of features_per_anchor_df: {features_per_anchor_df.shape}")
        logging.error(f"Head of features_per_anchor_df:\n{features_per_anchor_df.head()}")
        return pd.DataFrame() # Return empty to prevent downstream errors

    # Flatten the multi-level column index (e.g., ('peak', 'A1') -> 'peak_A1')
    pivoted_features_df.columns = ['_'.join(col).strip() for col in pivoted_features_df.columns.values]
    pivoted_features_df.reset_index(inplace=True) # Make 'burst_id' a regular column
    
    # Check for columns that are all NaN (indicates an anchor never reported or always had issues)
    all_nan_cols = pivoted_features_df.drop(columns=['burst_id']).columns[pivoted_features_df.drop(columns=['burst_id']).isnull().all()]
    if len(all_nan_cols) > 0:
        logging.warning(f"The following feature columns are entirely NaN, possibly indicating anchors that never reported: {list(all_nan_cols)}")

    # Fill NaN values (e.g., for missing anchors or features that couldn't be computed)
    # Using 0 is a common strategy, but consider if mean/median might be better for some features.
    pivoted_features_df.fillna(0, inplace=True)
    logging.info("Filled missing feature values (NaNs from pivot or computation) with 0.")
    logging.info(f"Shape of final pivoted features DataFrame: {pivoted_features_df.shape}")
    
    return pivoted_features_df

def align_labels(features_df: pd.DataFrame, original_df_with_labels: pd.DataFrame) -> pd.DataFrame:
    """Merges features with ground truth labels (ref_x, ref_y)."""
    if 'burst_id' not in features_df.columns:
        logging.error("'burst_id' not found in features_df. Cannot align labels.")
        return pd.DataFrame() # Or raise error
        
    # Extract unique positions per burst_id from the original preprocessed data
    # This original_df_with_labels should be the output of load_and_preprocess_data
    if 'ref_x' not in original_df_with_labels.columns or 'ref_y' not in original_df_with_labels.columns:
        logging.warning("Ground truth columns 'ref_x' or 'ref_y' not found in original_df_with_labels. Returning features without labels.")
        return features_df

    # Take the first ref_x, ref_y for each burst_id (should be consistent within a burst)
    ground_truth_positions = original_df_with_labels.groupby('burst_id')[['ref_x', 'ref_y']].first().reset_index()
    
    aligned_df = features_df.merge(ground_truth_positions, on='burst_id', how='inner')
    logging.info(f"Shape after aligning labels: {aligned_df.shape}. Dropped {len(features_df) - len(aligned_df)} feature rows due to no matching labels.")
    return aligned_df


def train_knn(train_features_with_labels_df: pd.DataFrame, model_output_path: Path, args_config):
    """Trains a k-NN regressor model with hyperparameter tuning."""
    logging.info(f"Starting k-NN training. Input data shape: {train_features_with_labels_df.shape}")

    if 'ref_x' not in train_features_with_labels_df.columns or 'ref_y' not in train_features_with_labels_df.columns:
        raise ValueError("Training data is missing 'ref_x' or 'ref_y' label columns.")
    if 'burst_id' not in train_features_with_labels_df.columns:
        raise ValueError("Training data is missing 'burst_id' column.")

    # Drop burst_id and label columns to get feature matrix X
    X = train_features_with_labels_df.drop(columns=['burst_id', 'ref_x', 'ref_y']).values
    y = train_features_with_labels_df[['ref_x', 'ref_y']].values

    if X.shape[0] == 0:
        raise ValueError("No training samples available (X is empty). Check data processing and label alignment.")
    if X.shape[1] == 0:
        raise ValueError("No features available for training (X has 0 columns). Check feature extraction.")


    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Standardize features
        ('knn', KNeighborsRegressor())
    ])

    param_grid = {
        'knn__n_neighbors': args_config.k_values,
        'knn__weights': args_config.weights,
        'knn__metric': args_config.metrics
    }

    # Inner CV for GridSearchCV to select hyperparameters
    cv_for_gridsearch = KFold(n_splits=args_config.cv_gridsearch, shuffle=True, random_state=args_config.seed)
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv_for_gridsearch, 
        scoring='neg_mean_squared_error', # Lower (more negative) is worse, so GridSearchCV maximizes this
        n_jobs=args_config.n_jobs if args_config.use_parallel_grid else 1, 
        verbose=2,
        refit=True # Refits the best estimator on the whole training data
    )

    logging.info("Starting GridSearchCV to find best k-NN parameters...")
    grid_search.fit(X, y)

    best_model_found = grid_search.best_estimator_
    joblib.dump(best_model_found, model_output_path)

    best_params_path = model_output_path.with_name(model_output_path.stem + '.best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump(grid_search.best_params_, f, indent=2)

    logging.info(f"Best k-NN parameters found: {grid_search.best_params_}")
    # best_score_ is negative MSE. Convert to RMSE for interpretability.
    best_rmse_from_grid = np.sqrt(-grid_search.best_score_) 
    logging.info(f"Best GridSearchCV CV score (RMSE, approx): {best_rmse_from_grid:.4f} meters")
    
    # Perform a separate (outer) cross-validation using the BEST estimator to get detailed performance metrics
    logging.info("Performing outer cross-validation with the best estimator for detailed metrics...")
    cv_outer_for_metrics = KFold(n_splits=args_config.cv_metrics_folds, shuffle=True, random_state=args_config.seed)
    y_cv_pred = cross_val_predict(
        best_model_found, X, y, 
        cv=cv_outer_for_metrics, 
        n_jobs=args_config.n_jobs if args_config.use_parallel_grid else 1 # Can reuse n_jobs for this
    )
    
    cv_dists = np.linalg.norm(y_cv_pred - y, axis=1)
    
    outer_cv_metrics = {
        'mean_error': float(np.mean(cv_dists)),
        'median_error': float(np.median(cv_dists)),
        'p75_error': float(np.quantile(cv_dists, 0.75)),
        'p90_error': float(np.quantile(cv_dists, 0.90)),
        'sample_count': float(len(cv_dists)),
        'best_rmse_from_gridsearch_cv': float(best_rmse_from_grid) # Include the grid search RMSE
    }
    
    outer_cv_metrics_path = model_output_path.with_name(model_output_path.stem + '.outer_cv_metrics.json')
    with open(outer_cv_metrics_path, 'w') as f:
        json.dump(outer_cv_metrics, f, indent=2)
    logging.info(f"Outer Cross-Validation Metrics (with best estimator): {outer_cv_metrics}")

    logging.info(f"k-NN model training complete. Model saved to: {model_output_path}")
    logging.info(f"Trained with {X.shape[0]} samples and {X.shape[1]} features.")


def test_knn(test_features_with_labels_df: pd.DataFrame, model_load_path: Path, results_output_dir: Path):
    """Tests the trained k-NN model on the test set."""
    logging.info(f"Loading trained k-NN model from: {model_load_path}")
    try:
        model = joblib.load(model_load_path)
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_load_path}. Exiting test.")
        return
    except Exception as e:
        logging.error(f"Error loading model: {e}. Exiting test.")
        return

    logging.info(f"Testing k-NN model. Input test data shape: {test_features_with_labels_df.shape}")

    if 'ref_x' not in test_features_with_labels_df.columns or 'ref_y' not in test_features_with_labels_df.columns:
        logging.warning("Test data is missing 'ref_x' or 'ref_y' label columns. Will only generate predictions if possible.")
        raise ValueError("Test data for evaluation must contain 'ref_x' and 'ref_y'.")

    X_test = test_features_with_labels_df.drop(columns=['burst_id', 'ref_x', 'ref_y']).values
    y_true = test_features_with_labels_df[['ref_x', 'ref_y']].values

    if X_test.shape[0] == 0:
        logging.error("No test samples available (X_test is empty). Exiting test.")
        return

    logging.info("Predicting positions on the test set...")
    y_pred = model.predict(X_test)

    dists = np.linalg.norm(y_pred - y_true, axis=1)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'burst_id': test_features_with_labels_df['burst_id'],
        'pred_x': y_pred[:, 0],
        'pred_y': y_pred[:, 1],
        # 'ref_x': y_true[:, 0],
        # 'ref_y': y_true[:, 1],
        # 'error_m': dists
    })

    test_metrics = {
        'mean_error_m': float(np.mean(dists)),
        'median_error_m': float(np.median(dists)),
        'p75_error_m': float(np.quantile(dists, 0.75)),
        'p90_error_m': float(np.quantile(dists, 0.90)),
        'sample_count': float(len(dists))
    }

    results_output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv_path = results_output_dir / 'knn_test_predictions.csv'
    metrics_json_path = results_output_dir / 'knn_test_metrics.json'

    results_df.to_csv(predictions_csv_path, index=False)
    with open(metrics_json_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)

    logging.info(f"Test predictions saved to: {predictions_csv_path}")
    logging.info(f"Test metrics saved to: {metrics_json_path}")
    logging.info(f"Test Metrics: {test_metrics}")


def main_script_logic(args):
    """Main logic for training or testing."""
    base_data_path = Path(args.data_dir)
    
    logging.info(f"Loading anchor data from: {base_data_path / 'anchors.txt'}")
    anchors_df = load_anchors(base_data_path / 'anchors.txt')
    num_system_anchors = len(anchors_df) # Get actual number of anchors in the system

    if args.mode == 'train':
        train_csv_path = base_data_path / args.train_file
        logging.info(f"Mode: TRAIN. Loading training data from: {train_csv_path}")
        # df_train_raw is the output of load_and_preprocess_data (already processed JSON)
        df_train_raw_processed = load_and_preprocess_data(train_csv_path, anchors_df, is_training=True)
        if df_train_raw_processed is None or df_train_raw_processed.empty:
            logging.error(f"No data loaded from {train_csv_path}. Exiting.")
            return

        features_train_df = extract_features(
            df_train_raw_processed, 
            n_anchor_system=num_system_anchors, 
            n_jobs=args.n_jobs, 
            use_parallel=args.use_parallel_feature_extraction
        )
        if features_train_df.empty:
            logging.error("Feature extraction for training data resulted in an empty DataFrame. Exiting.")
            return
        
        # Align labels (ref_x, ref_y) using the already processed df_train_raw_processed
        train_data_for_model = align_labels(features_train_df, df_train_raw_processed)
        if train_data_for_model.empty:
            logging.error("Training data became empty after aligning labels. Exiting.")
            return

        model_save_path = Path(args.model_output_dir) / args.model_filename
        model_save_path.parent.mkdir(parents=True, exist_ok=True) # Ensure model output directory exists
        train_knn(train_data_for_model, model_save_path, args)

    elif args.mode == 'test':
        if not args.test_file:
            logging.error("Test mode selected, but --test-file was not provided. Exiting.")
            return
        test_csv_path = base_data_path / args.test_file
        logging.info(f"Mode: TEST. Loading test data from: {test_csv_path}")
        # df_test_raw is the output of load_and_preprocess_data
        df_test_raw_processed = load_and_preprocess_data(test_csv_path, anchors_df, is_training=True) # Assume GT is present for testing metrics
        if df_test_raw_processed is None or df_test_raw_processed.empty:
            logging.error(f"No data loaded from {test_csv_path}. Exiting.")
            return

        features_test_df = extract_features(
            df_test_raw_processed, 
            n_anchor_system=num_system_anchors,
            n_jobs=args.n_jobs, 
            use_parallel=args.use_parallel_feature_extraction
        )
        if features_test_df.empty:
            logging.error("Feature extraction for test data resulted in an empty DataFrame. Exiting.")
            return
            
        # Align labels using the already processed df_test_raw_processed
        test_data_for_model = align_labels(features_test_df, df_test_raw_processed)
        if test_data_for_model.empty:
            logging.error("Test data became empty after aligning labels. Exiting.")
            return
            
        model_load_path = Path(args.model_output_dir) / args.model_filename # Assuming model is in output_dir
        test_results_output_dir = Path(args.test_results_dir)
        test_knn(test_data_for_model, model_load_path, test_results_output_dir)
    else:
        # This case should not be reached due to argparse choices
        raise ValueError("Unsupported mode. Choose 'train' or 'test'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="k-NN Baseline for Indoor Localization")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help="Mode to run: 'train' or 'test'.")
    
    # Data paths
    parser.add_argument('--data-dir', type=str, default='./dataset', help="Directory containing anchors.txt and data CSVs.")
    parser.add_argument('--train-file', type=str, default='training_NLOS.csv', help="Filename of the training data CSV within --data-dir (used in train mode).")
    parser.add_argument('--test-file', type=str, help="Filename of the test data CSV within --data-dir (required in test mode).")
    
    # Model saving/loading and output directories
    parser.add_argument('--model-output-dir', type=str, default='./knn_model_files', help="Directory to save/load the trained model and associated files.")
    parser.add_argument('--model-filename', type=str, default='knn_model.joblib', help="Filename for the saved k-NN model.")
    parser.add_argument('--test-results-dir', type=str, default='./knn_test_results_1', help="Directory to save test predictions and metrics.")
    
    # k-NN Hyperparameters for GridSearchCV
    parser.add_argument('--k-values', type=int, nargs='+', default=[3, 5, 7, 9, 11, 15], help="List of k values for k-NN.")
    parser.add_argument('--weights', type=str, nargs='+', default=['uniform', 'distance'], help="List of weight functions for k-NN.")
    parser.add_argument('--metrics', type=str, nargs='+', default=['euclidean', 'manhattan', 'minkowski'], help="List of distance metrics for k-NN.")
    
    # Cross-validation settings
    parser.add_argument('--cv-gridsearch', type=int, default=5, help="Number of folds for inner CV in GridSearchCV.")
    parser.add_argument('--cv-metrics-folds', type=int, default=5, help="Number of folds for outer CV for reporting final metrics (using best model).")
    
    # General settings
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--n-jobs', type=int, default=8, help="Number of parallel jobs for feature extraction and GridSearchCV (-1 uses all available cores).")
    parser.add_argument('--use-parallel-feature-extraction', action='store_true', help="Enable parallel processing for feature extraction.")
    parser.add_argument('--use-parallel-grid', action='store_true', help="Enable parallel processing for GridSearchCV (if n_jobs > 1 in GridSearchCV).")

    args = parser.parse_args()

    # Configure File Logging
    # Create a unique log directory for this run, perhaps based on mode and timestamp
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_output_dir_base = Path(args.model_output_dir if args.mode == 'train' else args.test_results_dir)
    current_run_log_dir = log_output_dir_base / "logs" / f"{args.mode}_{run_timestamp}"
    current_run_log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = current_run_log_dir / "run.log"

    # Remove existing handlers and reconfigure
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler() # Keep console output
        ]
    )

    logging.info(f"Script started in mode: {args.mode}")
    logging.info("Running with arguments:")
    for arg_name, value in vars(args).items():
        logging.info(f"  {arg_name}: {value}")
    
    main_script_logic(args)
    logging.info(f"Script finished mode: {args.mode}")
