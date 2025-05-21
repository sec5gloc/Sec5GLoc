import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from sklearn.cluster import KMeans
import logging # Added for potential error logging
from pathlib import Path

# Configure logging for this script if desired
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- Global Style Settings for Figures ----------
def set_paper_style(ax):
    ax.tick_params(labelsize=9)
    ax.grid(True, linewidth=0.3)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(9)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontsize(9)

FIG_SIZE = (2.5, 2)
DPI = 600

# ---------- Load Predictions ----------
pred_path = "knn_test_results/knn_test_predictions_NLOS.csv"
logging.info(f"Loading predictions from: {pred_path}")
try:
    pred_df = pd.read_csv(pred_path)
except FileNotFoundError:
    logging.error(f"Prediction file not found: {pred_path}. Exiting.")
    exit()
except Exception as e:
    logging.error(f"Error loading prediction file {pred_path}: {e}. Exiting.")
    exit()


# ---------- Load and Parse Ground Truth using Pandas ----------
gt_path = "dataset/test_NLOS.csv"
logging.info(f"Loading ground truth from: {gt_path}")
burst_gt_map = {}

try:
    # Use pandas to read the CSV, it handles quoting correctly
    df_gt_raw = pd.read_csv(gt_path, sep=';')
    
    if 'json' not in df_gt_raw.columns:
        logging.error(f"'json' column not found in ground truth file: {gt_path}. Exiting.")
        exit()

    for index, row in df_gt_raw.iterrows():
        json_string = row['json']
        try:
            data = json.loads(json_string)
            
            # Ensure all required keys are present
            if not all(k in data for k in ["burst_id", "ref_x", "ref_y"]):
                logging.warning(f"Skipping row {index+1} in {gt_path} due to missing required keys in JSON: {json_string[:100]}...")
                continue

            burst_id = int(data["burst_id"])
            ref_x = float(data["ref_x"])
            ref_y = float(data["ref_y"])
            
            if burst_id not in burst_gt_map: # Store only the first occurrence for each burst_id
                burst_gt_map[burst_id] = (ref_x, ref_y)
        except json.JSONDecodeError as e:
            logging.warning(f"Skipping row {index+1} in {gt_path} due to JSON decode error: {e}. Content: {json_string[:100]}...")
            continue
        except KeyError as e:
            logging.warning(f"Skipping row {index+1} in {gt_path} due to missing key {e} in JSON. Content: {json_string[:100]}...")
            continue
        except ValueError as e: # For int() or float() conversion errors
            logging.warning(f"Skipping row {index+1} in {gt_path} due to value conversion error: {e}. Content: {json_string[:100]}...")
            continue
        except Exception as e: # Catch any other unexpected errors for a row
            logging.warning(f"Skipping row {index+1} in {gt_path} due to an unexpected error: {e}. Content: {json_string[:100]}...")
            continue
            
except FileNotFoundError:
    logging.error(f"Ground truth file not found: {gt_path}. Exiting.")
    exit()
except Exception as e:
    logging.error(f"Error loading ground truth file {gt_path}: {e}. Exiting.")
    exit()

if not burst_gt_map:
    logging.error(f"No ground truth data successfully loaded from {gt_path}. Cannot proceed with matching. Exiting.")
    exit()
logging.info(f"Successfully loaded and parsed ground truth for {len(burst_gt_map)} unique burst_ids.")

# ---------- Match Predictions to Ground Truth ----------
logging.info("Matching predictions to ground truth...")
matched_data = []
for index, row in pred_df.iterrows():
    try:
        burst_id = int(row['burst_id'])
        if burst_id in burst_gt_map:
            ref_x, ref_y = burst_gt_map[burst_id]
            matched_data.append({
                "burst_id": burst_id,
                "pred_x": row["pred_x"],
                "pred_y": row["pred_y"],
                "gt_x": ref_x,
                "gt_y": ref_y
            })
        else:
            logging.debug(f"Burst ID {burst_id} from predictions not found in ground truth map.")
    except ValueError:
        logging.warning(f"Could not convert burst_id '{row['burst_id']}' to int in prediction row {index}. Skipping.")
        continue


if not matched_data:
    logging.error("No predictions could be matched with ground truth. Ensure burst_ids align and data exists. Exiting.")
    exit()

df = pd.DataFrame(matched_data)
df["error"] = np.sqrt((df["pred_x"] - df["gt_x"])**2 + (df["pred_y"] - df["gt_y"])**2)
logging.info(f"Successfully matched {len(df)} predictions with ground truth.")

# ---------- Create Output Directory for Figures ----------
figure_output_dir = Path("figures") # Define an output directory for figures
figure_output_dir.mkdir(parents=True, exist_ok=True)
logging.info(f"Saving figures to: {figure_output_dir}")


# ---------- Plot 1: CDF ----------
fig1_path = figure_output_dir / "figure1_cdf.pdf"
fig, ax = plt.subplots(figsize=FIG_SIZE)
sorted_error = np.sort(df["error"])
if len(sorted_error) == 0:
    logging.warning("No error data to plot CDF.")
else:
    cdf = np.arange(1, len(sorted_error) + 1) / len(sorted_error)
    ax.plot(sorted_error, cdf, label="CDF", linewidth=2, color="springgreen")
    p75 = np.percentile(sorted_error, 75)
    p90 = np.percentile(sorted_error, 90)
    ax.axvline(p75, linestyle='--', color='c', linewidth=2, label=f'P75 = {p75:.2f} m')
    ax.axvline(p90, linestyle='--', color='violet', linewidth=2, label=f'P90 = {p90:.2f} m')
    ax.set_xlabel("Localization Error (m)")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=6, loc='lower right', frameon=True, fancybox=False, framealpha=1)
    set_paper_style(ax)
    plt.tight_layout(pad=0.1)
    fig.savefig(fig1_path, dpi=DPI, bbox_inches='tight')
    logging.info(f"Saved CDF plot to {fig1_path}")
plt.close(fig)

# ---------- Plot 2: Scatter ----------
fig2_path = figure_output_dir / "figure2_scatter.pdf"
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.scatter(df["gt_x"], df["gt_y"], label="Ground Truth", alpha=0.6, color="blue", s=10)
sc = ax.scatter(df["pred_x"], df["pred_y"], c=df["error"], cmap="plasma", label="Prediction", alpha=0.7, s=10) #cmap="viridis"
cbar = plt.colorbar(sc, ax=ax)
cbar.ax.tick_params(labelsize=9)
cbar.set_label("Localization Error (m)", fontsize=9)
ax.set_xlabel("X coordinate (m)")
ax.set_ylabel("Y coordinate (m)")
ax.legend(fontsize=9, loc='upper right', frameon=True, fancybox=False, framealpha=0.5)
set_paper_style(ax)
plt.tight_layout(pad=0.1)
fig.savefig(fig2_path, dpi=DPI, bbox_inches='tight')
logging.info(f"Saved Scatter plot to {fig2_path}")
plt.close(fig)

# ---------- Plot 3: Heatmap ----------
fig3_path = figure_output_dir / "figure3_location_error_heatmap.pdf"
fig, ax = plt.subplots(figsize=FIG_SIZE)
if not df.empty:
    heatmap_data, xedges, yedges = np.histogram2d(df["gt_x"], df["gt_y"], bins=50, weights=df["error"])
    counts, _, _ = np.histogram2d(df["gt_x"], df["gt_y"], bins=[xedges, yedges])
    average_error = np.divide(heatmap_data, counts, out=np.zeros_like(heatmap_data), where=counts != 0)
    img = ax.imshow(average_error.T, origin='lower', aspect='auto',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="inferno")
    cbar = plt.colorbar(img, ax=ax)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Average Localization Error (m)", fontsize=9)
    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")
    set_paper_style(ax)
    plt.tight_layout(pad=0.1)
    fig.savefig(fig3_path, dpi=DPI, bbox_inches='tight')
    logging.info(f"Saved Heatmap plot to {fig3_path}")
else:
    logging.warning("DataFrame for heatmap is empty. Skipping heatmap plot.")
plt.close(fig)

logging.info("All plots processed and attempted to save them!")
