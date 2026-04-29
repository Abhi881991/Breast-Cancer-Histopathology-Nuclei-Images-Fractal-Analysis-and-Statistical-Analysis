import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression
from scipy import stats
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# ==============================
# PARAMETERS
# ==============================
INPUT_CSV = r"F:\BRACS-TIFF\Analysis Files\Segmented Images\UDH-Binary\all_centroids.csv"
OUTPUT_PREFIX = "Correlation_Dimension_Validation_Wilcoxon"

N_SIM = 20
MIN_POINTS = 20
R2_THRESHOLD = 0.90

# ==============================
# FRACTAL CALCULATION CORE
# ==============================

def compute_correlation_dimension(points, r_min, r_max):
    """Computes Dc and R2 using a sliding window for max fit."""
    N = len(points)
    if N < MIN_POINTS:
        return np.nan, np.nan

    dists = pdist(points)
    r_values = np.geomspace(r_min, r_max, 15)
    c_r = np.array([2 * np.sum(dists < r) / (N * (N - 1)) for r in r_values])

    valid = c_r > 0
    if np.sum(valid) < 5:
        return np.nan, np.nan

    log_r = np.log10(r_values[valid])
    log_c = np.log10(c_r[valid])

    best_r2 = -1
    best_dc = np.nan

    window = min(8, len(log_r))
    if window < 4:
        return np.nan, np.nan

    for i in range(len(log_r) - window + 1):
        x = log_r[i:i+window].reshape(-1, 1)
        y = log_c[i:i+window]
        reg = LinearRegression().fit(x, y)
        r2 = reg.score(x, y)
        slope = reg.coef_[0]  # FIX: extract scalar, not array
        if 0 < slope <= 2.1 and r2 > best_r2:
            best_r2 = r2
            best_dc = slope

    return best_dc, best_r2

# ==============================
# NULL MODEL GENERATORS
# ==============================

def generate_csr(n_points, x_lim, y_lim):
    """Complete Spatial Randomness (Uniform)"""
    return np.random.uniform(0, [x_lim, y_lim], size=(n_points, 2))

def generate_clustered(n_points, x_lim, y_lim):
    """Generic Clustering (Thomas Process-like)"""
    n_parents = max(5, n_points // 20)  # FIX: scale with data, not hardcoded
    parents = np.random.uniform(0, [x_lim, y_lim], size=(n_parents, 2))
    sigma = 0.10 * min(x_lim, y_lim)   # FIX: wider spread for realistic clustering
    points = []
    for _ in range(n_points):
        p = parents[np.random.randint(0, n_parents)]
        pt = np.clip(p + np.random.normal(0, sigma, size=2), [0, 0], [x_lim, y_lim])  # FIX: keep in domain
        points.append(pt)
    return np.array(points)

# ==============================
# MAIN ANALYSIS ENGINE
# ==============================

def analyze_fractal_validation(csv_path, n_sim=20):
    df_raw = pd.read_csv(csv_path)

    image_col = 'Image'
    x_col, y_col = 'X', 'Y'

    images = df_raw[image_col].unique()
    results = []

    for img_id in tqdm(images, desc="Validating Scale-Free Behavior"):
        img_data = df_raw[df_raw[image_col] == img_id]
        points = img_data[[x_col, y_col]].values
        n_points = len(points)

        if n_points < MIN_POINTS:
            continue

        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()

        dists = pdist(points)
        r_min = np.percentile(dists, 5) if len(dists) > 0 else 1
        r_max = min(x_range, y_range) * 0.25

        if r_min <= 0 or r_min >= r_max:
            continue

        dc_real, r2_real = compute_correlation_dimension(points, r_min, r_max)

        # FIX: enforce R² quality gate before accepting result
        if np.isnan(dc_real) or r2_real < R2_THRESHOLD:
            continue

        dc_csr_list = []
        dc_clustered_list = []

        for _ in range(n_sim):
            csr_pts = generate_csr(n_points, x_range, y_range)
            c_pts = generate_clustered(n_points, x_range, y_range)

            d_csr, _ = compute_correlation_dimension(csr_pts, r_min, r_max)
            d_clu, _ = compute_correlation_dimension(c_pts, r_min, r_max)

            if not np.isnan(d_csr): dc_csr_list.append(d_csr)
            if not np.isnan(d_clu): dc_clustered_list.append(d_clu)

        results.append({
            'image_id': img_id,
            'Dc_real': dc_real,
            'R2_real': r2_real,
            'Dc_csr_mean': np.mean(dc_csr_list) if dc_csr_list else np.nan,
            'Dc_clustered_mean': np.mean(dc_clustered_list) if dc_clustered_list else np.nan
        })

    return pd.DataFrame(results).dropna()

def scientific_validation(df, output_prefix):
    print("\n" + "="*60)
    print("SCIENTIFIC VALIDATION REPORT")
    print("="*60)

    required_cols = ['Dc_real', 'Dc_csr_mean', 'Dc_clustered_mean']
    plot_data = df.dropna(subset=required_cols)

    if len(plot_data) == 0:
        print("Error: No valid data points found. Check if analysis successful.")
        return

    real_vals = plot_data['Dc_real'].values.flatten()
    csr_vals = plot_data['Dc_csr_mean'].values.flatten()
    clu_vals = plot_data['Dc_clustered_mean'].values.flatten()

    # FIX: paired Wilcoxon is correct (nulls simulated per-image); add length guard
    for label, null_vals in [('CSR', csr_vals), ('Clustered', clu_vals)]:
        try:
            if len(real_vals) != len(null_vals):
                print(f"\nReal vs {label}: Skipped — length mismatch ({len(real_vals)} vs {len(null_vals)})")
                continue
            stat, p_val = stats.wilcoxon(real_vals, null_vals)
            print(f"\nReal vs {label}:")
            print(f"  Real Median:   {np.median(real_vals):.4f}")
            print(f"  {label} Median: {np.median(null_vals):.4f}")
            print(f"  Wilcoxon p-value: {p_val:.4e}")
        except ValueError as e:
            print(f"\nReal vs {label}: Statistical test failed ({e})")

    plt.figure(figsize=(10, 6))
    plt.hist(real_vals, bins=25, alpha=0.5, label='Real Pathology', color='blue', edgecolor='black')
    plt.hist(csr_vals, bins=25, alpha=0.4, label='CSR (Random)', color='green', edgecolor='black')
    plt.hist(clu_vals, bins=25, alpha=0.3, label='Clustered', color='red', edgecolor='black')
    plt.axvline(np.median(real_vals), color='blue', linestyle='dashed', linewidth=2, label='Real Median')
    plt.axvline(np.median(csr_vals), color='green', linestyle='dashed', linewidth=2, label='CSR Median')
    plt.title('Nuclear Spatial Organization vs. Null Models', fontweight='bold')
    plt.xlabel('Correlation Dimension (Dc)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)

    plot_path = f"{output_prefix}_distribution_final.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved plot: {plot_path}")

if __name__ == "__main__":
    if os.path.exists(INPUT_CSV):
        final_df = analyze_fractal_validation(INPUT_CSV, n_sim=N_SIM)
        final_df.to_csv(f"{OUTPUT_PREFIX}_results.csv", index=False)
        scientific_validation(final_df, OUTPUT_PREFIX)
    else:
        print(f"Error: File {INPUT_CSV} not found.")