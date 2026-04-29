import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.draw import ellipse as draw_ellipse
from skimage.morphology import erosion, disk

warnings.filterwarnings('ignore')

# ==============================
# PARAMETERS & PATHS
# ==============================
INPUT_DIR     = r"F:\BRACS-TIFF\Analysis Files\Segmented Images\UDH-Binary"
OUTPUT_PREFIX = "Minkowski_Validation"

R_MIN        = 1       # Minimum dilation radius (pixels)
R_MAX_FRAC   = 0.40    # R_MAX = R_MAX_FRAC × median minor axis per image
R_MAX_FLOOR  = 8       # Minimum R_MAX (pixels)
WINDOW_SIZE  = 4       # Minimum window for valid slope estimate
R2_THRESHOLD = 0.85    # Minimum local R² to accept a Dm estimate
N_SIM        = 5
SIGMA_AXIS   = 0.10    # Fractional std of semi-axis perturbation (10%)
SIGMA_ANGLE  = 0.05    # Std of orientation perturbation in radians

_DISK1 = disk(1)

# ==============================
# ANALYTICAL OLS
# ==============================

def _ols(x, y):
    """Closed-form OLS slope and R²."""
    n = x.shape[0]
    sx = x.sum()
    sy = y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()
    syy = (y * y).sum()

    denom = n * sxx - sx * sx
    if denom == 0.0:
        return np.nan, 0.0

    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n

    ss_res = ((y - (slope * x + intercept)) ** 2).sum()
    ss_tot = syy - (sy * sy) / n
    if ss_tot <= 0.0:
        return slope, 0.0

    r2 = 1.0 - ss_res / ss_tot
    return slope, r2


def _best_dm_r2(lr, log_A, window_size):
    best_r2 = -1.0
    best_dm = np.nan
    n = lr.shape[0]

    if n < window_size:
        return best_dm, best_r2

    for i in range(n - window_size + 1):
        x = lr[i:i+window_size]
        y = log_A[i:i+window_size]
        slope, r2 = _ols(x, y)
        if slope != slope:  # np.isnan check
            continue
        dm = 2.0 - slope
        if dm >= 1.0 and dm <= 2.0 and r2 > best_r2:
            best_r2 = r2
            best_dm = dm

    return best_dm, best_r2

# ==============================
# BOUNDARY EXTRACTION
# ==============================

def extract_boundary(img):
    """1-pixel-wide boundary via morphological erosion."""
    return img.astype(bool) & ~erosion(img.astype(bool), _DISK1)

# ==============================
# ADAPTIVE R_VALS PER IMAGE
# ==============================

def compute_r_vals(nucleus_params):
    """
    Computes image-specific r values with adaptive R_MAX.

    R_MAX = max(R_MAX_FLOOR, R_MAX_FRAC × median minor axis) keeps
    dilation within the boundary-dominated scaling regime (A(r) ∝ r^(2−Dm)).

    Linear integer steps (arange) are used rather than geomspace because
    over integer ranges of 8–15, geomspace collapses to fewer unique values
    than WINDOW_SIZE after rounding and deduplication.
    """
    if not nucleus_params:
        return None, None
    median_minor = np.median([p['b'] * 2.0 for p in nucleus_params])
    r_max        = max(R_MAX_FLOOR, int(round(R_MAX_FRAC * median_minor)))
    r_vals       = np.arange(R_MIN, r_max + 1, dtype=int)
    if len(r_vals) < WINDOW_SIZE:
        return None, None
    return r_vals, np.log10(r_vals.astype(float))

# ==============================
# CORE DIMENSION CALCULATION
# ==============================

def compute_dm(boundary, r_vals, log_r):
    """
    Computes Dm and local R² from a boundary mask using the Minkowski
    sausage method: A(r) ∝ r^(2−Dm) → Dm = 2 − slope of log A vs log r.
    Physical constraint: 1.0 ≤ Dm ≤ 2.0 for a boundary curve in 2D.

    Returns best Dm and its R² from the sliding window with highest R².
    ScalingRange, GlobalR², and LinearityRatio are not reported:
    at standard histology resolution with nuclear minor axis ~20px,
    r spans < 1 decade — insufficient to populate these metrics
    meaningfully. Both real and smooth null boundaries fall in the
    boundary-dominated regime (A(r) ≈ P·r) over this range, making
    global linearity and scaling range non-discriminating.
    Consistent with Landini & Rippin (1993): nuclear boundaries are
    asymptotic fractals, not true scale-free fractals.
    """
    if boundary.sum() == 0:
        return np.nan, np.nan

    dist_map = distance_transform_edt(~boundary)
    areas    = (dist_map[:, :, np.newaxis] <= r_vals).sum(axis=(0, 1)).astype(float)

    valid = areas > 0
    if valid.sum() < WINDOW_SIZE:
        return np.nan, np.nan

    lr    = log_r[valid]
    log_A = np.log10(areas[valid])

    return _best_dm_r2(lr, log_A, WINDOW_SIZE)

# ==============================
# NULL MODEL: SHAPE-PARAMETER PERTURBATION
# ==============================

def precompute_nucleus_params(img):
    """
    Extracts per-nucleus ellipse parameters once per image.
    Stores only scalars — reused across all N_SIM simulations.
    """
    labeled = label(img)
    params  = []
    for prop in regionprops(labeled):
        a = prop.major_axis_length / 2.0
        b = prop.minor_axis_length / 2.0
        if a < 1.0 or b < 1.0:
            continue
        cy, cx = prop.centroid
        params.append({'cy': cy, 'cx': cx, 'a': a, 'b': b,
                        'ori': prop.orientation})
    return params

def build_perturbed_boundary(nucleus_params, img_shape):
    """
    Generates one null boundary by perturbing ellipse shape parameters
    with small Gaussian noise (SIGMA_AXIS=10%, SIGMA_ANGLE~3°) and
    rasterizing each nucleus independently via draw_ellipse.

    Guarantees a connected closed boundary at every realisation:
    draw_ellipse always produces a filled region; extract_boundary of
    a filled rasterized ellipse is always a connected 1-pixel-wide curve.
    The Minkowski sausage relationship A(r) ∝ r^(2−Dm) is therefore
    valid for every null realisation — unlike pixel translocation
    (Reiss et al., 2015 applied to boundaries) which produces isolated
    pixel clouds where A(r) ∝ r² and Dm → 0.
    """
    filled_canvas = np.zeros(img_shape, dtype=bool)
    for p in nucleus_params:
        a_p   = max(1.0, p['a'] + np.random.normal(0.0, SIGMA_AXIS * p['a']))
        b_p   = max(1.0, p['b'] + np.random.normal(0.0, SIGMA_AXIS * p['b']))
        ori_p = p['ori'] + np.random.normal(0.0, SIGMA_ANGLE)

        nucleus_mask = np.zeros(img_shape, dtype=np.uint8)
        rr, cc = draw_ellipse(
            int(round(p['cy'])), int(round(p['cx'])),
            int(round(a_p)),     int(round(b_p)),
            shape=img_shape,     rotation=ori_p)
        nucleus_mask[rr, cc] = 1
        filled_canvas |= (nucleus_mask > 0)

    # Extract boundary once from aggregated shape
    return extract_boundary(filled_canvas)

# ------------------------------
# Parallel null simulation helper
# ------------------------------

def compute_null_dm(nucleus_params, img_shape, r_vals, log_r):
    mask = build_perturbed_boundary(nucleus_params, img_shape)
    return compute_dm(mask, r_vals, log_r)

# ==============================
# MAIN ANALYSIS ENGINE
# ==============================

def run_minkowski_validation():
    files   = [f for f in os.listdir(INPUT_DIR) if f.endswith('.tif')]
    results = []

    print(f"Analyzing {len(files)} images for Dm Validation...")

    for f in tqdm(files, desc="Processing"):
        raw_img = imread(os.path.join(INPUT_DIR, f))
        if raw_img.ndim == 3:
            raw_img = raw_img[:, :, 0]
        img = (raw_img > 0).astype(np.uint8)

        nucleus_params = precompute_nucleus_params(img)
        if not nucleus_params:
            continue

        r_vals, log_r = compute_r_vals(nucleus_params)
        if r_vals is None:
            continue

        # Real image
        dm_r, r2_r = compute_dm(extract_boundary(img), r_vals, log_r)
        if np.isnan(dm_r) or r2_r < R2_THRESHOLD:
            continue

        # Null model simulations (parallelized over independent draws)
        null_dm_list = []
        with ProcessPoolExecutor(max_workers=min(N_SIM, os.cpu_count() or 1)) as executor:
            futures = [executor.submit(compute_null_dm, nucleus_params, img.shape, r_vals, log_r)
                       for _ in range(N_SIM)]
            for fut in as_completed(futures):
                dm_n, _ = fut.result()
                if not np.isnan(dm_n):
                    null_dm_list.append(dm_n)

        if not null_dm_list:
            continue

        results.append({
            'Image':       f,
            'Dm_real':     dm_r,
            'R2_real':     r2_r,
            'Dm_null_mean': np.mean(null_dm_list)
        })

    return pd.DataFrame(results).dropna()

# ==============================
# SCIENTIFIC REPORTING
# ==============================

def scientific_reporting(df):
    print("\n" + "="*60)
    print("SCIENTIFIC VALIDATION: Dm BOUNDARY COMPLEXITY")
    print("="*60)

    r_vals = df['Dm_real'].values
    n_vals = df['Dm_null_mean'].values

    if len(r_vals) < 2:
        print("Insufficient valid pairs for statistical testing.")
        return
    if len(r_vals) != len(n_vals):
        print("Length mismatch — cannot run paired test.")
        return

    stat, p_val = stats.wilcoxon(r_vals, n_vals)
    diff        = r_vals - n_vals
    cohen_d     = np.mean(diff) / np.std(diff, ddof=1)

    print(f"\nMinkowski Dimension (Dm):")
    print(f"  N pairs:      {len(r_vals)}")
    print(f"  Real Median:  {np.median(r_vals):.4f}")
    print(f"  Null Median:  {np.median(n_vals):.4f}")
    print(f"  Wilcoxon p:   {p_val:.4e}")
    print(f"  Cohen's d:    {cohen_d:.3f}")

# ==============================
# VISUALIZATION
# ==============================

def visualize(df):
    r_vals = df['Dm_real'].values
    n_vals = df['Dm_null_mean'].values

    plt.figure(figsize=(8, 5))
    plt.hist(r_vals, bins=25, alpha=0.5, color='blue',
             edgecolor='black', label='Real')
    plt.hist(n_vals, bins=25, alpha=0.5, color='gray',
             edgecolor='black', label='Null (Perturbed Ellipse)')
    plt.axvline(np.median(r_vals), color='blue',
                linestyle='dashed', linewidth=2, label='Real Median')
    plt.axvline(np.median(n_vals), color='gray',
                linestyle='dashed', linewidth=2, label='Null Median')
    plt.xlabel('Minkowski Dimension (Dm)')
    plt.ylabel('Frequency')
    plt.title('Dm: Real Nuclear Boundaries vs Perturbed Ellipse Null',
              fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    df = run_minkowski_validation()
    print(f"\nTotal images passing all filters: {len(df)}")
    df.to_csv(f"{OUTPUT_PREFIX}_results.csv", index=False)
    print(f"✓ Results saved to {OUTPUT_PREFIX}_results.csv")
    scientific_reporting(df)
    visualize(df)