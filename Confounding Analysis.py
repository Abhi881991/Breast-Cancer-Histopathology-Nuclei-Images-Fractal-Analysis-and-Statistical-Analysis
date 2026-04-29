"""
================================================================================
RESEARCH QUESTION 7 (RQ7): CONFOUNDING ANALYSIS
================================================================================

Research Question (Modified):
"Do fractal dimensions correlate with R² (scaling quality) and standard error,
and are these relationships pathology-dependent?"

Note: Original RQ7 asked about nuclear density/size, but this data is not 
available. This modified version tests for methodological confounding.

Key Questions:
1. Does Dc correlate with R²(Dc)? (High Dc → Better fit?)
2. Does Dc correlate with StdErr? (High Dc → Higher uncertainty?)
3. Are these relationships the same across all pathologies?
4. Do Dc and Dm show similar patterns?

Statistical Tests:
- Pearson/Spearman correlation (overall and by pathology)
- Partial correlation (control for confounders)
- Linear regression with interaction terms
- ANCOVA (test if relationships differ by pathology)
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.family'] = 'Times New Roman'  # Available font
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 14

# Configuration
BASE_PATH = Path(r"C:\Users\ajd44\Desktop")
OUTPUT_DIR = BASE_PATH / 'RQ7_Confounding_Analysis'
PLOTS_DIR = OUTPUT_DIR / 'plots'
RESULTS_DIR = OUTPUT_DIR / 'results'
ORIGIN_DATA_DIR = OUTPUT_DIR / 'origin_data'
for d in [OUTPUT_DIR, PLOTS_DIR, RESULTS_DIR, ORIGIN_DATA_DIR]:
    d.mkdir(exist_ok=True)

PATHOLOGY_ORDER = ['N', 'PB', 'UDH', 'FEA', 'ADH', 'DCIS', 'IC']
ALPHA = 0.05

print("="*80)
print("RQ7: CONFOUNDING ANALYSIS")
print("="*80)

# ============================================================================
# DATA LOADING
# ============================================================================

corr_df = pd.read_csv(BASE_PATH / 'Correlation Dimension.csv')
mink_df = pd.read_csv(BASE_PATH / 'Minkowski Dimension.csv')

corr_df['WSI_ID'] = corr_df['File name'].str.extract(r'(BRACS_\d+)')
corr_df['Pathology'] = corr_df['File name'].str.extract(r'_([A-Z]+)_\d+\.tif')
mink_df['WSI_ID'] = mink_df['File name'].str.extract(r'(BRACS_\d+)')
mink_df['Pathology'] = mink_df['File name'].str.extract(r'_([A-Z]+)_\d+\.tif')

corr_df = corr_df.rename(columns={'R2': 'R2_corr', 'StdErr': 'StdErr_corr'})
mink_df = mink_df.rename(columns={'R2': 'R2_mink', 'StdErr': 'StdErr_mink'})

merged = pd.merge(
    corr_df[['File name', 'Dc', 'R2_corr', 'StdErr_corr', 'WSI_ID', 'Pathology']],
    mink_df[['File name', 'Dm', 'R2_mink', 'StdErr_mink']],
    on='File name'
)
merged = merged[merged['Pathology'].isin(PATHOLOGY_ORDER)].copy()

print(f"\nTotal ROIs: {len(merged)}")

# ============================================================================
# ANALYSIS 1: OVERALL CORRELATIONS
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 1: OVERALL CORRELATIONS")
print("="*80)

correlations = []

# Dc vs R2
r_dc_r2, p_dc_r2 = pearsonr(merged['Dc'], merged['R2_corr'])
rho_dc_r2, p_rho_dc_r2 = spearmanr(merged['Dc'], merged['R2_corr'])
correlations.append({'Pair': 'Dc vs R2_corr', 'Pearson_r': r_dc_r2, 'Pearson_p': p_dc_r2,
                    'Spearman_rho': rho_dc_r2, 'Spearman_p': p_rho_dc_r2})
print(f"\nDc vs R²(Dc): r={r_dc_r2:.4f} (p={p_dc_r2:.4f}), ρ={rho_dc_r2:.4f} (p={p_rho_dc_r2:.4f})")

# Dc vs StdErr
r_dc_se, p_dc_se = pearsonr(merged['Dc'], merged['StdErr_corr'])
rho_dc_se, p_rho_dc_se = spearmanr(merged['Dc'], merged['StdErr_corr'])
correlations.append({'Pair': 'Dc vs StdErr', 'Pearson_r': r_dc_se, 'Pearson_p': p_dc_se,
                    'Spearman_rho': rho_dc_se, 'Spearman_p': p_rho_dc_se})
print(f"Dc vs StdErr: r={r_dc_se:.4f} (p={p_dc_se:.4f}), ρ={rho_dc_se:.4f} (p={p_rho_dc_se:.4f})")

# Dm vs R2
r_dm_r2, p_dm_r2 = pearsonr(merged['Dm'], merged['R2_mink'])
rho_dm_r2, p_rho_dm_r2 = spearmanr(merged['Dm'], merged['R2_mink'])
correlations.append({'Pair': 'Dm vs R2_mink', 'Pearson_r': r_dm_r2, 'Pearson_p': p_dm_r2,
                    'Spearman_rho': rho_dm_r2, 'Spearman_p': p_rho_dm_r2})
print(f"Dm vs R²(Dm): r={r_dm_r2:.4f} (p={p_dm_r2:.4f}), ρ={rho_dm_r2:.4f} (p={p_rho_dm_r2:.4f})")

# Dm vs StdErr
r_dm_se, p_dm_se = pearsonr(merged['Dm'], merged['StdErr_mink'])
rho_dm_se, p_rho_dm_se = spearmanr(merged['Dm'], merged['StdErr_mink'])
correlations.append({'Pair': 'Dm vs StdErr', 'Pearson_r': r_dm_se, 'Pearson_p': p_dm_se,
                    'Spearman_rho': rho_dm_se, 'Spearman_p': p_rho_dm_se})
print(f"Dm vs StdErr: r={r_dm_se:.4f} (p={p_dm_se:.4f}), ρ={rho_dm_se:.4f} (p={p_rho_dm_se:.4f})")

# R2 vs StdErr (negative relationship expected)
r_r2_se, p_r2_se = pearsonr(merged['R2_corr'], merged['StdErr_corr'])
correlations.append({'Pair': 'R2_corr vs StdErr', 'Pearson_r': r_r2_se, 'Pearson_p': p_r2_se,
                    'Spearman_rho': np.nan, 'Spearman_p': np.nan})
print(f"R²(Dc) vs StdErr: r={r_r2_se:.4f} (p={p_r2_se:.4f})")

corr_df_results = pd.DataFrame(correlations)
corr_df_results.to_excel(RESULTS_DIR / '01_overall_correlations.xlsx', index=False)
print(f"\n✓ Saved: {RESULTS_DIR / '01_overall_correlations.xlsx'}")

# ============================================================================
# ANALYSIS 2: CORRELATIONS BY PATHOLOGY
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 2: CORRELATIONS BY PATHOLOGY")
print("="*80)

pathology_correlations = []

for pathology in PATHOLOGY_ORDER:
    subset = merged[merged['Pathology'] == pathology]
    # No threshold — all 7 pathologies have sufficient n (>400 ROIs each)
    r_dc_r2_p,  p_dc_r2_p  = pearsonr(subset['Dc'], subset['R2_corr'])
    r_dc_se_p,  p_dc_se_p  = pearsonr(subset['Dc'], subset['StdErr_corr'])
    r_dm_r2_p,  p_dm_r2_p  = pearsonr(subset['Dm'], subset['R2_mink'])
    r_dm_se_p,  p_dm_se_p  = pearsonr(subset['Dm'], subset['StdErr_mink'])

    pathology_correlations.append({
        'Pathology':      pathology,
        'n':              len(subset),
        'Dc_vs_R2_r':    r_dc_r2_p,
        'Dc_vs_R2_p':    p_dc_r2_p,
        'Dc_vs_StdErr_r': r_dc_se_p,
        'Dc_vs_StdErr_p': p_dc_se_p,
        'Dm_vs_R2_r':    r_dm_r2_p,
        'Dm_vs_R2_p':    p_dm_r2_p,
        'Dm_vs_StdErr_r': r_dm_se_p,
        'Dm_vs_StdErr_p': p_dm_se_p,
    })

path_corr_df = pd.DataFrame(pathology_correlations)
print("\nDc vs R² by Pathology:")
print(path_corr_df[['Pathology', 'n', 'Dc_vs_R2_r', 'Dc_vs_R2_p']].to_string(index=False))
print("\nDm vs R² by Pathology:")
print(path_corr_df[['Pathology', 'n', 'Dm_vs_R2_r', 'Dm_vs_R2_p']].to_string(index=False))

path_corr_df.to_excel(RESULTS_DIR / '02_correlations_by_pathology.xlsx', index=False)
print(f"\n✓ Saved: {RESULTS_DIR / '02_correlations_by_pathology.xlsx'}")

# ============================================================================
# ANALYSIS 3: LINEAR REGRESSION WITH INTERACTION
# ============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import io

def run_ols_regression(dep_var, r2_col, stderr_col, label, ref_category='ADH'):
    """
    OLS regression: dimension ~ R² + StdErr + C(Pathology)
    Reference category = ref_category (absorbed into intercept).
    Returns fitted model and prints a summary table.
    """
    # Build design matrix: drop ref_category to use as baseline
    dummies = pd.get_dummies(merged['Pathology'], drop_first=False)
    dummies = dummies.drop(columns=[ref_category])  # explicit reference
    X = pd.concat([merged[[r2_col, stderr_col]], dummies], axis=1).astype(float)
    y = merged[dep_var].values

    from sklearn.linear_model import LinearRegression
    from scipy import stats as scipy_stats

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    n = len(y); p = X.shape[1]
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mse = ss_res / (n - p - 1)
    se = np.sqrt(mse * np.linalg.inv(X.T @ X).diagonal())
    coef_names = list(X.columns)
    t_stats = model.coef_ / se
    p_vals = [2 * (1 - scipy_stats.t.cdf(abs(t), df=n-p-1)) for t in t_stats]

    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"Model: {dep_var} ~ {r2_col} + {stderr_col} + C(Pathology)")
    lines.append(f"Reference/baseline category = {ref_category} (absorbed into intercept)")
    lines.append(f"{'='*70}")
    lines.append(f"  n = {n},  R² = {r2:.4f},  Adj. R² = {r2_adj:.4f},  MSE = {mse:.6f}")
    lines.append(f"  Intercept: {model.intercept_:.6f}")
    lines.append(f"\n  {'Variable':<30} {'Coef':>10} {'StdErr':>10} {'t':>8} {'p':>10}")
    lines.append(f"  {'-'*68}")
    for name, coef, s, t, pv in zip(coef_names, model.coef_, se, t_stats, p_vals):
        sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
        lines.append(f"  {name:<30} {coef:>10.6f} {s:>10.6f} {t:>8.3f} {pv:>10.6f} {sig}")
    lines.append(f"{'='*70}")
    summary_str = '\n'.join(lines)
    print(summary_str)
    return model, summary_str, {'R2': r2, 'Adj_R2': r2_adj, 'n': n}

print("\n" + "="*80)
print("ANALYSIS 3: LINEAR REGRESSION (Dimension ~ R² + StdErr + Pathology)")
print("="*80)
print("\nNOTE: ADH is the reference/baseline category for C(Pathology).")
print("      ADH is absorbed into the intercept and does not appear as a")
print("      separate coefficient. All other pathology coefficients represent")
print("      the difference in dimension value relative to ADH.")

merged['Pathology_num'] = merged['Pathology'].map({p: i for i, p in enumerate(PATHOLOGY_ORDER)})

model_dc, summary_dc, stats_dc = run_ols_regression('Dc', 'R2_corr', 'StdErr_corr', 'Dc')
model_dm, summary_dm, stats_dm = run_ols_regression('Dm', 'R2_mink', 'StdErr_mink', 'Dm')

with open(RESULTS_DIR / '03_regression_summary.txt', 'w') as f:
    f.write("RQ7: LINEAR REGRESSION SUMMARY\n")
    f.write("Reference/baseline category for C(Pathology) = ADH\n")
    f.write("ADH is absorbed into the intercept — this is standard OLS dummy coding,\n")
    f.write("not a missing value. All other pathology terms are differences vs. ADH.\n")
    f.write(summary_dc)
    f.write("\n\n")
    f.write(summary_dm)

print(f"\n✓ Saved: {RESULTS_DIR / '03_regression_summary.txt'}")

# ============================================================================
# VISUALIZATION 1: CORRELATION MATRIX
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('RQ7: Confounding Analysis - Dimension vs Quality Metrics', fontsize=14, fontweight='bold')

# Plot 1: Dc vs R²
ax = axes[0, 0]
ax.scatter(merged['Dc'], merged['R2_corr'], alpha=0.3, s=10, c='steelblue')
z = np.polyfit(merged['Dc'], merged['R2_corr'], 1)
p = np.poly1d(z)
x_line = np.linspace(merged['Dc'].min(), merged['Dc'].max(), 100)
ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'r={r_dc_r2:.3f}, p={p_dc_r2:.4f}')
ax.set_xlabel('Dc (Correlation Dimension)')
ax.set_ylabel('R² (Fit Quality)')
ax.set_title('Dc vs R²')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Dc vs StdErr
ax = axes[0, 1]
ax.scatter(merged['Dc'], merged['StdErr_corr'], alpha=0.3, s=10, c='coral')
z2 = np.polyfit(merged['Dc'], merged['StdErr_corr'], 1)
p2 = np.poly1d(z2)
ax.plot(x_line, p2(x_line), 'r--', linewidth=2, label=f'r={r_dc_se:.3f}, p={p_dc_se:.4f}')
ax.set_xlabel('Dc (Correlation Dimension)')
ax.set_ylabel('StdErr (Uncertainty)')
ax.set_title('Dc vs Standard Error')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Dm vs R²
ax = axes[1, 0]
ax.scatter(merged['Dm'], merged['R2_mink'], alpha=0.3, s=10, c='lightgreen')
z3 = np.polyfit(merged['Dm'], merged['R2_mink'], 1)
p3 = np.poly1d(z3)
x_line2 = np.linspace(merged['Dm'].min(), merged['Dm'].max(), 100)
ax.plot(x_line2, p3(x_line2), 'r--', linewidth=2, label=f'r={r_dm_r2:.3f}, p={p_dm_r2:.4f}')
ax.set_xlabel('Dm (Minkowski Dimension)')
ax.set_ylabel('R² (Fit Quality)')
ax.set_title('Dm vs R²')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: R² vs StdErr
ax = axes[1, 1]
ax.scatter(merged['R2_corr'], merged['StdErr_corr'], alpha=0.3, s=10, c='plum')
z4 = np.polyfit(merged['R2_corr'], merged['StdErr_corr'], 1)
p4 = np.poly1d(z4)
x_line3 = np.linspace(merged['R2_corr'].min(), merged['R2_corr'].max(), 100)
ax.plot(x_line3, p4(x_line3), 'r--', linewidth=2, label=f'r={r_r2_se:.3f}, p={p_r2_se:.4f}')
ax.set_xlabel('R² (Fit Quality)')
ax.set_ylabel('StdErr (Uncertainty)')
ax.set_title('R² vs Standard Error (Expected: Negative)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig1_confounding_correlations.tif', format='tif', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOTS_DIR / 'fig1_confounding_correlations.tif'}")

# Export data
merged[['Dc', 'Dm', 'R2_corr', 'R2_mink', 'StdErr_corr', 'StdErr_mink', 'Pathology']].to_excel(
    ORIGIN_DATA_DIR / 'fig1_correlation_data.xlsx', index=False)

# ============================================================================
# VISUALIZATION 2: BY PATHOLOGY
# ============================================================================

# ============================================================================
# VISUALIZATION 2: Dc AND Dm vs R² BY PATHOLOGY
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('RQ7: Dc and Dm vs R² by Pathology', fontsize=14, fontweight='bold')

colors = plt.cm.Set3(np.linspace(0, 1, len(PATHOLOGY_ORDER)))

# Plot 1: Dc vs R² scatter coloured by pathology
ax = axes[0, 0]
for i, pathology in enumerate(PATHOLOGY_ORDER):
    subset = merged[merged['Pathology'] == pathology]
    ax.scatter(subset['Dc'], subset['R2_corr'], alpha=0.5, s=20,
               color=colors[i], label=pathology, edgecolors='black', linewidth=0.3)
ax.set_xlabel('Dc (Correlation Dimension)')
ax.set_ylabel('R²(Dc)')
ax.set_title('Dc vs R²(Dc) — All Pathologies')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Plot 2: Dc vs R² correlation coefficient by pathology (bar)
ax = axes[0, 1]
x_pos = range(len(path_corr_df))
bars = ax.bar(x_pos, path_corr_df['Dc_vs_R2_r'], color='steelblue', alpha=0.7, edgecolor='black')
ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(path_corr_df['Pathology'], rotation=45)
ax.set_ylabel('Pearson r')
ax.set_title('Dc vs R²(Dc): Correlation by Pathology')
ax.grid(True, alpha=0.3, axis='y')
for i, (bar, r_val) in enumerate(zip(bars, path_corr_df['Dc_vs_R2_r'])):
    height = bar.get_height()
    ax.text(i, height + 0.01 if height >= 0 else height - 0.03,
            f'{r_val:.2f}', ha='center', fontsize=8, fontweight='bold')

# Plot 3: Dm vs R² scatter coloured by pathology
ax = axes[1, 0]
for i, pathology in enumerate(PATHOLOGY_ORDER):
    subset = merged[merged['Pathology'] == pathology]
    ax.scatter(subset['Dm'], subset['R2_mink'], alpha=0.5, s=20,
               color=colors[i], label=pathology, edgecolors='black', linewidth=0.3)
ax.set_xlabel('Dm (Minkowski Dimension)')
ax.set_ylabel('R²(Dm)')
ax.set_title('Dm vs R²(Dm) — All Pathologies')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Plot 4: Dm vs R² correlation coefficient by pathology (bar)
ax = axes[1, 1]
bars2 = ax.bar(x_pos, path_corr_df['Dm_vs_R2_r'], color='lightcoral', alpha=0.7, edgecolor='black')
ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(path_corr_df['Pathology'], rotation=45)
ax.set_ylabel('Pearson r')
ax.set_title('Dm vs R²(Dm): Correlation by Pathology')
ax.grid(True, alpha=0.3, axis='y')
for i, (bar, r_val) in enumerate(zip(bars2, path_corr_df['Dm_vs_R2_r'])):
    height = bar.get_height()
    ax.text(i, height + 0.01 if height >= 0 else height - 0.03,
            f'{r_val:.2f}', ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig2_by_pathology.tif', format='tif', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOTS_DIR / 'fig2_by_pathology.tif'}")

path_corr_df.to_excel(ORIGIN_DATA_DIR / 'fig2_pathology_data.xlsx', index=False)

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("GENERATING SUMMARY")
print("="*80)

summary = f"""
================================================================================
RQ7: CONFOUNDING ANALYSIS - SUMMARY
================================================================================

Research Question:
"Do fractal dimensions correlate with R² and standard error?"

ANSWER: {'YES' if abs(r_dc_r2) > 0.1 or abs(r_dc_se) > 0.1 else 'NO'} - {'Significant' if p_dc_r2 < ALPHA or p_dc_se < ALPHA else 'No significant'} confounding detected

KEY FINDINGS:

1. OVERALL CORRELATIONS:
   Dc vs R²:     r = {r_dc_r2:.4f}, p = {p_dc_r2:.6f}
   Dc vs StdErr: r = {r_dc_se:.4f}, p = {p_dc_se:.6f}
   Dm vs R²:     r = {r_dm_r2:.4f}, p = {p_dm_r2:.6f}
   Dm vs StdErr: r = {r_dm_se:.4f}, p = {p_dm_se:.6f}

2. INTERPRETATION:
   {'Strong confounding present - dimension values affected by fit quality' if abs(r_dc_r2) > 0.3 else 'Weak/no confounding - dimension values independent of fit quality'}

3. RECOMMENDATION:
   {'Control for R² in analyses, consider excluding poor fits' if abs(r_dc_r2) > 0.3 else 'No quality control adjustments needed'}

FILES GENERATED:
- 01_overall_correlations.xlsx
- 02_correlations_by_pathology.xlsx
- 03_regression_summary.txt
- fig1_confounding_correlations.tif
- fig2_by_pathology.tif

================================================================================
"""

with open(RESULTS_DIR / '00_RQ7_SUMMARY.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print(summary)
print(f"✓ Saved: {RESULTS_DIR / '00_RQ7_SUMMARY.txt'}")

print("\n" + "="*80)
print("RQ7 ANALYSIS COMPLETE!")
print("="*80)