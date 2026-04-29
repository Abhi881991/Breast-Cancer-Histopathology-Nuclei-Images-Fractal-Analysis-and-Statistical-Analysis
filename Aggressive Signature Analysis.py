"""
================================================================================
RESEARCH QUESTION 9 (RQ9): AGGRESSIVE LESION SIGNATURE ANALYSIS
================================================================================

Research Question:
"What is the fractal 'signature' of the most aggressive lesion in each patient,
and does it predict overall pathological diagnosis?"

Signature Components:
- Max Dc, Min Dc, Range Dc
- Max Dm, Min Dm, Range Dm (ADDED)
- Pathology at max Dc and max Dm ROIs
- Combined signature (max Dc + max Dm)

Tests both dimensions equally for comprehensive signature analysis.
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.family'] = 'Times New Roman'  # Available font
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 14

BASE_PATH = Path(r"C:\Users\ajd44\Desktop")
OUTPUT_DIR = BASE_PATH / 'RQ9_Aggressive_Signature'
PLOTS_DIR = OUTPUT_DIR / 'plots'
RESULTS_DIR = OUTPUT_DIR / 'results'
ORIGIN_DATA_DIR = OUTPUT_DIR / 'origin_data'
for d in [OUTPUT_DIR, PLOTS_DIR, RESULTS_DIR, ORIGIN_DATA_DIR]:
    d.mkdir(exist_ok=True)

PATHOLOGY_ORDER = ['N', 'PB', 'UDH', 'FEA', 'ADH', 'DCIS', 'IC']
PATHOLOGY_SEVERITY = {p: i for i, p in enumerate(PATHOLOGY_ORDER)}

print("="*80)
print("RQ9: AGGRESSIVE LESION SIGNATURE (Dc AND Dm) - CORRECTED")
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

merged = pd.merge(
    corr_df[['File name', 'Dc', 'Pathology', 'WSI_ID']],
    mink_df[['File name', 'Dm']],
    on='File name'
)
merged = merged[merged['Pathology'].isin(PATHOLOGY_ORDER)].copy()
merged['Severity'] = merged['Pathology'].map(PATHOLOGY_SEVERITY)

print(f"Total ROIs: {len(merged)}")
print(f"Total WSIs: {merged['WSI_ID'].nunique()}")

# ============================================================================
# ANALYSIS 1: SIGNATURE EXTRACTION (DC AND DM)
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 1: EXTRACTING SIGNATURES (Dc AND Dm)")
print("="*80)

wsi_signatures = merged.groupby('WSI_ID').agg({
    'Dc': ['mean', 'std', 'min', 'max', 'count'],
    'Dm': ['mean', 'std', 'min', 'max'],
    'Pathology': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
    'Severity': 'max'
}).reset_index()

wsi_signatures.columns = ['WSI_ID', 'Dc_mean', 'Dc_std', 'Dc_min', 'Dc_max', 'n_ROIs',
                          'Dm_mean', 'Dm_std', 'Dm_min', 'Dm_max', 'Dominant_Pathology', 
                          'Max_Severity']

wsi_signatures['Dc_range'] = wsi_signatures['Dc_max'] - wsi_signatures['Dc_min']
wsi_signatures['Dm_range'] = wsi_signatures['Dm_max'] - wsi_signatures['Dm_min']

# Pathology at max Dc
max_dc_info = merged.loc[merged.groupby('WSI_ID')['Dc'].idxmax()][['WSI_ID', 'Pathology', 'Dc', 'Dm']]
max_dc_info.columns = ['WSI_ID', 'MaxDc_Pathology', 'MaxDc_Value', 'MaxDc_Dm']
wsi_signatures = wsi_signatures.merge(max_dc_info, on='WSI_ID')

# Pathology at max Dm
max_dm_info = merged.loc[merged.groupby('WSI_ID')['Dm'].idxmax()][['WSI_ID', 'Pathology', 'Dm', 'Dc']]
max_dm_info.columns = ['WSI_ID', 'MaxDm_Pathology', 'MaxDm_Value', 'MaxDm_Dc']
wsi_signatures = wsi_signatures.merge(max_dm_info, on='WSI_ID')

# Min values
min_dc = merged.loc[merged.groupby('WSI_ID')['Dc'].idxmin()][['WSI_ID', 'Pathology', 'Dc']]
min_dc.columns = ['WSI_ID', 'MinDc_Pathology', 'MinDc_Value']
wsi_signatures = wsi_signatures.merge(min_dc, on='WSI_ID')

min_dm = merged.loc[merged.groupby('WSI_ID')['Dm'].idxmin()][['WSI_ID', 'Pathology', 'Dm']]
min_dm.columns = ['WSI_ID', 'MinDm_Pathology', 'MinDm_Value']
wsi_signatures = wsi_signatures.merge(min_dm, on='WSI_ID')

wsi_signatures['Is_Mixed'] = merged.groupby('WSI_ID')['Pathology'].nunique().values > 1

print(f"\nWSIs analyzed: {len(wsi_signatures)}")
print(f"Mixed pathology: {wsi_signatures['Is_Mixed'].sum()} ({100*wsi_signatures['Is_Mixed'].mean():.1f}%)")

wsi_signatures.to_excel(RESULTS_DIR / '01_wsi_signatures.xlsx', index=False)
print(f"✓ Saved: {RESULTS_DIR / '01_wsi_signatures.xlsx'}")

# ============================================================================
# ANALYSIS 2: PREDICTIVE COMPARISON (DC AND DM)
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 2: PREDICTIVE POWER COMPARISON")
print("="*80)
print("⚠ CORRECTED: Binary labels fixed (FEA/ADH = Benign, not Malignant)")
print("  Binary classification: Benign (N/PB/UDH/FEA/ADH) vs Malignant (DCIS/IC)")
print("="*80)

# Binary classification - CORRECTED
# FEA and ADH are PRE-MALIGNANT (atypical hyperplasias), NOT invasive carcinoma
# Only DCIS and IC are truly malignant
wsi_signatures['Binary'] = wsi_signatures['Dominant_Pathology'].map({
    'N': 0, 'PB': 0, 'UDH': 0,
    'FEA': 0, 'ADH': 0,  # ← CORRECTED: Pre-malignant, not malignant
    'DCIS': 1, 'IC': 1   # True malignancy
})

y_true = wsi_signatures['Binary'].values

metrics = {
    'Dc_mean': wsi_signatures['Dc_mean'],
    'Dc_max': wsi_signatures['Dc_max'],
    'Dc_range': wsi_signatures['Dc_range'],
    'Dm_mean': wsi_signatures['Dm_mean'],
    'Dm_max': wsi_signatures['Dm_max'],
    'Dm_range': wsi_signatures['Dm_range'],
    'Combined_max': wsi_signatures['Dc_max'] + wsi_signatures['Dm_max']
}

results = []
print(f"\nAUC (Benign vs Malignant):")
for name, values in metrics.items():
    auc_val = roc_auc_score(y_true, values)
    r_corr, p_corr = pearsonr(values, wsi_signatures['Max_Severity'])
    results.append({
        'Metric': name,
        'AUC': auc_val,
        'Corr_Severity': r_corr,
        'P_value': p_corr
    })
    print(f"  {name:15s}: AUC={auc_val:.4f}, r={r_corr:.4f}, p={p_corr:.6f}")

results_df = pd.DataFrame(results)
results_df.to_excel(RESULTS_DIR / '02_predictive_comparison.xlsx', index=False)
print(f"\n✓ Saved: {RESULTS_DIR / '02_predictive_comparison.xlsx'}")

best = results_df.loc[results_df['AUC'].idxmax()]
print(f"\n⭐ BEST: {best['Metric']} (AUC={best['AUC']:.4f})")

# ============================================================================
# ANALYSIS 3: SIGNATURE BY PATHOLOGY (DC AND DM)
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 3: SIGNATURES BY PATHOLOGY")
print("="*80)

sig_by_path = wsi_signatures.groupby('Dominant_Pathology').agg({
    'Dc_max': ['mean', 'std'], 'Dc_min': ['mean', 'std'], 'Dc_range': ['mean', 'std'],
    'Dm_max': ['mean', 'std'], 'Dm_min': ['mean', 'std'], 'Dm_range': ['mean', 'std'],
    'n_ROIs': 'mean'
}).round(4)

print("\nMax values by pathology:")
print(sig_by_path[['Dc_max', 'Dm_max']])

# ANOVA
groups_dc = [wsi_signatures[wsi_signatures['Dominant_Pathology']==p]['Dc_max'].values for p in PATHOLOGY_ORDER]
groups_dm = [wsi_signatures[wsi_signatures['Dominant_Pathology']==p]['Dm_max'].values for p in PATHOLOGY_ORDER]
f_dc, p_dc = stats.f_oneway(*groups_dc)
f_dm, p_dm = stats.f_oneway(*groups_dm)

print(f"\nANOVA Results:")
print(f"  Max Dc: F={f_dc:.4f}, p={p_dc:.6f}")
print(f"  Max Dm: F={f_dm:.4f}, p={p_dm:.6f}")

sig_by_path.to_excel(RESULTS_DIR / '03_signature_by_pathology.xlsx')
print(f"✓ Saved: {RESULTS_DIR / '03_signature_by_pathology.xlsx'}")

# ============================================================================
# ANALYSIS 4: MIXED vs PURE
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 4: MIXED vs PURE WSIs")
print("="*80)

mixed = wsi_signatures[wsi_signatures['Is_Mixed']]
pure = wsi_signatures[~wsi_signatures['Is_Mixed']]

for metric in ['Dc_max', 'Dm_max', 'Dc_range', 'Dm_range']:
    t_stat, p_val = stats.ttest_ind(mixed[metric], pure[metric])
    print(f"{metric:10s}: Mixed={mixed[metric].mean():.4f}, Pure={pure[metric].mean():.4f}, "
          f"t={t_stat:.2f}, p={p_val:.4f}")

comparison = pd.DataFrame([
    {'Group': 'Mixed', 'N': len(mixed), 
     'Dc_max': mixed['Dc_max'].mean(), 'Dm_max': mixed['Dm_max'].mean(),
     'Dc_range': mixed['Dc_range'].mean(), 'Dm_range': mixed['Dm_range'].mean()},
    {'Group': 'Pure', 'N': len(pure),
     'Dc_max': pure['Dc_max'].mean(), 'Dm_max': pure['Dm_max'].mean(),
     'Dc_range': pure['Dc_range'].mean(), 'Dm_range': pure['Dm_range'].mean()}
])
comparison.to_excel(RESULTS_DIR / '04_mixed_vs_pure.xlsx', index=False)
print(f"\n✓ Saved: {RESULTS_DIR / '04_mixed_vs_pure.xlsx'}")

# ============================================================================
# VISUALIZATION 1: COMPREHENSIVE SIGNATURE PLOT
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('RQ9: Aggressive Lesion Signatures (Dc AND Dm)', fontsize=14, fontweight='bold')

# Plot 1: Max Dc by pathology
ax = axes[0, 0]
data_dc = [wsi_signatures[wsi_signatures['Dominant_Pathology']==p]['Dc_max'].values for p in PATHOLOGY_ORDER]
bp = ax.boxplot(data_dc, labels=PATHOLOGY_ORDER, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_ylabel('Max Dc')
ax.set_title(f'Max Dc by Pathology\nF={f_dc:.2f}, p={p_dc:.4f}')
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Max Dm by pathology
ax = axes[0, 1]
data_dm = [wsi_signatures[wsi_signatures['Dominant_Pathology']==p]['Dm_max'].values for p in PATHOLOGY_ORDER]
bp = ax.boxplot(data_dm, labels=PATHOLOGY_ORDER, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightcoral')
ax.set_ylabel('Max Dm')
ax.set_title(f'Max Dm by Pathology\nF={f_dm:.2f}, p={p_dm:.4f}')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Mean vs Max (Dc)
ax = axes[0, 2]
ax.scatter(wsi_signatures['Dc_mean'], wsi_signatures['Dc_max'], 
          alpha=0.5, s=30, c=wsi_signatures['Max_Severity'], cmap='RdYlGn_r')
ax.plot([1.3, 1.8], [1.3, 1.8], 'k--', alpha=0.5)
ax.set_xlabel('Mean Dc')
ax.set_ylabel('Max Dc')
ax.set_title('Mean vs Max Dc')
ax.grid(True, alpha=0.3)

# Plot 4: Range Dc by pathology
ax = axes[1, 0]
data_range_dc = [wsi_signatures[wsi_signatures['Dominant_Pathology']==p]['Dc_range'].values for p in PATHOLOGY_ORDER]
bp = ax.boxplot(data_range_dc, labels=PATHOLOGY_ORDER, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightyellow')
ax.set_ylabel('Dc Range')
ax.set_title('Dc Range by Pathology')
ax.grid(True, alpha=0.3, axis='y')

# Plot 5: Range Dm by pathology
ax = axes[1, 1]
data_range_dm = [wsi_signatures[wsi_signatures['Dominant_Pathology']==p]['Dm_range'].values for p in PATHOLOGY_ORDER]
bp = ax.boxplot(data_range_dm, labels=PATHOLOGY_ORDER, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightgreen')
ax.set_ylabel('Dm Range')
ax.set_title('Dm Range by Pathology')
ax.grid(True, alpha=0.3, axis='y')

# Plot 6: Dc vs Dm (max values)
ax = axes[1, 2]
ax.scatter(wsi_signatures['Dc_max'], wsi_signatures['Dm_max'],
          alpha=0.5, s=30, c=wsi_signatures['Max_Severity'], cmap='RdYlGn_r')
ax.set_xlabel('Max Dc')
ax.set_ylabel('Max Dm')
ax.set_title('Max Dc vs Max Dm')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig1_signatures_dc_dm.tif', format='tif', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOTS_DIR / 'fig1_signatures_dc_dm.tif'}")

# ============================================================================
# VISUALIZATION 2: ROC COMPARISON
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
fig.suptitle('RQ9: Predictive Power (Dc AND Dm)', fontsize=14, fontweight='bold')

colors = ['blue', 'red', 'purple', 'green', 'orange', 'brown', 'pink']
for (name, values), color in zip(metrics.items(), colors):
    fpr, tpr, _ = roc_curve(y_true, values)
    auc_val = roc_auc_score(y_true, values)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc_val:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves: All Signature Metrics')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig2_roc_all_metrics.tif', format='tif', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOTS_DIR / 'fig2_roc_all_metrics.tif'}")

# Export
wsi_signatures.to_excel(ORIGIN_DATA_DIR / 'wsi_signatures.xlsx', index=False)
results_df.to_excel(ORIGIN_DATA_DIR / 'predictive_metrics.xlsx', index=False)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("GENERATING SUMMARY")
print("="*80)

summary = f"""
================================================================================
RQ9: AGGRESSIVE LESION SIGNATURE - CORRECTED METHODOLOGY - SUMMARY
================================================================================

Research Question:
"What is the fractal signature of the most aggressive lesion?"

ANSWER: {best['Metric']} is best predictor (AUC={best['AUC']:.4f})

⚠ CRITICAL CORRECTION APPLIED:
Binary labels CORRECTED: Benign (N/PB/UDH/FEA/ADH) vs Malignant (DCIS/IC only)
- Previous version incorrectly labeled FEA/ADH as malignant
- FEA and ADH are pre-malignant hyperplasias, NOT invasive carcinoma

PREDICTIVE COMPARISON (CORRECTED):
  Dc metrics:
    Mean:  {results_df[results_df['Metric']=='Dc_mean']['AUC'].values[0]:.4f}
    Max:   {results_df[results_df['Metric']=='Dc_max']['AUC'].values[0]:.4f}
    Range: {results_df[results_df['Metric']=='Dc_range']['AUC'].values[0]:.4f}
  
  Dm metrics:
    Mean:  {results_df[results_df['Metric']=='Dm_mean']['AUC'].values[0]:.4f}
    Max:   {results_df[results_df['Metric']=='Dm_max']['AUC'].values[0]:.4f}
    Range: {results_df[results_df['Metric']=='Dm_range']['AUC'].values[0]:.4f}
  
  Combined: {results_df[results_df['Metric']=='Combined_max']['AUC'].values[0]:.4f}

ANOVA (Max by Pathology):
  Dc: F={f_dc:.4f}, p={p_dc:.6f}
  Dm: F={f_dm:.4f}, p={p_dm:.6f}

MIXED vs PURE:
  Dc_max: Mixed={mixed['Dc_max'].mean():.4f}, Pure={pure['Dc_max'].mean():.4f}
  Dm_max: Mixed={mixed['Dm_max'].mean():.4f}, Pure={pure['Dm_max'].mean():.4f}

FILES: 4 Excel, 2 TIF, 2 Origin
================================================================================
"""

with open(RESULTS_DIR / '00_RQ9_SUMMARY.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print(summary)
print(f"✓ Saved: {RESULTS_DIR / '00_RQ9_SUMMARY.txt'}")
print("\n" + "="*80)
print("RQ9 COMPLETE (Dc AND Dm analyzed)")
print("="*80)