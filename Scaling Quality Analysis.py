"""
================================================================================
RESEARCH QUESTION 6 (RQ6): SCALING QUALITY vs PATHOLOGY ANALYSIS
================================================================================

Research Question:
"What is the relationship between R² goodness-of-fit and pathological diagnosis? 
Do more aggressive lesions show better or worse fractal scaling behavior?"

Key Question: Does R² (scaling quality) vary systematically by pathology type?

R² Interpretation:
- R² > 0.99: Excellent fractal scaling
- R² > 0.95: Good fractal scaling  
- R² > 0.90: Acceptable scaling
- R² < 0.90: Poor scaling

From RQ1: Dc has mean R² = 0.997 (excellent), but 4 ROIs with R² < 0.95 (all Normal)
Question: Is this pattern systematic or random?
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, kruskal, shapiro, levene, pearsonr, spearmanr
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import scikit-posthocs for Dunn's test
try:
    from scikit_posthocs import posthoc_dunn
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False
    print("⚠ scikit-posthocs not available, will use alternative for non-parametric post-hoc")

plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.family'] = 'Times New Roman'  # Available font
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['legend.fontsize'] = 16

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = Path(r"C:\Users\ajd44\Desktop")
CORR_FILE = BASE_PATH / "Correlation Dimension.csv"
MINK_FILE = BASE_PATH / "Minkowski Dimension.csv"

OUTPUT_DIR = BASE_PATH / 'RQ6_Scaling_Quality_Analysis'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR = OUTPUT_DIR / 'plots'
RESULTS_DIR = OUTPUT_DIR / 'results'
ORIGIN_DATA_DIR = OUTPUT_DIR / 'origin_data'
for d in [PLOTS_DIR, RESULTS_DIR, ORIGIN_DATA_DIR]:
    d.mkdir(exist_ok=True)

ALPHA = 0.05
PATHOLOGY_ORDER = ['N', 'PB', 'UDH', 'FEA', 'ADH', 'DCIS', 'IC']
R2_EXCELLENT, R2_GOOD, R2_ACCEPTABLE = 0.99, 0.95, 0.90

print("=" * 80)
print("RQ6: SCALING QUALITY (R²) vs PATHOLOGY ANALYSIS")
print("=" * 80)

# ============================================================================
# DATA LOADING
# ============================================================================

print("\nLoading data...")
corr_df = pd.read_csv(CORR_FILE)
mink_df = pd.read_csv(MINK_FILE)

corr_df['WSI_ID'] = corr_df['File name'].str.extract(r'(BRACS_\d+)')
corr_df['Pathology'] = corr_df['File name'].str.extract(r'_([A-Z]+)_\d+\.tif')
mink_df['WSI_ID'] = mink_df['File name'].str.extract(r'(BRACS_\d+)')
mink_df['Pathology'] = mink_df['File name'].str.extract(r'_([A-Z]+)_\d+\.tif')

corr_df = corr_df.rename(columns={'R2': 'R2_corr'})
mink_df = mink_df.rename(columns={'R2': 'R2_mink'})

corr_df = corr_df[corr_df['Pathology'].isin(PATHOLOGY_ORDER)].copy()
mink_df = mink_df[mink_df['Pathology'].isin(PATHOLOGY_ORDER)].copy()

merged = pd.merge(
    corr_df[['File name', 'Dc', 'R2_corr', 'WSI_ID', 'Pathology']],
    mink_df[['File name', 'Dm', 'R2_mink']],
    on='File name'
)

print(f"Total ROIs: {len(merged)}")
print(f"\nDc R² Summary: Mean={merged['R2_corr'].mean():.6f}, Min={merged['R2_corr'].min():.6f}")
print(f"Dm R² Summary: Mean={merged['R2_mink'].mean():.6f}, Min={merged['R2_mink'].min():.6f}")

# ============================================================================
# ANALYSIS 1: DESCRIPTIVE STATISTICS BY PATHOLOGY
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 1: R² BY PATHOLOGY")
print("=" * 80)

r2_stats = merged.groupby('Pathology').agg({
    'R2_corr': ['mean', 'std', 'median', 'min', 'max'],
    'R2_mink': ['mean', 'std', 'median', 'min', 'max']
}).round(6)

print("\nCorrelation Dimension R²:")
print(r2_stats['R2_corr'])

r2_stats.to_excel(RESULTS_DIR / '01_r2_by_pathology.xlsx')
print(f"\n✓ Saved: {RESULTS_DIR / '01_r2_by_pathology.xlsx'}")

# ============================================================================
# CRITICAL STATISTICAL CORRECTIONS (BASED ON RQ1-5 FINDINGS)
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICAL ASSUMPTION CHECKS (CRITICAL)")
print("=" * 80)

print("\n⚠ APPLYING CORRECTIONS FROM RQ1-5:")
print("  • Check normality (Shapiro-Wilk)")
print("  • Check variance homogeneity (Levene)")
print("  • Calculate ICC (check clustering)")
print("  • Use non-parametric tests if assumptions violated")
print("  • Add effect sizes (Hedges' g)")
print("  • Test trend across progression")

# ============================================================================
# CHECK #1: NORMALITY (SHAPIRO-WILK)
# ============================================================================

print("\n" + "-" * 80)
print("CHECK #1: NORMALITY TEST (Shapiro-Wilk)")
print("-" * 80)

normality_results = []
all_normal_dc = True
all_normal_dm = True

for pathology in PATHOLOGY_ORDER:
    r2_dc = merged[merged['Pathology'] == pathology]['R2_corr'].values
    r2_dm = merged[merged['Pathology'] == pathology]['R2_mink'].values
    
    stat_dc, p_dc = shapiro(r2_dc)
    stat_dm, p_dm = shapiro(r2_dm)
    
    normality_results.append({
        'Pathology': pathology,
        'n': len(r2_dc),
        'Shapiro_Dc': stat_dc,
        'p_Dc': p_dc,
        'Normal_Dc': p_dc >= 0.05,
        'Shapiro_Dm': stat_dm,
        'p_Dm': p_dm,
        'Normal_Dm': p_dm >= 0.05
    })
    
    if p_dc < 0.05:
        all_normal_dc = False
    if p_dm < 0.05:
        all_normal_dm = False

normality_df = pd.DataFrame(normality_results)
print("\nNormality Test Results:")
print(normality_df.to_string(index=False))

print(f"\n⚠ NORMALITY ASSESSMENT:")
print(f"  Dc: {'ALL groups normal' if all_normal_dc else 'VIOLATED (non-normal distributions)'}")
print(f"  Dm: {'ALL groups normal' if all_normal_dm else 'VIOLATED (non-normal distributions)'}")
if not all_normal_dc or not all_normal_dm:
    print(f"  → Should use Kruskal-Wallis instead of ANOVA")

# ============================================================================
# CHECK #2: VARIANCE HOMOGENEITY (LEVENE)
# ============================================================================

print("\n" + "-" * 80)
print("CHECK #2: VARIANCE HOMOGENEITY (Levene's Test)")
print("-" * 80)

groups_dc = [merged[merged['Pathology'] == p]['R2_corr'].values for p in PATHOLOGY_ORDER]
groups_dm = [merged[merged['Pathology'] == p]['R2_mink'].values for p in PATHOLOGY_ORDER]

stat_levene_dc, p_levene_dc = levene(*groups_dc)
stat_levene_dm, p_levene_dm = levene(*groups_dm)

print(f"\nDc R²: Levene's W = {stat_levene_dc:.4f}, p = {p_levene_dc:.6f}")
print(f"  → {'Equal variances' if p_levene_dc >= 0.05 else 'VIOLATED (unequal variances)'}")

print(f"\nDm R²: Levene's W = {stat_levene_dm:.4f}, p = {p_levene_dm:.6f}")
print(f"  → {'Equal variances' if p_levene_dm >= 0.05 else 'VIOLATED (unequal variances)'}")

equal_var_dc = p_levene_dc >= 0.05
equal_var_dm = p_levene_dm >= 0.05

# ============================================================================
# CHECK #3: ICC (INTRACLASS CORRELATION - CLUSTERING)
# ============================================================================

print("\n" + "-" * 80)
print("CHECK #3: INTRACLASS CORRELATION (ICC) - Clustering Check")
print("-" * 80)

def calculate_icc(data, value_col, cluster_col):
    """Calculate ICC to check if values clustered within clusters"""
    cluster_means = data.groupby(cluster_col)[value_col].mean()
    grand_mean = data[value_col].mean()
    
    # Between-cluster variance
    n_per_cluster = data.groupby(cluster_col).size()
    ss_between = sum(n_per_cluster * (cluster_means - grand_mean)**2)
    df_between = len(cluster_means) - 1
    ms_between = ss_between / df_between
    
    # Within-cluster variance
    ss_within = sum((data[value_col] - data.groupby(cluster_col)[value_col].transform('mean'))**2)
    df_within = len(data) - len(cluster_means)
    ms_within = ss_within / df_within
    
    # ICC
    k_avg = n_per_cluster.mean()
    icc = (ms_between - ms_within) / (ms_between + (k_avg - 1) * ms_within)
    
    return icc, ms_between, ms_within

icc_dc, ms_between_dc, ms_within_dc = calculate_icc(merged, 'R2_corr', 'WSI_ID')
icc_dm, ms_between_dm, ms_within_dm = calculate_icc(merged, 'R2_mink', 'WSI_ID')

print(f"\nDc R²: ICC = {icc_dc:.3f}")
print(f"  Between-WSI variance: {100*icc_dc:.1f}%")
print(f"  Within-WSI variance:  {100*(1-icc_dc):.1f}%")
print(f"  → {'MODERATE to HIGH clustering' if icc_dc > 0.5 else 'LOW to MODERATE clustering' if icc_dc > 0.2 else 'Minimal clustering'}")

print(f"\nDm R²: ICC = {icc_dm:.3f}")
print(f"  Between-WSI variance: {100*icc_dm:.1f}%")
print(f"  Within-WSI variance:  {100*(1-icc_dm):.1f}%")
print(f"  → {'MODERATE to HIGH clustering' if icc_dm > 0.5 else 'LOW to MODERATE clustering' if icc_dm > 0.2 else 'Minimal clustering'}")

need_wsi_aggregation = (icc_dc > 0.5 or icc_dm > 0.5)

if need_wsi_aggregation:
    print(f"\n⚠ WARNING: High ICC detected!")
    print(f"  → R² values clustered within WSIs")
    print(f"  → ROI-level analysis = PSEUDOREPLICATION")
    print(f"  → Performing WSI-level aggregation now...")
    
    # ========================================================================
    # WSI-LEVEL AGGREGATION
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("WSI-LEVEL AGGREGATION (Addressing Clustering)")
    print("=" * 80)
    
    # Aggregate to WSI level — group by WSI_ID only (not WSI+Pathology,
    # which would split mixed-pathology WSIs and inflate n from 368 to 824)
    wsi_level = merged.groupby('WSI_ID').agg(
        R2_corr=('R2_corr', 'mean'),
        R2_mink=('R2_mink', 'mean'),
        Dc=('Dc', 'mean'),
        Dm=('Dm', 'mean'),
        Pathology=('Pathology', lambda x: x.mode()[0])  # majority pathology
    ).reset_index()
    
    print(f"\n✓ Aggregated from {len(merged)} ROIs to {len(wsi_level)} WSIs")
    print(f"\nWSI-level sample sizes by pathology:")
    for pathology in PATHOLOGY_ORDER:
        n_wsi = len(wsi_level[wsi_level['Pathology'] == pathology])
        n_roi = len(merged[merged['Pathology'] == pathology])
        avg_roi = n_roi / n_wsi if n_wsi > 0 else 0
        print(f"  {pathology:5s}: {n_wsi:3d} WSIs (avg {avg_roi:.1f} ROIs/WSI)")
    
    # Store both levels for comparison
    merged_roi_level = merged.copy()
    merged_wsi_level = wsi_level.copy()
    
    print(f"\n⚠ IMPORTANT:")
    print(f"  ROI-level analysis (n={len(merged_roi_level)}) shown above for reference")
    print(f"  WSI-level analysis (n={len(merged_wsi_level)}) will be performed below")
    print(f"  WSI-level is STATISTICALLY VALID (addresses pseudoreplication)")
    
    # Use WSI-level for main analysis
    merged_analysis = wsi_level.copy()
    analysis_level = 'WSI'
    
else:
    print(f"\n✓ LOW CLUSTERING:")
    print(f"  ICC < 0.5 for both dimensions")
    print(f"  ROI-level analysis is statistically appropriate")
    
    merged_analysis = merged.copy()
    merged_wsi_level = None
    analysis_level = 'ROI'

# ============================================================================
# CHECK #4: R²(Dc) vs R²(Dm) CORRELATION
# ============================================================================

print("\n" + "-" * 80)
print("CHECK #4: R²(Dc) vs R²(Dm) CORRELATION")
print("-" * 80)

r_pearson, p_pearson = pearsonr(merged['R2_corr'], merged['R2_mink'])
r_spearman, p_spearman = spearmanr(merged['R2_corr'], merged['R2_mink'])

print(f"\nPearson r = {r_pearson:.4f}, p = {p_pearson:.6f}")
print(f"Spearman ρ = {r_spearman:.4f}, p = {p_spearman:.6f}")
print(f"Shared variance: {100*r_pearson**2:.1f}%")

if abs(r_pearson) > 0.7:
    print(f"  → STRONG correlation: R² quality similar for Dc and Dm")
elif abs(r_pearson) > 0.4:
    print(f"  → MODERATE correlation: Somewhat related")
else:
    print(f"  → WEAK correlation: R² quality largely independent")

# ============================================================================
# DECISION: WHICH TESTS TO USE
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICAL TEST SELECTION")
print("=" * 80)

use_parametric_dc = all_normal_dc and equal_var_dc
use_parametric_dm = all_normal_dm and equal_var_dm

print(f"\nDc R²:")
print(f"  Normality: {'✓ Pass' if all_normal_dc else '✗ Fail'}")
print(f"  Equal variance: {'✓ Pass' if equal_var_dc else '✗ Fail'}")
print(f"  → Use: {'ANOVA + Tukey' if use_parametric_dc else 'Kruskal-Wallis + Dunn'}")

print(f"\nDm R²:")
print(f"  Normality: {'✓ Pass' if all_normal_dm else '✗ Fail'}")
print(f"  Equal variance: {'✓ Pass' if equal_var_dm else '✗ Fail'}")
print(f"  → Use: {'ANOVA + Tukey' if use_parametric_dm else 'Kruskal-Wallis + Dunn'}")

print(f"\nClustering:")
print(f"  ICC(Dc): {icc_dc:.3f} - {'⚠ HIGH (need WSI aggregation)' if icc_dc > 0.5 else '✓ Acceptable'}")
print(f"  ICC(Dm): {icc_dm:.3f} - {'⚠ HIGH (need WSI aggregation)' if icc_dm > 0.5 else '✓ Acceptable'}")

# Save assumption check results
assumption_results = pd.DataFrame([{
    'Test': 'Normality (Shapiro-Wilk)',
    'Dc_Result': 'Pass' if all_normal_dc else 'FAIL',
    'Dc_Details': f'All p >= 0.05' if all_normal_dc else f'Some p < 0.05',
    'Dm_Result': 'Pass' if all_normal_dm else 'FAIL',
    'Dm_Details': f'All p >= 0.05' if all_normal_dm else f'Some p < 0.05'
}, {
    'Test': 'Variance Equality (Levene)',
    'Dc_Result': 'Pass' if equal_var_dc else 'FAIL',
    'Dc_Details': f'p = {p_levene_dc:.6f}',
    'Dm_Result': 'Pass' if equal_var_dm else 'FAIL',
    'Dm_Details': f'p = {p_levene_dm:.6f}'
}, {
    'Test': 'ICC (Clustering)',
    'Dc_Result': 'Low' if icc_dc < 0.5 else 'HIGH',
    'Dc_Details': f'ICC = {icc_dc:.3f}',
    'Dm_Result': 'Low' if icc_dm < 0.5 else 'HIGH',
    'Dm_Details': f'ICC = {icc_dm:.3f}'
}, {
    'Test': 'R²(Dc) vs R²(Dm) correlation',
    'Dc_Result': f'r = {r_pearson:.3f}',
    'Dc_Details': f'{100*r_pearson**2:.1f}% shared',
    'Dm_Result': '',
    'Dm_Details': ''
}])

with pd.ExcelWriter(RESULTS_DIR / '00_assumption_checks.xlsx', engine='openpyxl') as writer:
    assumption_results.to_excel(writer, sheet_name='Summary', index=False)
    normality_df.to_excel(writer, sheet_name='Normality_Details', index=False)

print(f"\n✓ Saved: {RESULTS_DIR / '00_assumption_checks.xlsx'}")

# ============================================================================
# ANALYSIS 2: PARAMETRIC vs NON-PARAMETRIC TESTS
# ============================================================================

print("\n" + "=" * 80)
print(f"ANALYSIS 2: BETWEEN-PATHOLOGY COMPARISON ({analysis_level}-Level, Corrected)")
print("=" * 80)

print(f"\n⚠ Using {analysis_level}-level data (n={len(merged_analysis)})")

groups_dc = [merged_analysis[merged_analysis['Pathology'] == p]['R2_corr'].values for p in PATHOLOGY_ORDER]
groups_dm = [merged_analysis[merged_analysis['Pathology'] == p]['R2_mink'].values for p in PATHOLOGY_ORDER]

# ============================================================================
# Dc R²: PARAMETRIC vs NON-PARAMETRIC
# ============================================================================

print("\n" + "-" * 80)
print("Dc R² Analysis:")
print("-" * 80)

if use_parametric_dc:
    print("Using PARAMETRIC tests (assumptions met):")
    
    # ANOVA
    f_dc, p_dc = f_oneway(*groups_dc)
    grand_mean_dc = merged['R2_corr'].mean()
    ss_between_dc = sum(len(g) * (g.mean() - grand_mean_dc)**2 for g in groups_dc)
    ss_total_dc = sum((merged['R2_corr'] - grand_mean_dc)**2)
    eta2_dc = ss_between_dc / ss_total_dc
    
    print(f"  ANOVA: F = {f_dc:.4f}, p = {p_dc:.6f}, η² = {eta2_dc:.6f}")
    print(f"  → {'Significant' if p_dc < ALPHA else 'Not significant'} differences")
    
    test_name_dc = 'ANOVA'
    test_stat_dc = f_dc
    test_p_dc = p_dc
    effect_size_dc = eta2_dc
    effect_name_dc = 'Eta_squared'
    
else:
    print("Using NON-PARAMETRIC tests (assumptions violated):")
    
    # Kruskal-Wallis
    H_dc, p_dc = kruskal(*groups_dc)
    n_dc = sum(len(g) for g in groups_dc)
    k_dc = len(groups_dc)
    epsilon2_dc = (H_dc - k_dc + 1) / (n_dc - k_dc)
    
    print(f"  Kruskal-Wallis: H = {H_dc:.4f}, p = {p_dc:.6f}, ε² = {epsilon2_dc:.6f}")
    print(f"  → {'Significant' if p_dc < ALPHA else 'Not significant'} differences")
    
    test_name_dc = 'Kruskal-Wallis'
    test_stat_dc = H_dc
    test_p_dc = p_dc
    effect_size_dc = epsilon2_dc
    effect_name_dc = 'Epsilon_squared'

# ============================================================================
# Dm R²: PARAMETRIC vs NON-PARAMETRIC
# ============================================================================

print("\n" + "-" * 80)
print("Dm R² Analysis:")
print("-" * 80)

if use_parametric_dm:
    print("Using PARAMETRIC tests (assumptions met):")
    
    # ANOVA
    f_dm, p_dm = f_oneway(*groups_dm)
    grand_mean_dm = merged['R2_mink'].mean()
    ss_between_dm = sum(len(g) * (g.mean() - grand_mean_dm)**2 for g in groups_dm)
    ss_total_dm = sum((merged['R2_mink'] - grand_mean_dm)**2)
    eta2_dm = ss_between_dm / ss_total_dm
    
    print(f"  ANOVA: F = {f_dm:.4f}, p = {p_dm:.6f}, η² = {eta2_dm:.6f}")
    print(f"  → {'Significant' if p_dm < ALPHA else 'Not significant'} differences")
    
    test_name_dm = 'ANOVA'
    test_stat_dm = f_dm
    test_p_dm = p_dm
    effect_size_dm = eta2_dm
    effect_name_dm = 'Eta_squared'
    
else:
    print("Using NON-PARAMETRIC tests (assumptions violated):")
    
    # Kruskal-Wallis
    H_dm, p_dm = kruskal(*groups_dm)
    n_dm = sum(len(g) for g in groups_dm)
    k_dm = len(groups_dm)
    epsilon2_dm = (H_dm - k_dm + 1) / (n_dm - k_dm)
    
    print(f"  Kruskal-Wallis: H = {H_dm:.4f}, p = {p_dm:.6f}, ε² = {epsilon2_dm:.6f}")
    print(f"  → {'Significant' if p_dm < ALPHA else 'Not significant'} differences")
    
    test_name_dm = 'Kruskal-Wallis'
    test_stat_dm = H_dm
    test_p_dm = p_dm
    effect_size_dm = epsilon2_dm
    effect_name_dm = 'Epsilon_squared'

# Save results
comparison_results = pd.DataFrame([
    {'Dimension': 'Dc', 'Test_Used': test_name_dc, 'Test_Statistic': test_stat_dc, 
     'P_value': test_p_dc, 'Effect_Size': effect_size_dc, 'Effect_Size_Type': effect_name_dc},
    {'Dimension': 'Dm', 'Test_Used': test_name_dm, 'Test_Statistic': test_stat_dm, 
     'P_value': test_p_dm, 'Effect_Size': effect_size_dm, 'Effect_Size_Type': effect_name_dm}
])
comparison_results.to_excel(RESULTS_DIR / '02_between_pathology_tests.xlsx', index=False)
print(f"\n✓ Saved: {RESULTS_DIR / '02_between_pathology_tests.xlsx'}")

# ============================================================================
# ANALYSIS 3: POST-HOC COMPARISONS WITH EFFECT SIZES
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 3: POST-HOC COMPARISONS (with Effect Sizes)")
print("=" * 80)

from itertools import combinations

def calculate_hedges_g(group1, group2):
    """Calculate Hedges' g with small sample correction"""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = group1.mean(), group2.mean()
    sd1, sd2 = group1.std(ddof=1), group2.std(ddof=1)
    
    if sd1 > 0 and sd2 > 0:
        pooled_std = np.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        correction = 1 - (3 / (4*(n1 + n2) - 9)) if (n1 + n2) > 3 else 1
        hedges_g = d * correction
    else:
        hedges_g = 0
    return hedges_g

if test_p_dc < ALPHA:
    print("\nDc R² Pairwise Comparisons:")
    
    if use_parametric_dc:
        # Tukey's HSD
        tukey = pairwise_tukeyhsd(merged_analysis['R2_corr'], merged_analysis['Pathology'], alpha=ALPHA)
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        
        # Add effect sizes
        effect_sizes = []
        for p1, p2 in combinations(PATHOLOGY_ORDER, 2):
            g1 = merged_analysis[merged_analysis['Pathology'] == p1]['R2_corr'].values
            g2 = merged_analysis[merged_analysis['Pathology'] == p2]['R2_corr'].values
            hedges_g = calculate_hedges_g(g1, g2)
            effect_sizes.append(hedges_g)
        
        tukey_df['Hedges_g'] = effect_sizes
        tukey_df.to_excel(RESULTS_DIR / '03_posthoc_dc_tukey.xlsx', index=False)
        print(f"  Used Tukey's HSD (parametric)")
        print(f"  ✓ Saved: {RESULTS_DIR / '03_posthoc_dc_tukey.xlsx'}")
    else:
        # Dunn's post-hoc (non-parametric)
        from scikit_posthocs import posthoc_dunn
        dunn_df = posthoc_dunn(merged_analysis, val_col='R2_corr', group_col='Pathology', p_adjust='bonferroni')
        dunn_df.to_excel(RESULTS_DIR / '03_posthoc_dc_dunn.xlsx')
        print(f"  Used Dunn's test (non-parametric, Bonferroni correction)")
        print(f"  ✓ Saved: {RESULTS_DIR / '03_posthoc_dc_dunn.xlsx'}")

# ============================================================================
# ANALYSIS 4: TREND ANALYSIS (Progression)
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 4: TREND ACROSS PATHOLOGICAL PROGRESSION")
print("=" * 80)

# Encode progression as ordinal
progression_map = {p: i for i, p in enumerate(PATHOLOGY_ORDER)}
merged_analysis['Progression_Order'] = merged_analysis['Pathology'].map(progression_map)

# Spearman correlation (monotonic trend)
rho_dc, p_rho_dc = spearmanr(merged_analysis['Progression_Order'], merged_analysis['R2_corr'])
rho_dm, p_rho_dm = spearmanr(merged_analysis['Progression_Order'], merged_analysis['R2_mink'])

print(f"\nDc R²: Spearman ρ = {rho_dc:.4f}, p = {p_rho_dc:.6f}")
if p_rho_dc >= ALPHA:
    print(f"  → NOT significant (p = {p_rho_dc:.4f} ≥ {ALPHA}) → No trend")
elif rho_dc > 0:
    magnitude = 'strong' if abs(rho_dc) > 0.5 else 'moderate' if abs(rho_dc) > 0.3 else 'weak'
    print(f"  → SIGNIFICANT POSITIVE trend (p < {ALPHA}): R² increases with progression ({magnitude}, ρ = {rho_dc:.3f})")
else:
    magnitude = 'strong' if abs(rho_dc) > 0.5 else 'moderate' if abs(rho_dc) > 0.3 else 'weak'
    print(f"  → SIGNIFICANT NEGATIVE trend (p < {ALPHA}): R² decreases with progression ({magnitude}, ρ = {rho_dc:.3f})")

print(f"\nDm R²: Spearman ρ = {rho_dm:.4f}, p = {p_rho_dm:.6f}")
if p_rho_dm >= ALPHA:
    print(f"  → NOT significant (p = {p_rho_dm:.4f} ≥ {ALPHA}) → No trend")
elif rho_dm > 0:
    magnitude = 'strong' if abs(rho_dm) > 0.5 else 'moderate' if abs(rho_dm) > 0.3 else 'weak'
    print(f"  → SIGNIFICANT POSITIVE trend (p < {ALPHA}): R² increases with progression ({magnitude}, ρ = {rho_dm:.3f})")
else:
    magnitude = 'strong' if abs(rho_dm) > 0.5 else 'moderate' if abs(rho_dm) > 0.3 else 'weak'
    print(f"  → SIGNIFICANT NEGATIVE trend (p < {ALPHA}): R² decreases with progression ({magnitude}, ρ = {rho_dm:.3f})")

def trend_label(rho, p, alpha=ALPHA):
    if p >= alpha:
        return 'No trend (not significant)'
    direction = 'Positive' if rho > 0 else 'Negative'
    magnitude = 'strong' if abs(rho) > 0.5 else 'moderate' if abs(rho) > 0.3 else 'weak'
    return f'{direction} ({magnitude})'

trend_results = pd.DataFrame([
    {'Dimension': 'Dc', 'Spearman_rho': rho_dc, 'P_value': p_rho_dc,
     'Significant': p_rho_dc < ALPHA,
     'Interpretation': trend_label(rho_dc, p_rho_dc)},
    {'Dimension': 'Dm', 'Spearman_rho': rho_dm, 'P_value': p_rho_dm,
     'Significant': p_rho_dm < ALPHA,
     'Interpretation': trend_label(rho_dm, p_rho_dm)}
])
trend_results.to_excel(RESULTS_DIR / '04_trend_analysis.xlsx', index=False)
print(f"\n✓ Saved: {RESULTS_DIR / '04_trend_analysis.xlsx'}")

# ============================================================================
# ANALYSIS 5: CORRELATION - R² vs DIMENSION VALUE
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 4: CORRELATION - R² vs DIMENSION VALUE")
print("=" * 80)

r_dc, p_r_dc = pearsonr(merged_analysis['Dc'], merged_analysis['R2_corr'])
r_dm, p_r_dm = pearsonr(merged_analysis['Dm'], merged_analysis['R2_mink'])

print(f"\nDc vs R²(Dc): r = {r_dc:.4f}, p = {p_r_dc:.6f}")
print(f"Dm vs R²(Dm): r = {r_dm:.4f}, p = {p_r_dm:.6f}")

correlation_results = pd.DataFrame([
    {'Variables': 'Dc vs R2_corr', 'Pearson_r': r_dc, 'P_value': p_r_dc},
    {'Variables': 'Dm vs R2_mink', 'Pearson_r': r_dm, 'P_value': p_r_dm}
])
correlation_results.to_excel(RESULTS_DIR / '04_correlation_results.xlsx', index=False)
print(f"\n✓ Saved: {RESULTS_DIR / '04_correlation_results.xlsx'}")

# ============================================================================
# ANALYSIS 5: POOR FIT ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 5: POOR FIT ANALYSIS (R² < 0.95)")
print("=" * 80)

poor_dc = merged_analysis[merged_analysis['R2_corr'] < R2_GOOD]
poor_dm = merged_analysis[merged_analysis['R2_mink'] < R2_GOOD]

print(f"\nPoor Dc fits ({analysis_level}-level): {len(poor_dc)} ({100*len(poor_dc)/len(merged_analysis):.2f}%)")
if len(poor_dc) > 0:
    print(f"  By pathology: {dict(poor_dc['Pathology'].value_counts())}")
    print(f"  Mean Dc: {poor_dc['Dc'].mean():.4f} (vs {merged['Dc'].mean():.4f} overall)")

print(f"\nPoor Dm fits: {len(poor_dm)} ({100*len(poor_dm)/len(merged_analysis):.2f}%)")
if len(poor_dm) > 0:
    print(f"  By pathology: {dict(poor_dm['Pathology'].value_counts())}")

poor_fits = pd.DataFrame({
    'Dimension': ['Dc', 'Dm'],
    'N_poor_fits': [len(poor_dc), len(poor_dm)],
    'Percent': [100*len(poor_dc)/len(merged_analysis), 100*len(poor_dm)/len(merged_analysis)]
})
poor_fits.to_excel(RESULTS_DIR / '05_poor_fits_summary.xlsx', index=False)
print(f"\n✓ Saved: {RESULTS_DIR / '05_poor_fits_summary.xlsx'}")

# ============================================================================
# VISUALIZATION 1: R² DISTRIBUTION BY PATHOLOGY
# ============================================================================

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(24, 24))
fig.suptitle('RQ6: R² Distribution by Pathology', fontsize=14, fontweight='bold')

# Plot 1: Boxplot Dc R²
ax = axes[0, 0]
data_dc = [merged_analysis[merged_analysis['Pathology'] == p]['R2_corr'].values for p in PATHOLOGY_ORDER]
bp = ax.boxplot(data_dc, labels=PATHOLOGY_ORDER, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.axhline(R2_EXCELLENT, color='green', linestyle='--', label=f'Excellent ({R2_EXCELLENT})')
ax.axhline(R2_GOOD, color='orange', linestyle='--', label=f'Good ({R2_GOOD})')
ax.set_ylabel('R² (Correlation Dimension)')
ax.set_title(f'Dc R²: {test_name_dc} stat={test_stat_dc:.2f}, p={test_p_dc:.4f}')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Boxplot Dm R²
ax = axes[0, 1]
data_dm = [merged_analysis[merged_analysis['Pathology'] == p]['R2_mink'].values for p in PATHOLOGY_ORDER]
bp = ax.boxplot(data_dm, labels=PATHOLOGY_ORDER, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('deepskyblue')
ax.axhline(R2_EXCELLENT, color='green', linestyle='--')
ax.axhline(R2_GOOD, color='orange', linestyle='--')
ax.set_ylabel('R² (Minkowski Dimension)')
ax.set_title(f'Dm R²: {test_name_dm} stat={test_stat_dm:.2f}, p={test_p_dm:.4f}')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Scatter Dc vs R²
ax = axes[1, 0]
ax.scatter(merged_analysis['Dc'], merged_analysis['R2_corr'], alpha=0.3, s=10)
ax.axhline(R2_GOOD, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Dc (Correlation Dimension)')
ax.set_ylabel('R²')
ax.set_title(f'Dc vs R²: r={r_dc:.3f}, p={p_r_dc:.4f}')
ax.grid(True, alpha=0.3)

# Plot 4: Scatter Dm vs R²
ax = axes[1, 1]
ax.scatter(merged_analysis['Dm'], merged_analysis['R2_mink'], alpha=0.3, s=10)
ax.axhline(R2_GOOD, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Dm (Minkowski Dimension)')
ax.set_ylabel('R²')
ax.set_title(f'Dm vs R²: r={r_dm:.3f}, p={p_r_dm:.4f}')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig1_r2_by_pathology.tif', format='tif', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOTS_DIR / 'fig1_r2_by_pathology.tif'}")

# Export data
merged[['Pathology', 'Dc', 'Dm', 'R2_corr', 'R2_mink']].to_excel(
    ORIGIN_DATA_DIR / 'fig1_r2_data.xlsx', index=False)

# ============================================================================
# VISUALIZATION 2: HISTOGRAM OF R² VALUES
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('RQ6: Overall R² Distribution', fontsize=14, fontweight='bold')

ax = axes[0]
ax.hist(merged['R2_corr'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(R2_EXCELLENT, color='green', linestyle='--', linewidth=2, label='Excellent')
ax.axvline(R2_GOOD, color='orange', linestyle='--', linewidth=2, label='Good')
ax.axvline(merged['R2_corr'].mean(), color='red', linestyle='-', linewidth=2, 
          label=f'Mean={merged["R2_corr"].mean():.4f}')
ax.set_xlabel('R² (Correlation)')
ax.set_ylabel('Frequency')
ax.set_title('Correlation Dimension R² Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(merged['R2_mink'], bins=50, color='coral', alpha=0.7, edgecolor='black')
ax.axvline(R2_EXCELLENT, color='green', linestyle='--', linewidth=2)
ax.axvline(R2_GOOD, color='orange', linestyle='--', linewidth=2)
ax.axvline(merged['R2_mink'].mean(), color='red', linestyle='-', linewidth=2,
          label=f'Mean={merged["R2_mink"].mean():.4f}')
ax.set_xlabel('R² (Minkowski)')
ax.set_ylabel('Frequency')
ax.set_title('Minkowski Dimension R² Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig2_r2_histogram.tif', format='tif', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOTS_DIR / 'fig2_r2_histogram.tif'}")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING SUMMARY REPORT")
print("=" * 80)

summary_report = f"""
================================================================================
RESEARCH QUESTION 6 (RQ6): SCALING QUALITY vs PATHOLOGY - SUMMARY
================================================================================

Research Question:
"What is the relationship between R² goodness-of-fit and pathological diagnosis?"

ANSWER: {'YES' if p_dc < ALPHA or p_dm < ALPHA else 'NO'} - R² {'DOES' if p_dc < ALPHA or p_dm < ALPHA else 'does NOT'} vary significantly by pathology

================================================================================
KEY FINDINGS
================================================================================

1. OVERALL R² QUALITY
   --------------------------------------------------------
   Correlation Dimension (Dc):
   - Mean R²: {merged['R2_corr'].mean():.6f}
   - Median R²: {merged['R2_corr'].median():.6f}
   - Range: [{merged['R2_corr'].min():.6f}, {merged['R2_corr'].max():.6f}]
   - Excellent (>0.99): {100*len(merged[merged['R2_corr'] > R2_EXCELLENT])/len(merged):.1f}%
   - Good (>0.95): {100*len(merged[merged['R2_corr'] > R2_GOOD])/len(merged):.1f}%
   
   Minkowski Dimension (Dm):
   - Mean R²: {merged['R2_mink'].mean():.6f}
   - Median R²: {merged['R2_mink'].median():.6f}
   - Range: [{merged['R2_mink'].min():.6f}, {merged['R2_mink'].max():.6f}]
   - Excellent (>0.99): {100*len(merged[merged['R2_mink'] > R2_EXCELLENT])/len(merged):.1f}%
   - Good (>0.95): {100*len(merged[merged['R2_mink'] > R2_GOOD])/len(merged):.1f}%

2. STATISTICAL TEST RESULTS (R² BY PATHOLOGY)
   --------------------------------------------------------
   Correlation Dimension:
   - Test used: {test_name_dc}
   - Test statistic: {test_stat_dc:.4f}
   - P-value: {test_p_dc:.6f}
   - Effect size ({effect_name_dc}): {effect_size_dc:.6f}
   - Interpretation: {'SIGNIFICANT' if test_p_dc < ALPHA else 'NOT SIGNIFICANT'} differences
   
   Minkowski Dimension:
   - Test used: {test_name_dm}
   - Test statistic: {test_stat_dm:.4f}
   - P-value: {test_p_dm:.6f}
   - Effect size ({effect_name_dm}): {effect_size_dm:.6f}
   - Interpretation: {'SIGNIFICANT' if test_p_dm < ALPHA else 'NOT SIGNIFICANT'} differences

3. ASSUMPTION CHECKS
   --------------------------------------------------------
   Normality (Shapiro-Wilk):
   - Dc: {'PASSED' if all_normal_dc else 'VIOLATED'} (all groups p >= 0.05: {'Yes' if all_normal_dc else 'No'})
   - Dm: {'PASSED' if all_normal_dm else 'VIOLATED'} (all groups p >= 0.05: {'Yes' if all_normal_dm else 'No'})
   
   Variance Equality (Levene):
   - Dc: {'PASSED' if equal_var_dc else 'VIOLATED'} (p = {p_levene_dc:.6f})
   - Dm: {'PASSED' if equal_var_dm else 'VIOLATED'} (p = {p_levene_dm:.6f})
   
   Clustering (ICC):
   - Dc R²: ICC = {icc_dc:.3f} ({'HIGH' if icc_dc > 0.5 else 'MODERATE' if icc_dc > 0.2 else 'LOW'})
   - Dm R²: ICC = {icc_dm:.3f} ({'HIGH' if icc_dm > 0.5 else 'MODERATE' if icc_dm > 0.2 else 'LOW'})
   {'  ⚠ WARNING: High clustering detected - consider WSI-level aggregation' if icc_dc > 0.5 or icc_dm > 0.5 else ''}

4. TREND ANALYSIS (Progression)
   --------------------------------------------------------
   Dc R²: Spearman ρ = {rho_dc:.4f}, p = {p_rho_dc:.6f}
   → {'NOT significant → No trend' if p_rho_dc >= ALPHA else ('POSITIVE trend (weak)' if rho_dc > 0 and abs(rho_dc) <= 0.3 else 'POSITIVE trend (moderate/strong)' if rho_dc > 0 else 'NEGATIVE trend (weak)' if abs(rho_dc) <= 0.3 else 'NEGATIVE trend (moderate/strong)')} across progression
   
   Dm R²: Spearman ρ = {rho_dm:.4f}, p = {p_rho_dm:.6f}
   → {'NOT significant → No trend' if p_rho_dm >= ALPHA else ('POSITIVE trend (weak)' if rho_dm > 0 and abs(rho_dm) <= 0.3 else 'POSITIVE trend (moderate/strong)' if rho_dm > 0 else 'NEGATIVE trend (weak)' if abs(rho_dm) <= 0.3 else 'NEGATIVE trend (moderate/strong)')} across progression

5. CORRELATION: R² vs DIMENSION VALUE
   --------------------------------------------------------
   Dc vs R²(Dc): r = {r_dc:.4f}, p = {p_r_dc:.6f}
   → {'Significant' if p_r_dc < ALPHA else 'Not significant'} {'positive' if r_dc > 0 else 'negative'} correlation
   
   Dm vs R²(Dm): r = {r_dm:.4f}, p = {p_r_dm:.6f}
   → {'Significant' if p_r_dm < ALPHA else 'Not significant'} correlation
   
   R²(Dc) vs R²(Dm): r = {r_pearson:.4f}, p = {p_pearson:.6f}
   → {'Strong' if abs(r_pearson) > 0.7 else 'Moderate' if abs(r_pearson) > 0.4 else 'Weak'} correlation

6. POOR FIT ANALYSIS (R² < 0.95)
   --------------------------------------------------------
   Correlation Dimension:
   - Number of poor fits: {len(poor_dc)} ({100*len(poor_dc)/len(merged_analysis):.2f}%)
   - Pathologies affected: {list(poor_dc['Pathology'].unique()) if len(poor_dc) > 0 else 'None'}
   
   Minkowski Dimension:
   - Number of poor fits: {len(poor_dm)} ({100*len(poor_dm)/len(merged_analysis):.2f}%)

================================================================================
INTERPRETATION
================================================================================

OVERALL SCALING QUALITY:
Both dimensions show {'excellent' if merged['R2_corr'].mean() > 0.99 and merged['R2_mink'].mean() > 0.95 else 'good'} fractal scaling across most ROIs.
{'Over 95% of ROIs show excellent fits, confirming strong fractal behavior.' if len(merged[merged['R2_corr'] > R2_EXCELLENT])/len(merged) > 0.95 else 'Most ROIs show good scaling quality.'}

STATISTICAL APPROACH:
Used {test_name_dc} for Dc (assumptions {'met' if use_parametric_dc else 'VIOLATED'})
Used {test_name_dm} for Dm (assumptions {'met' if use_parametric_dm else 'VIOLATED'})
{'⚠ Non-parametric tests used due to normality/variance violations' if not use_parametric_dc or not use_parametric_dm else '✓ Parametric tests appropriate'}

PATHOLOGY EFFECT:
{'R² varies significantly by pathology type.' if test_p_dc < ALPHA or test_p_dm < ALPHA else 'R² does not vary significantly by pathology type.'}
{f'Effect size: {effect_name_dc} = {effect_size_dc:.4f} ({"Large" if effect_size_dc > 0.14 else "Medium" if effect_size_dc > 0.06 else "Small"})' if test_p_dc < ALPHA else 'Effect size: negligible'}

CLUSTERING:
{f'⚠ ICC = {icc_dc:.3f} for Dc, {icc_dm:.3f} for Dm' if icc_dc > 0.5 or icc_dm > 0.5 else f'✓ Low clustering (ICC = {icc_dc:.3f}, {icc_dm:.3f})'}
{'→ R² values clustered within WSIs - ROI-level analysis may be pseudoreplication' if icc_dc > 0.5 or icc_dm > 0.5 else '→ ROI-level analysis statistically valid'}

POOR FITS:
{f'{len(poor_dc)} ROIs show poor Dc fits, suggesting non-fractal or multi-fractal behavior.' if len(poor_dc) > 0 else 'Virtually all ROIs show excellent fractal scaling.'}
{'Poor fits concentrated in specific pathologies.' if len(poor_dc) > 0 and len(poor_dc['Pathology'].unique()) < 4 else ''}

================================================================================
CLINICAL/METHODOLOGICAL IMPLICATIONS
================================================================================

VALIDATION OF FRACTAL ANALYSIS:
{'✓ Excellent R² across pathologies validates fractal approach' if merged['R2_corr'].mean() > 0.99 else '⚠ Some ROIs show questionable fractal behavior'}
{'✓ Nuclear distributions truly exhibit scale-invariant properties' if len(poor_dc) < len(merged_analysis)*0.05 else '⚠ Consider excluding poor-fit ROIs'}

STATISTICAL VALIDITY:
{f'⚠ High ICC ({max(icc_dc, icc_dm):.3f}) suggests WSI-level aggregation needed for valid inference' if max(icc_dc, icc_dm) > 0.5 else '✓ ROI-level analysis statistically appropriate'}
{f'✓ Used appropriate tests ({test_name_dc}) after checking assumptions' if not use_parametric_dc else '✓ Parametric assumptions met'}

QUALITY CONTROL:
{f'→ Recommend excluding {len(poor_dc)} ROIs with R² < {R2_GOOD}' if len(poor_dc) > 0 and len(poor_dc)/len(merged_analysis) < 0.1 else '→ No quality control exclusions needed'}
→ Report R² as quality metric in publications
→ Use R² > {R2_GOOD} as inclusion criterion

BIOLOGICAL INTERPRETATION:
{'→ Different pathologies show different fractal scaling quality' if test_p_dc < ALPHA and effect_size_dc > 0.06 else '→ Scaling quality consistent across disease progression'}
{'→ Consider multifractal analysis for poor-fit ROIs' if len(poor_dc) > len(merged_analysis)*0.05 else '→ Monofractal analysis appropriate'}

================================================================================
FILES GENERATED
================================================================================

RESULTS (Excel):
- 01_r2_by_pathology.xlsx - R² statistics by pathology
- 02_anova_results.xlsx - ANOVA F-tests
- 03_posthoc_tukey.xlsx - Pairwise comparisons (if significant)
- 04_correlation_results.xlsx - R² vs dimension correlations
- 05_poor_fits_summary.xlsx - Poor fit analysis

PLOTS (TIF, 300 DPI):
- fig1_r2_by_pathology.tif - R² distributions and correlations (4-panel)
- fig2_r2_histogram.tif - Overall R² histograms (2-panel)

ORIGIN DATA:
- fig1_r2_data.xlsx - Raw data for plotting

================================================================================
CONCLUSION
================================================================================

R² (scaling quality) {'varies significantly' if p_dc < ALPHA or p_dm < ALPHA else 'does not vary significantly'} by pathology type.

KEY TAKEAWAY:
{f'Fractal analysis is highly appropriate for this dataset - {100*len(merged[merged["R2_corr"] > R2_GOOD])/len(merged_analysis):.1f}% of ROIs show good-to-excellent scaling.' if len(merged[merged['R2_corr'] > R2_GOOD])/len(merged_analysis) > 0.95 else 'Most ROIs show acceptable scaling, validating the fractal approach.'}

{'RECOMMENDATION: Some pathologies show poorer scaling - investigate biological reasons or consider multifractal methods.' if test_p_dc < ALPHA and effect_size_dc > 0.06 else 'RECOMMENDATION: Scaling quality is consistently high - monofractal analysis appropriate.'}

================================================================================
"""

with open(RESULTS_DIR / '00_RQ6_SUMMARY.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(summary_report)
print(f"\n✓ Saved: {RESULTS_DIR / '00_RQ6_SUMMARY.txt'}")

# ============================================================================
# COMPLETION
# ============================================================================

print("\n" + "=" * 80)
print("RQ6 ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nAll results saved to: {OUTPUT_DIR}")
print(f"Generated files:")
print(f"  - Plots: {len(list(PLOTS_DIR.glob('*.tif')))} TIF figures")
print(f"  - Results: {len(list(RESULTS_DIR.glob('*.xlsx')))} Excel files")
print(f"  - Origin data: {len(list(ORIGIN_DATA_DIR.glob('*.xlsx')))} files")
print("=" * 80)