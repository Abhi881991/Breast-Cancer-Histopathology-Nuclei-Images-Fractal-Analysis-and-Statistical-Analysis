"""
================================================================================
RESEARCH QUESTION 3 (RQ3): PATHOLOGY DISCRIMINATION ANALYSIS
================================================================================

Research Question:
"Do different pathological tissue subtypes exhibit statistically distinct 
fractal characteristics of nuclear spatial distribution?"

Theoretical Background:
-----------------------

PATHOLOGY PROGRESSION IN BREAST TISSUE (BRACS Dataset):
    The BRACS classification system represents a continuum from normal to 
    invasive carcinoma:
    
    N → PB → UDH → FEA → ADH → DCIS → IC
    
    Where:
    - N    : Normal tissue (baseline)
    - PB   : Pathological Benign (mild abnormality)
    - UDH  : Usual Ductal Hyperplasia (proliferation without atypia)
    - FEA  : Flat Epithelial Atypia (low-grade atypia)
    - ADH  : Atypical Ductal Hyperplasia (precursor lesion)
    - DCIS : Ductal Carcinoma In Situ (non-invasive malignancy)
    - IC   : Invasive Carcinoma (invasive malignancy)

HYPOTHESIS:
    As tissue progresses from normal to malignant, nuclear spatial organization
    becomes increasingly disordered and complex, reflected in fractal dimension
    changes.
    
    Biological basis:
    - Normal tissue: Organized, regular nuclear distribution
    - Hyperplasia: Increased density but maintained organization
    - Atypia: Loss of polarity, irregular spacing
    - Carcinoma: Chaotic distribution, loss of tissue architecture
    
    Fractal dimension implications:
    - Higher Dc → More space-filling, denser nuclear packing
    - Higher Dm → More complex nuclear boundaries, irregular shapes
    - Changes may not be linear across progression spectrum

STATISTICAL TESTS USED:
-----------------------

1. ONE-WAY ANOVA (Analysis of Variance)
   Purpose: Test if mean fractal dimensions differ across pathologies
   Null hypothesis: μ_N = μ_PB = μ_UDH = μ_FEA = μ_ADH = μ_DCIS = μ_IC
   Alternative: At least one mean differs
   
   F-statistic: F = MS_between / MS_within
   
   Assumptions:
   - Independence of observations (satisfied: different ROIs)
   - Normality within groups (tested via Shapiro-Wilk)
   - Homogeneity of variance (tested via Levene's test)
   
   Effect size: Eta-squared (η²)
   η² = SS_between / SS_total
   Interpretation:
   - η² < 0.01: Small effect
   - η² < 0.06: Medium effect
   - η² < 0.14: Large effect
   - η² ≥ 0.14: Very large effect

2. KRUSKAL-WALLIS H TEST (Non-parametric Alternative)
   Purpose: Test group differences when normality violated
   Null hypothesis: All groups have same distribution
   
   H-statistic: H = (12/[N(N+1)]) × Σ(R_i²/n_i) - 3(N+1)
   
   Use when:
   - Normality assumption violated
   - Outliers present
   - Ordinal data
   
   Effect size: Epsilon-squared (ε²)
   ε² = H / (N² - 1) / (N + 1)

3. POST-HOC PAIRWISE COMPARISONS
   Purpose: Identify which specific pairs differ
   
   Methods used:
   a) Tukey's HSD (Honest Significant Difference)
      - Controls familywise error rate
      - Assumes equal variances
      - Best for balanced designs
   
   b) Games-Howell test
      - Does not assume equal variances
      - Robust to unequal sample sizes
      - More conservative
   
   c) Dunn's test (for Kruskal-Wallis)
      - Non-parametric post-hoc
      - Uses rank sums
   
   Multiple testing correction:
   - Bonferroni: α_adjusted = α / k (k = number of comparisons)
   - Most conservative
   - Controls Type I error

4. EFFECT SIZES (COHEN'S d FOR PAIRWISE COMPARISONS)
   Purpose: Quantify magnitude of differences
   
   Formula: d = (μ₁ - μ₂) / σ_pooled
   
   where: σ_pooled = √[(σ₁² + σ₂²) / 2]
   
   Interpretation:
   - |d| < 0.2: Negligible
   - 0.2 ≤ |d| < 0.5: Small
   - 0.5 ≤ |d| < 0.8: Medium
   - |d| ≥ 0.8: Large

5. LINEAR DISCRIMINANT ANALYSIS (LDA)
   Purpose: Find linear combinations that best separate groups
   
   Objective: Maximize between-class / within-class variance
   
   Discriminant functions: Y = w₁X₁ + w₂X₂ + ... + constant
   
   Outputs:
   - Canonical discriminant functions
   - Classification accuracy
   - Confusion matrix
   
   Use: Determines if fractal dimensions can classify pathologies

6. MULTINOMIAL LOGISTIC REGRESSION
   Purpose: Model probability of pathology given fractal dimensions
   
   Model: P(Y=k|X) = exp(β_k·X) / Σ exp(β_j·X)
   
   Outputs:
   - Odds ratios for each dimension
   - Confidence intervals
   - Overall model fit (AIC, BIC)
   
   Interpretation: How fractal dimensions relate to pathology odds

7. RANDOM FOREST CLASSIFICATION
   Purpose: Non-linear classification and feature importance
   
   Advantages:
   - No assumptions about distributions
   - Handles complex interactions
   - Feature importance ranking
   - Robust to outliers
   
   Metrics:
   - Overall accuracy
   - Class-specific precision/recall
   - F1-scores
   - ROC-AUC (one-vs-rest)

8. TREND ANALYSIS (POLYNOMIAL CONTRASTS)
   Purpose: Test for linear/quadratic trends across progression
   
   Linear trend: Tests if means increase/decrease monotonically
   Quadratic trend: Tests for U-shaped or inverted-U patterns
   
   Orthogonal contrasts ensure independence of trend components
   
   Use: Determines if fractal dimension follows progression sequence

9. HOMOGENEITY TESTS
   a) Levene's Test: Tests equality of variances
      H₀: σ₁² = σ₂² = ... = σ_k²
      
   b) Bartlett's Test: More powerful but sensitive to normality
      Use when normality satisfied
   
   Purpose: Validate ANOVA assumptions

10. NORMALITY TESTS
    a) Shapiro-Wilk: Tests if data normally distributed
       H₀: Data comes from normal distribution
       
    b) Kolmogorov-Smirnov: Alternative normality test
    
    c) Anderson-Darling: Emphasizes tails
    
    Purpose: Determine if parametric tests appropriate

11. CONFUSION MATRIX ANALYSIS
    Metrics:
    - Accuracy: (TP + TN) / Total
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1-score: 2 × (Precision × Recall) / (Precision + Recall)
    
    Interpretation: Classification performance by pathology

12. RECEIVER OPERATING CHARACTERISTIC (ROC) ANALYSIS
    Purpose: Evaluate binary classification performance
    
    For multiclass: One-vs-Rest approach
    Each class vs all others
    
    AUC interpretation:
    - 0.9-1.0: Excellent
    - 0.8-0.9: Good
    - 0.7-0.8: Fair
    - 0.6-0.7: Poor
    - 0.5-0.6: Fail

PARAMETERS CHOSEN:
------------------
- Significance level: α = 0.05
- Multiple comparison correction: Bonferroni
- Effect size thresholds: Cohen (1988) guidelines
- Random Forest: 1000 trees, max depth = 15
- Cross-validation: 5-fold stratified
- Bootstrap samples: 10,000 for confidence intervals
- LDA components: min(n_features, n_classes-1)
- ROC curve points: 100 thresholds

EXPECTED OUTCOMES:
------------------
If RQ3 is TRUE (pathologies have distinct fractal characteristics):
1. ANOVA F-statistic significant (p < 0.05) ✓
2. Large effect size (η² > 0.14) ✓
3. Multiple pairwise comparisons significant ✓
4. High classification accuracy (>70%) ✓
5. Clear separation in LDA space ✓
6. ROC-AUC > 0.8 for most pathologies ✓
7. Significant linear/quadratic trends ✓

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (f_oneway, kruskal, shapiro, levene, bartlett,
                         ttest_ind, mannwhitneyu, chi2_contingency)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_curve, auc, roc_auc_score, accuracy_score,
                            precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import combinations
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot parameters
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.family'] = 'Times New Roman'  # Available font
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 16
# ============================================================================
# CONFIGURATION
# ============================================================================

# Input file paths
BASE_PATH = Path(r"C:\Users\ajd44\Desktop")
CORR_FILE = BASE_PATH / "Correlation Dimension.csv"
MINK_FILE = BASE_PATH / "Minkowski Dimension.csv"

# Output directory
OUTPUT_DIR = BASE_PATH / 'RQ3_Pathology_Discrimination_Analysis'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Create subdirectories
PLOTS_DIR = OUTPUT_DIR / 'plots'
RESULTS_DIR = OUTPUT_DIR / 'results'
ORIGIN_DATA_DIR = OUTPUT_DIR / 'origin_data'
PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
ORIGIN_DATA_DIR.mkdir(exist_ok=True)

# Analysis parameters
ALPHA = 0.05
N_BOOTSTRAP = 10000
RF_N_ESTIMATORS = 1000
RF_MAX_DEPTH = 15
N_FOLDS = 5
RANDOM_STATE = 42

# Pathology order (progression sequence)
PATHOLOGY_ORDER = ['N', 'PB', 'UDH', 'FEA', 'ADH', 'DCIS', 'IC']

# Effect size thresholds (Cohen, 1988)
COHEN_SMALL = 0.2
COHEN_MEDIUM = 0.5
COHEN_LARGE = 0.8

print("=" * 80)
print("RQ3: PATHOLOGY DISCRIMINATION USING FRACTAL DIMENSIONS")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Significance level: α = {ALPHA}")
print(f"Pathology progression: {' → '.join(PATHOLOGY_ORDER)}")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: DATA LOADING AND PREPARATION")
print("=" * 80)

# Load data
corr_df = pd.read_csv(CORR_FILE)
mink_df = pd.read_csv(MINK_FILE)

# Extract metadata
corr_df['WSI_ID'] = corr_df['File name'].str.extract(r'(BRACS_\d+)')
corr_df['Pathology'] = corr_df['File name'].str.extract(r'_([A-Z]+)_\d+\.tif')

mink_df['WSI_ID'] = mink_df['File name'].str.extract(r'(BRACS_\d+)')
mink_df['Pathology'] = mink_df['File name'].str.extract(r'_([A-Z]+)_\d+\.tif')

# Merge datasets
merged = pd.merge(
    corr_df[['File name', 'Dc', 'R2', 'StdErr', 'WSI_ID', 'Pathology']],
    mink_df[['File name', 'Dm', 'R2', 'StdErr']],
    on='File name',
    suffixes=('_corr', '_mink')
)

# Filter to only include pathologies in the progression sequence
merged = merged[merged['Pathology'].isin(PATHOLOGY_ORDER)].copy()

# Create numeric progression variable (0=N, 1=PB, ..., 6=IC)
merged['Progression'] = merged['Pathology'].map({p: i for i, p in enumerate(PATHOLOGY_ORDER)})

print(f"\nTotal ROIs: {len(merged)}")
print(f"\nSample sizes by pathology:")
pathology_counts = merged['Pathology'].value_counts()[PATHOLOGY_ORDER]
for pathology, count in pathology_counts.items():
    print(f"  {pathology:5s}: {count:4d} ROIs")

# Summary statistics by pathology
print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS BY PATHOLOGY")
print("=" * 80)

summary_stats = merged.groupby('Pathology').agg({
    'Dc': ['mean', 'std', 'median', 'min', 'max'],
    'Dm': ['mean', 'std', 'median', 'min', 'max']
}).round(4)

print("\nCorrelation Dimension (Dc):")
print(summary_stats['Dc'])
print("\nMinkowski Dimension (Dm):")
print(summary_stats['Dm'])

# Save summary statistics
summary_stats.to_excel(RESULTS_DIR / '01_descriptive_statistics.xlsx')
print(f"\n✓ Saved: {RESULTS_DIR / '01_descriptive_statistics.xlsx'}")

# ============================================================================
# ANALYSIS 1: ASSUMPTION TESTING
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 1: TESTING ANOVA ASSUMPTIONS")
print("=" * 80)

assumptions_results = []

# Test 1: Normality (Shapiro-Wilk for each group)
print("\n1. NORMALITY TESTS (Shapiro-Wilk)")
print("-" * 40)

for pathology in PATHOLOGY_ORDER:
    subset = merged[merged['Pathology'] == pathology]
    
    if len(subset) >= 3:
        # Test Dc
        stat_dc, p_dc = shapiro(subset['Dc'])
        # Test Dm
        stat_dm, p_dm = shapiro(subset['Dm'])
        
        assumptions_results.append({
            'Pathology': pathology,
            'Test': 'Shapiro-Wilk (Dc)',
            'Statistic': stat_dc,
            'P_value': p_dc,
            'Assumption_Met': p_dc > ALPHA,
            'Interpretation': 'Normal' if p_dc > ALPHA else 'Non-normal'
        })
        
        assumptions_results.append({
            'Pathology': pathology,
            'Test': 'Shapiro-Wilk (Dm)',
            'Statistic': stat_dm,
            'P_value': p_dm,
            'Assumption_Met': p_dm > ALPHA,
            'Interpretation': 'Normal' if p_dm > ALPHA else 'Non-normal'
        })
        
        print(f"{pathology}: Dc p={p_dc:.4f} {'✓' if p_dc > ALPHA else '✗'}, "
              f"Dm p={p_dm:.4f} {'✓' if p_dm > ALPHA else '✗'}")

# Test 2: Homogeneity of variance (Levene's test)
print("\n2. HOMOGENEITY OF VARIANCE (Levene's Test)")
print("-" * 40)

groups_dc = [merged[merged['Pathology'] == p]['Dc'].values for p in PATHOLOGY_ORDER]
groups_dm = [merged[merged['Pathology'] == p]['Dm'].values for p in PATHOLOGY_ORDER]

levene_dc_stat, levene_dc_p = levene(*groups_dc)
levene_dm_stat, levene_dm_p = levene(*groups_dm)

print(f"Dc: Levene statistic = {levene_dc_stat:.4f}, p = {levene_dc_p:.4f} "
      f"{'✓' if levene_dc_p > ALPHA else '✗'}")
print(f"Dm: Levene statistic = {levene_dm_stat:.4f}, p = {levene_dm_p:.4f} "
      f"{'✓' if levene_dm_p > ALPHA else '✗'}")

assumptions_results.append({
    'Pathology': 'All',
    'Test': "Levene (Dc)",
    'Statistic': levene_dc_stat,
    'P_value': levene_dc_p,
    'Assumption_Met': levene_dc_p > ALPHA,
    'Interpretation': 'Homogeneous' if levene_dc_p > ALPHA else 'Heterogeneous'
})

assumptions_results.append({
    'Pathology': 'All',
    'Test': "Levene (Dm)",
    'Statistic': levene_dm_stat,
    'P_value': levene_dm_p,
    'Assumption_Met': levene_dm_p > ALPHA,
    'Interpretation': 'Homogeneous' if levene_dm_p > ALPHA else 'Heterogeneous'
})

# Summary
assumptions_df = pd.DataFrame(assumptions_results)
assumptions_df.to_excel(RESULTS_DIR / '02_assumption_tests.xlsx', index=False)
print(f"\n✓ Saved: {RESULTS_DIR / '02_assumption_tests.xlsx'}")

# Determine which tests to use
use_parametric_dc = levene_dc_p > ALPHA  # Can use ANOVA if variances homogeneous
use_parametric_dm = levene_dm_p > ALPHA

print(f"\nRecommendation:")
print(f"  Dc: Use {'ANOVA (parametric)' if use_parametric_dc else 'Kruskal-Wallis (non-parametric)'}")
print(f"  Dm: Use {'ANOVA (parametric)' if use_parametric_dm else 'Kruskal-Wallis (non-parametric)'}")

# ============================================================================
# ANALYSIS 2: ONE-WAY ANOVA / KRUSKAL-WALLIS
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 2: OMNIBUS TESTS (GROUP DIFFERENCES)")
print("=" * 80)

omnibus_results = []

# ANOVA for Dc
f_dc, p_anova_dc = f_oneway(*groups_dc)

# Calculate effect size (eta-squared)
grand_mean_dc = merged['Dc'].mean()
ss_between_dc = sum(len(g) * (g.mean() - grand_mean_dc)**2 for g in groups_dc)
ss_total_dc = sum((merged['Dc'] - grand_mean_dc)**2)
eta2_dc = ss_between_dc / ss_total_dc
omega2_dc = (ss_between_dc - (len(PATHOLOGY_ORDER)-1)*np.var(merged['Dc'], ddof=1)) / (ss_total_dc + np.var(merged['Dc'], ddof=1))

print(f"\nCorrelation Dimension (Dc) - ANOVA:")
print(f"  F({len(PATHOLOGY_ORDER)-1}, {len(merged)-len(PATHOLOGY_ORDER)}) = {f_dc:.4f}")
print(f"  p-value = {p_anova_dc:.6e}")
print(f"  η² (eta-squared) = {eta2_dc:.4f}")
print(f"  ω² (omega-squared) = {omega2_dc:.4f}")
print(f"  Effect size: {'Very large' if eta2_dc >= 0.14 else 'Large' if eta2_dc >= 0.06 else 'Medium' if eta2_dc >= 0.01 else 'Small'}")

omnibus_results.append({
    'Dimension': 'Dc (Correlation)',
    'Test': 'One-way ANOVA',
    'Statistic': f'F = {f_dc:.4f}',
    'P_value': p_anova_dc,
    'Eta_squared': eta2_dc,
    'Omega_squared': omega2_dc,
    'Significant': p_anova_dc < ALPHA,
    'Interpretation': f"{'Very large' if eta2_dc >= 0.14 else 'Large' if eta2_dc >= 0.06 else 'Medium' if eta2_dc >= 0.01 else 'Small'} effect"
})

# Kruskal-Wallis for Dc (non-parametric alternative)
h_dc, p_kw_dc = kruskal(*groups_dc)
epsilon2_dc = (h_dc - len(PATHOLOGY_ORDER) + 1) / (len(merged) - len(PATHOLOGY_ORDER))

print(f"\nCorrelation Dimension (Dc) - Kruskal-Wallis:")
print(f"  H = {h_dc:.4f}")
print(f"  p-value = {p_kw_dc:.6e}")
print(f"  ε² (epsilon-squared) = {epsilon2_dc:.4f}")

omnibus_results.append({
    'Dimension': 'Dc (Correlation)',
    'Test': 'Kruskal-Wallis',
    'Statistic': f'H = {h_dc:.4f}',
    'P_value': p_kw_dc,
    'Eta_squared': np.nan,
    'Omega_squared': epsilon2_dc,
    'Significant': p_kw_dc < ALPHA,
    'Interpretation': 'Non-parametric test'
})

# ANOVA for Dm
f_dm, p_anova_dm = f_oneway(*groups_dm)

grand_mean_dm = merged['Dm'].mean()
ss_between_dm = sum(len(g) * (g.mean() - grand_mean_dm)**2 for g in groups_dm)
ss_total_dm = sum((merged['Dm'] - grand_mean_dm)**2)
eta2_dm = ss_between_dm / ss_total_dm
omega2_dm = (ss_between_dm - (len(PATHOLOGY_ORDER)-1)*np.var(merged['Dm'], ddof=1)) / (ss_total_dm + np.var(merged['Dm'], ddof=1))

print(f"\nMinkowski Dimension (Dm) - ANOVA:")
print(f"  F({len(PATHOLOGY_ORDER)-1}, {len(merged)-len(PATHOLOGY_ORDER)}) = {f_dm:.4f}")
print(f"  p-value = {p_anova_dm:.6e}")
print(f"  η² (eta-squared) = {eta2_dm:.4f}")
print(f"  ω² (omega-squared) = {omega2_dm:.4f}")
print(f"  Effect size: {'Very large' if eta2_dm >= 0.14 else 'Large' if eta2_dm >= 0.06 else 'Medium' if eta2_dm >= 0.01 else 'Small'}")

omnibus_results.append({
    'Dimension': 'Dm (Minkowski)',
    'Test': 'One-way ANOVA',
    'Statistic': f'F = {f_dm:.4f}',
    'P_value': p_anova_dm,
    'Eta_squared': eta2_dm,
    'Omega_squared': omega2_dm,
    'Significant': p_anova_dm < ALPHA,
    'Interpretation': f"{'Very large' if eta2_dm >= 0.14 else 'Large' if eta2_dm >= 0.06 else 'Medium' if eta2_dm >= 0.01 else 'Small'} effect"
})

# Kruskal-Wallis for Dm
h_dm, p_kw_dm = kruskal(*groups_dm)
epsilon2_dm = (h_dm - len(PATHOLOGY_ORDER) + 1) / (len(merged) - len(PATHOLOGY_ORDER))

print(f"\nMinkowski Dimension (Dm) - Kruskal-Wallis:")
print(f"  H = {h_dm:.4f}")
print(f"  p-value = {p_kw_dm:.6e}")
print(f"  ε² (epsilon-squared) = {epsilon2_dm:.4f}")

omnibus_results.append({
    'Dimension': 'Dm (Minkowski)',
    'Test': 'Kruskal-Wallis',
    'Statistic': f'H = {h_dm:.4f}',
    'P_value': p_kw_dm,
    'Eta_squared': np.nan,
    'Omega_squared': epsilon2_dm,
    'Significant': p_kw_dm < ALPHA,
    'Interpretation': 'Non-parametric test'
})

omnibus_df = pd.DataFrame(omnibus_results)
omnibus_df.to_excel(RESULTS_DIR / '03_omnibus_tests.xlsx', index=False)
print(f"\n✓ Saved: {RESULTS_DIR / '03_omnibus_tests.xlsx'}")

# ============================================================================
# ANALYSIS 3: POST-HOC PAIRWISE COMPARISONS
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 3: POST-HOC PAIRWISE COMPARISONS")
print("=" * 80)

# Tukey's HSD for Dc
print("\nTukey's HSD for Correlation Dimension (Dc):")
tukey_dc = pairwise_tukeyhsd(merged['Dc'], merged['Pathology'], alpha=ALPHA)
print(tukey_dc)

# Tukey's HSD for Dm
print("\nTukey's HSD for Minkowski Dimension (Dm):")
tukey_dm = pairwise_tukeyhsd(merged['Dm'], merged['Pathology'], alpha=ALPHA)
print(tukey_dm)

# Manual pairwise comparisons with effect sizes
pairwise_results = []

print("\n" + "-" * 80)
print("PAIRWISE EFFECT SIZES (Cohen's d)")
print("-" * 80)

for p1, p2 in combinations(PATHOLOGY_ORDER, 2):
    group1_dc = merged[merged['Pathology'] == p1]['Dc']
    group2_dc = merged[merged['Pathology'] == p2]['Dc']
    
    group1_dm = merged[merged['Pathology'] == p1]['Dm']
    group2_dm = merged[merged['Pathology'] == p2]['Dm']
    
    # T-test for Dc
    t_dc, p_dc = ttest_ind(group1_dc, group2_dc)
    
    # Cohen's d for Dc
    pooled_std_dc = np.sqrt(((len(group1_dc)-1)*group1_dc.std()**2 + 
                             (len(group2_dc)-1)*group2_dc.std()**2) / 
                            (len(group1_dc) + len(group2_dc) - 2))
    cohens_d_dc = (group1_dc.mean() - group2_dc.mean()) / pooled_std_dc
    
    # T-test for Dm
    t_dm, p_dm = ttest_ind(group1_dm, group2_dm)
    
    # Cohen's d for Dm
    pooled_std_dm = np.sqrt(((len(group1_dm)-1)*group1_dm.std()**2 + 
                             (len(group2_dm)-1)*group2_dm.std()**2) / 
                            (len(group1_dm) + len(group2_dm) - 2))
    cohens_d_dm = (group1_dm.mean() - group2_dm.mean()) / pooled_std_dm
    
    pairwise_results.append({
        'Comparison': f'{p1} vs {p2}',
        'Pathology_1': p1,
        'Pathology_2': p2,
        'Mean_Dc_1': group1_dc.mean(),
        'Mean_Dc_2': group2_dc.mean(),
        'Diff_Dc': group1_dc.mean() - group2_dc.mean(),
        'Cohens_d_Dc': cohens_d_dc,
        'Effect_Dc': 'Negligible' if abs(cohens_d_dc) < COHEN_SMALL else 
                     'Small' if abs(cohens_d_dc) < COHEN_MEDIUM else
                     'Medium' if abs(cohens_d_dc) < COHEN_LARGE else 'Large',
        'p_value_Dc': p_dc,
        'Mean_Dm_1': group1_dm.mean(),
        'Mean_Dm_2': group2_dm.mean(),
        'Diff_Dm': group1_dm.mean() - group2_dm.mean(),
        'Cohens_d_Dm': cohens_d_dm,
        'Effect_Dm': 'Negligible' if abs(cohens_d_dm) < COHEN_SMALL else 
                     'Small' if abs(cohens_d_dm) < COHEN_MEDIUM else
                     'Medium' if abs(cohens_d_dm) < COHEN_LARGE else 'Large',
        'p_value_Dm': p_dm
    })

pairwise_df = pd.DataFrame(pairwise_results)

# Bonferroni correction
n_comparisons = len(pairwise_df)
bonferroni_alpha = ALPHA / n_comparisons

pairwise_df['Significant_Dc_Bonferroni'] = pairwise_df['p_value_Dc'] < bonferroni_alpha
pairwise_df['Significant_Dm_Bonferroni'] = pairwise_df['p_value_Dm'] < bonferroni_alpha

# Sort by effect size
pairwise_df_sorted_dc = pairwise_df.sort_values('Cohens_d_Dc', key=abs, ascending=False)

print("\nTop 10 largest effect sizes for Dc:")
print(pairwise_df_sorted_dc[['Comparison', 'Cohens_d_Dc', 'Effect_Dc', 'Significant_Dc_Bonferroni']].head(10).to_string(index=False))

pairwise_df.to_excel(RESULTS_DIR / '04_pairwise_comparisons.xlsx', index=False)
print(f"\n✓ Saved: {RESULTS_DIR / '04_pairwise_comparisons.xlsx'}")

# ============================================================================
# ANALYSIS 4: TREND ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 4: TREND ANALYSIS ACROSS PROGRESSION")
print("=" * 80)

# Linear trend test using polynomial contrast
from scipy.stats import pearsonr, spearmanr

# Correlation with progression number
r_dc, p_r_dc = pearsonr(merged['Progression'], merged['Dc'])
rho_dc, p_rho_dc = spearmanr(merged['Progression'], merged['Dc'])

print(f"\nCorrelation Dimension (Dc) vs Progression:")
print(f"  Pearson r = {r_dc:.4f}, p = {p_r_dc:.6e}")
print(f"  Spearman ρ = {rho_dc:.4f}, p = {p_rho_dc:.6e}")
print(f"  Interpretation: {'Significant' if p_r_dc < ALPHA else 'Not significant'} {'positive' if r_dc > 0 else 'negative'} trend")

r_dm, p_r_dm = pearsonr(merged['Progression'], merged['Dm'])
rho_dm, p_rho_dm = spearmanr(merged['Progression'], merged['Dm'])

print(f"\nMinkowski Dimension (Dm) vs Progression:")
print(f"  Pearson r = {r_dm:.4f}, p = {p_r_dm:.6e}")
print(f"  Spearman ρ = {rho_dm:.4f}, p = {p_rho_dm:.6e}")
print(f"  Interpretation: {'Significant' if p_r_dm < ALPHA else 'Not significant'} {'positive' if r_dm > 0 else 'negative'} trend")

trend_results = pd.DataFrame([
    {'Dimension': 'Dc', 'Test': 'Pearson', 'Coefficient': r_dc, 'P_value': p_r_dc},
    {'Dimension': 'Dc', 'Test': 'Spearman', 'Coefficient': rho_dc, 'P_value': p_rho_dc},
    {'Dimension': 'Dm', 'Test': 'Pearson', 'Coefficient': r_dm, 'P_value': p_r_dm},
    {'Dimension': 'Dm', 'Test': 'Spearman', 'Coefficient': rho_dm, 'P_value': p_rho_dm}
])

trend_results.to_excel(RESULTS_DIR / '05_trend_analysis.xlsx', index=False)
print(f"\n✓ Saved: {RESULTS_DIR / '05_trend_analysis.xlsx'}")

# ============================================================================
# ANALYSIS 5: CLASSIFICATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 5: MACHINE LEARNING CLASSIFICATION")
print("=" * 80)

# Prepare data
X = merged[['Dc', 'Dm']].values
y = merged['Pathology'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified K-fold cross-validation
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Model 1: Random Forest
print("\n1. RANDOM FOREST CLASSIFIER")
print("-" * 40)

rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                            random_state=RANDOM_STATE, n_jobs=-1)

# Cross-validation scores
cv_scores_rf = cross_val_score(rf, X_scaled, y, cv=skf, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
print(f"Min: {cv_scores_rf.min():.4f}, Max: {cv_scores_rf.max():.4f}")

# Train final model
rf.fit(X_scaled, y)

# Feature importance
feature_importance_rf = rf.feature_importances_
print(f"\nFeature importance:")
print(f"  Dc: {100*feature_importance_rf[0]:.2f}%")
print(f"  Dm: {100*feature_importance_rf[1]:.2f}%")

# Get predictions for confusion matrix
y_pred_rf = cross_val_predict(rf, X_scaled, y, cv=skf)

# Confusion matrix
cm_rf = confusion_matrix(y, y_pred_rf, labels=PATHOLOGY_ORDER)

# Classification report
report_rf = classification_report(y, y_pred_rf, labels=PATHOLOGY_ORDER, 
                                  output_dict=True, zero_division=0)

print("\nClassification Report:")
print(classification_report(y, y_pred_rf, labels=PATHOLOGY_ORDER, zero_division=0))

# Model 2: Linear Discriminant Analysis
print("\n2. LINEAR DISCRIMINANT ANALYSIS (LDA)")
print("-" * 40)

lda = LinearDiscriminantAnalysis()
cv_scores_lda = cross_val_score(lda, X_scaled, y, cv=skf, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores_lda.mean():.4f} ± {cv_scores_lda.std():.4f}")

# Train LDA
lda.fit(X_scaled, y)
X_lda = lda.transform(X_scaled)

print(f"\nNumber of discriminant functions: {lda.n_components}")
print(f"Explained variance ratio: {lda.explained_variance_ratio_}")

# Save classification results
classification_results = pd.DataFrame({
    'Model': ['Random Forest', 'LDA'],
    'Mean_Accuracy': [cv_scores_rf.mean(), cv_scores_lda.mean()],
    'Std_Accuracy': [cv_scores_rf.std(), cv_scores_lda.std()],
    'Min_Accuracy': [cv_scores_rf.min(), cv_scores_lda.min()],
    'Max_Accuracy': [cv_scores_rf.max(), cv_scores_lda.max()]
})

classification_results.to_excel(RESULTS_DIR / '06_classification_performance.xlsx', index=False)
print(f"\n✓ Saved: {RESULTS_DIR / '06_classification_performance.xlsx'}")

# Save confusion matrix
cm_df = pd.DataFrame(cm_rf, index=PATHOLOGY_ORDER, columns=PATHOLOGY_ORDER)
cm_df.to_excel(RESULTS_DIR / '07_confusion_matrix.xlsx')
print(f"✓ Saved: {RESULTS_DIR / '07_confusion_matrix.xlsx'}")

# Save detailed classification report
report_df = pd.DataFrame(report_rf).T
report_df.to_excel(RESULTS_DIR / '08_classification_report.xlsx')
print(f"✓ Saved: {RESULTS_DIR / '08_classification_report.xlsx'}")

# ============================================================================
# ANALYSIS 6: ROC ANALYSIS (One-vs-Rest)
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 6: ROC CURVE ANALYSIS (One-vs-Rest)")
print("=" * 80)

# Binarize labels for one-vs-rest ROC
y_bin = label_binarize(y, classes=PATHOLOGY_ORDER)

# Train one-vs-rest classifier
rf_ovr = OneVsRestClassifier(RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,
                                                     max_depth=RF_MAX_DEPTH,
                                                     random_state=RANDOM_STATE,
                                                     n_jobs=-1))
rf_ovr.fit(X_scaled, y_bin)
y_score = rf_ovr.predict_proba(X_scaled)

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i, pathology in enumerate(PATHOLOGY_ORDER):
    fpr[pathology] = []
    tpr[pathology] = []
    roc_auc[pathology] = 0
    
    if y_bin[:, i].sum() > 0:  # Only if class exists
        fpr[pathology], tpr[pathology], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[pathology] = auc(fpr[pathology], tpr[pathology])
        
        print(f"{pathology}: AUC = {roc_auc[pathology]:.4f}")

# Save ROC results
roc_results = pd.DataFrame({
    'Pathology': PATHOLOGY_ORDER,
    'AUC': [roc_auc[p] for p in PATHOLOGY_ORDER],
    'Interpretation': [
        'Excellent' if roc_auc[p] >= 0.9 else
        'Good' if roc_auc[p] >= 0.8 else
        'Fair' if roc_auc[p] >= 0.7 else
        'Poor' if roc_auc[p] >= 0.6 else 'Fail'
        for p in PATHOLOGY_ORDER
    ]
})

roc_results.to_excel(RESULTS_DIR / '09_roc_auc_scores.xlsx', index=False)
print(f"\n✓ Saved: {RESULTS_DIR / '09_roc_auc_scores.xlsx'}")

# ============================================================================
# VISUALIZATION 1: BOX PLOTS BY PATHOLOGY
# ============================================================================

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('RQ3: Fractal Dimensions by Pathology Type',
             fontsize=14, fontweight='bold')

# Plot 1: Correlation Dimension
ax = axes[0]
data_dc = [merged[merged['Pathology'] == p]['Dc'].values for p in PATHOLOGY_ORDER]
bp1 = ax.boxplot(data_dc, labels=PATHOLOGY_ORDER, patch_artist=True, notch=True)

for patch, color in zip(bp1['boxes'], plt.cm.Set3(np.linspace(0, 1, len(PATHOLOGY_ORDER)))):
    patch.set_facecolor(color)

ax.set_xlabel('Pathology Type', fontsize=11)
ax.set_ylabel('Correlation Dimension (Dc)', fontsize=11)
ax.set_title(f'Correlation Dimension\nANOVA: F={f_dc:.2f}, p<{p_anova_dc:.2e}, η²={eta2_dc:.3f}',
            fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Minkowski Dimension
ax = axes[1]
data_dm = [merged[merged['Pathology'] == p]['Dm'].values for p in PATHOLOGY_ORDER]
bp2 = ax.boxplot(data_dm, labels=PATHOLOGY_ORDER, patch_artist=True, notch=True)

for patch, color in zip(bp2['boxes'], plt.cm.Set3(np.linspace(0, 1, len(PATHOLOGY_ORDER)))):
    patch.set_facecolor(color)

ax.set_xlabel('Pathology Type', fontsize=11)
ax.set_ylabel('Minkowski Dimension (Dm)', fontsize=11)
ax.set_title(f'Minkowski Dimension\nANOVA: F={f_dm:.2f}, p<{p_anova_dm:.2e}, η²={eta2_dm:.3f}',
            fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig1_boxplots_by_pathology.tif', format='tif', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOTS_DIR / 'fig1_boxplots_by_pathology.tif'}")

# Export boxplot data
boxplot_export = merged[['Pathology', 'Dc', 'Dm']].copy()
boxplot_export.to_excel(ORIGIN_DATA_DIR / 'fig1_boxplot_data.xlsx', index=False)
print(f"✓ Saved: {ORIGIN_DATA_DIR / 'fig1_boxplot_data.xlsx'}")

## **VISUALIZATION 2: Violin Plots (Distribution Shapes)**

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('RQ3: Distribution of Fractal Dimensions by Pathology',
             fontsize=14, fontweight='bold')

# Violin plot for Dc
ax = axes[0]
parts_dc = ax.violinplot([merged[merged['Pathology'] == p]['Dc'].values 
                          for p in PATHOLOGY_ORDER],
                         positions=range(len(PATHOLOGY_ORDER)),
                         showmeans=True, showmedians=True)
ax.set_xticks(range(len(PATHOLOGY_ORDER)))
ax.set_xticklabels(PATHOLOGY_ORDER)
ax.set_ylabel('Correlation Dimension (Dc)')
ax.set_title('Distribution Shapes - Correlation Dimension')
ax.grid(True, alpha=0.3, axis='y')

# Violin plot for Dm
ax = axes[1]
parts_dm = ax.violinplot([merged[merged['Pathology'] == p]['Dm'].values 
                          for p in PATHOLOGY_ORDER],
                         positions=range(len(PATHOLOGY_ORDER)),
                         showmeans=True, showmedians=True)
ax.set_xticks(range(len(PATHOLOGY_ORDER)))
ax.set_xticklabels(PATHOLOGY_ORDER)
ax.set_ylabel('Minkowski Dimension (Dm)')
ax.set_title('Distribution Shapes - Minkowski Dimension')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig2_violin_plots.tif', format='tif', dpi=300)
plt.close()

## **VISUALIZATION 3: Effect Size Heatmap**

# Create Cohen's d matrix for all pairwise comparisons
effect_matrix_dc = np.zeros((len(PATHOLOGY_ORDER), len(PATHOLOGY_ORDER)))
effect_matrix_dm = np.zeros((len(PATHOLOGY_ORDER), len(PATHOLOGY_ORDER)))

for i, p1 in enumerate(PATHOLOGY_ORDER):
    for j, p2 in enumerate(PATHOLOGY_ORDER):
        if i != j:
            g1_dc = merged[merged['Pathology'] == p1]['Dc']
            g2_dc = merged[merged['Pathology'] == p2]['Dc']
            pooled_std = np.sqrt(((len(g1_dc)-1)*g1_dc.std()**2 + 
                                 (len(g2_dc)-1)*g2_dc.std()**2) / 
                                (len(g1_dc) + len(g2_dc) - 2))
            effect_matrix_dc[i, j] = (g1_dc.mean() - g2_dc.mean()) / pooled_std
            
            # Same for Dm
            g1_dm = merged[merged['Pathology'] == p1]['Dm']
            g2_dm = merged[merged['Pathology'] == p2]['Dm']
            pooled_std_dm = np.sqrt(((len(g1_dm)-1)*g1_dm.std()**2 + 
                                     (len(g2_dm)-1)*g2_dm.std()**2) / 
                                    (len(g1_dm) + len(g2_dm) - 2))
            effect_matrix_dm[i, j] = (g1_dm.mean() - g2_dm.mean()) / pooled_std_dm

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("RQ3: Pairwise Effect Sizes (Cohen's d)", fontsize=14, fontweight='bold')

# Heatmap for Dc
ax = axes[0]
im1 = ax.imshow(effect_matrix_dc, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
ax.set_xticks(range(len(PATHOLOGY_ORDER)))
ax.set_yticks(range(len(PATHOLOGY_ORDER)))
ax.set_xticklabels(PATHOLOGY_ORDER)
ax.set_yticklabels(PATHOLOGY_ORDER)
ax.set_title('Correlation Dimension (Dc)')
plt.colorbar(im1, ax=ax, label="Cohen's d")

# Add values to cells
for i in range(len(PATHOLOGY_ORDER)):
    for j in range(len(PATHOLOGY_ORDER)):
        if i != j:
            text = ax.text(j, i, f'{effect_matrix_dc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=14)

# Heatmap for Dm
ax = axes[1]
im2 = ax.imshow(effect_matrix_dm, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
ax.set_xticks(range(len(PATHOLOGY_ORDER)))
ax.set_yticks(range(len(PATHOLOGY_ORDER)))
ax.set_xticklabels(PATHOLOGY_ORDER)
ax.set_yticklabels(PATHOLOGY_ORDER)
ax.set_title('Minkowski Dimension (Dm)')
plt.colorbar(im2, ax=ax, label="Cohen's d")

for i in range(len(PATHOLOGY_ORDER)):
    for j in range(len(PATHOLOGY_ORDER)):
        if i != j:
            text = ax.text(j, i, f'{effect_matrix_dm[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=14)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig3_effect_size_heatmap.tif', format='tif', dpi=300)
plt.close()

## **VISUALIZATION 4: Trend Analysis Plots**

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('RQ3: Fractal Dimensions vs Pathology Progression',
             fontsize=14, fontweight='bold')

# Trend for Dc
ax = axes[0]
for pathology in PATHOLOGY_ORDER:
    subset = merged[merged['Pathology'] == pathology]
    progression = subset['Progression'].iloc[0]
    ax.scatter([progression]*len(subset), subset['Dc'], alpha=0.3, s=20)

# Add mean with error bars
means_dc = [merged[merged['Pathology']==p]['Dc'].mean() for p in PATHOLOGY_ORDER]
stds_dc = [merged[merged['Pathology']==p]['Dc'].std() for p in PATHOLOGY_ORDER]
ax.errorbar(range(len(PATHOLOGY_ORDER)), means_dc, yerr=stds_dc,
           fmt='ro-', linewidth=2, markersize=8, capsize=5, label='Mean ± SD')

# Fit line
z = np.polyfit(merged['Progression'], merged['Dc'], 1)
p = np.poly1d(z)
ax.plot(range(len(PATHOLOGY_ORDER)), p(range(len(PATHOLOGY_ORDER))),
       'b--', linewidth=2, label=f'Linear fit: r={r_dc:.3f}')

ax.set_xticks(range(len(PATHOLOGY_ORDER)))
ax.set_xticklabels(PATHOLOGY_ORDER)
ax.set_xlabel('Pathology Progression →')
ax.set_ylabel('Correlation Dimension (Dc)')
ax.set_title(f'Dc vs Progression\nPearson r={r_dc:.3f}, p={p_r_dc:.4e}')
ax.legend()
ax.grid(True, alpha=0.3)

# Same for Dm
ax = axes[1]
for pathology in PATHOLOGY_ORDER:
    subset = merged[merged['Pathology'] == pathology]
    progression = subset['Progression'].iloc[0]
    ax.scatter([progression]*len(subset), subset['Dm'], alpha=0.3, s=20)

means_dm = [merged[merged['Pathology']==p]['Dm'].mean() for p in PATHOLOGY_ORDER]
stds_dm = [merged[merged['Pathology']==p]['Dm'].std() for p in PATHOLOGY_ORDER]
ax.errorbar(range(len(PATHOLOGY_ORDER)), means_dm, yerr=stds_dm,
           fmt='ro-', linewidth=2, markersize=8, capsize=5, label='Mean ± SD')

z_dm = np.polyfit(merged['Progression'], merged['Dm'], 1)
p_dm = np.poly1d(z_dm)
ax.plot(range(len(PATHOLOGY_ORDER)), p_dm(range(len(PATHOLOGY_ORDER))),
       'b--', linewidth=2, label=f'Linear fit: r={r_dm:.3f}')

ax.set_xticks(range(len(PATHOLOGY_ORDER)))
ax.set_xticklabels(PATHOLOGY_ORDER)
ax.set_xlabel('Pathology Progression →')
ax.set_ylabel('Minkowski Dimension (Dm)')
ax.set_title(f'Dm vs Progression\nPearson r={r_dm:.3f}, p={p_r_dm:.4e}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig4_trend_analysis.tif', format='tif', dpi=300)
plt.close()

## **VISUALIZATION 5: LDA Scatter Plot**

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
fig.suptitle('RQ3: Linear Discriminant Analysis', fontsize=14, fontweight='bold')

colors = plt.cm.Set3(np.linspace(0, 1, len(PATHOLOGY_ORDER)))

for i, pathology in enumerate(PATHOLOGY_ORDER):
    subset_idx = merged['Pathology'] == pathology
    if lda.scalings_.shape[1] >= 2:
        ax.scatter(X_lda[subset_idx, 0], X_lda[subset_idx, 1],
                  c=[colors[i]], label=pathology, alpha=0.6, s=50, edgecolors='black')
    else:
        ax.scatter(X_lda[subset_idx, 0], np.zeros(subset_idx.sum()),
                  c=[colors[i]], label=pathology, alpha=0.6, s=50, edgecolors='black')

ax.set_xlabel(f'LD1 ({100*lda.explained_variance_ratio_[0]:.1f}% variance)', fontsize=11)
if lda.scalings_.shape[1] >= 2:
    ax.set_ylabel(f'LD2 ({100*lda.explained_variance_ratio_[1]:.1f}% variance)', fontsize=11)
ax.set_title('Discriminant Function Scatter Plot')
ax.legend(ncol=2, fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig5_lda_scatter.tif', format='tif', dpi=300)
plt.close()

## **VISUALIZATION 6: Confusion Matrix Heatmap**

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
fig.suptitle('RQ3: Confusion Matrix (Random Forest)', fontsize=14, fontweight='bold')

# Normalize confusion matrix
cm_normalized = cm_rf.astype('float') / cm_rf.sum(axis=1)[:, np.newaxis]

im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(PATHOLOGY_ORDER)))
ax.set_yticks(range(len(PATHOLOGY_ORDER)))
ax.set_xticklabels(PATHOLOGY_ORDER, fontsize=10)
ax.set_yticklabels(PATHOLOGY_ORDER, fontsize=10)
ax.set_xlabel('Predicted Pathology', fontsize=11)
ax.set_ylabel('True Pathology', fontsize=11)

# Add text annotations
for i in range(len(PATHOLOGY_ORDER)):
    for j in range(len(PATHOLOGY_ORDER)):
        text = ax.text(j, i, f'{cm_rf[i, j]}\n({100*cm_normalized[i, j]:.1f}%)',
                      ha="center", va="center",
                      color="white" if cm_normalized[i, j] > 0.5 else "black",
                      fontsize=9)

plt.colorbar(im, ax=ax, label='Proportion')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig6_confusion_matrix.tif', format='tif', dpi=300)
plt.close()

## **VISUALIZATION 7: ROC Curves**

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
fig.suptitle('RQ3: ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')

for idx, pathology in enumerate(PATHOLOGY_ORDER):
    row, col = idx // 4, idx % 4
    ax = axes[row, col]
    
    if len(fpr[pathology]) > 0:
        ax.plot(fpr[pathology], tpr[pathology], color='darkorange', lw=2,
               label=f'ROC (AUC = {roc_auc[pathology]:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{pathology} vs Rest')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

# Hide last subplot if odd number
if len(PATHOLOGY_ORDER) % 2 == 1:
    axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig7_roc_curves.tif', format='tif', dpi=300)
plt.close()

## **VISUALIZATION 8: Feature Importance**

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
fig.suptitle('RQ3: Feature Importance (Random Forest)', fontsize=14, fontweight='bold')

features = ['Dc\n(Correlation Dimension)', 'Dm\n(Minkowski Dimension)']
importances = feature_importance_rf

bars = ax.bar(features, importances, color=['steelblue', 'coral'],
             edgecolor='black', alpha=0.7, width=0.6)
ax.set_ylabel('Importance Score', fontsize=11)
ax.set_title('Relative Contribution to Classification')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, max(importances)*1.2])

for bar, imp in zip(bars, importances):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{100*imp:.1f}%', ha='center', va='bottom',
           fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig8_feature_importance.tif', format='tif', dpi=300)
plt.close()


## ================================
## FINAL SUMMARY REPORT
## ================================

# --- ROC-AUC summary (already correct) ---
roc_summary = '\n'.join([
    f"   {p}: {roc_auc[p]:.3f} "
    f"({'Excellent' if roc_auc[p] >= 0.9 else 'Good' if roc_auc[p] >= 0.8 else 'Fair' if roc_auc[p] >= 0.7 else 'Poor'})"
    for p in PATHOLOGY_ORDER
])

# --- Interpretation blocks (MUST be computed BEFORE the f-string) ---
if p_anova_dc < ALPHA:
    anova_interpretation = (
        f"The ANOVA shows "
        f"{'very large' if eta2_dc >= 0.14 else 'large' if eta2_dc >= 0.06 else 'medium'} "
        f"effect size (η²={eta2_dc:.3f}), indicating that "
        f"{100*eta2_dc:.1f}% of variance in fractal dimensions is explained by pathology type."
    )
else:
    anova_interpretation = "Groups do not differ significantly in fractal dimensions."

if cv_scores_rf.mean() > 0.5:
    classification_interpretation = (
        f"Classification accuracy of {100*cv_scores_rf.mean():.1f}% demonstrates that fractal dimensions can "
        f"{'reliably' if cv_scores_rf.mean() > 0.7 else 'moderately'} distinguish pathologies."
    )
else:
    classification_interpretation = "Low classification accuracy suggests limited discriminative power."

if abs(r_dc) > 0.3:
    trend_interpretation = (
        f"The {'positive' if r_dc > 0 else 'negative'} trend "
        f"(r={r_dc:.3f}) indicates fractal complexity "
        f"{'increases' if r_dc > 0 else 'decreases'} with malignancy progression."
    )
else:
    trend_interpretation = "No clear linear trend across progression sequence."

# --- Final report ---
summary_report = f"""
================================================================================
RESEARCH QUESTION 3 (RQ3): PATHOLOGY DISCRIMINATION - SUMMARY
================================================================================

Question: Do different pathological tissue subtypes exhibit statistically 
distinct fractal characteristics of nuclear spatial distribution?

ANSWER: {('YES' if p_anova_dc < ALPHA and eta2_dc >= 0.06 and cv_scores_rf.mean() > 0.6 else 'PARTIAL' if p_anova_dc < ALPHA else 'NO')}

Pathologies {'CAN' if p_anova_dc < ALPHA and eta2_dc >= 0.06 else 'SHOW LIMITED ABILITY TO'} be distinguished using fractal dimensions.

================================================================================
KEY FINDINGS
================================================================================

1. OMNIBUS TESTS (Overall Group Differences):
   ------------------------------------------------
   Correlation Dimension (Dc):
   - ANOVA: F = {f_dc:.2f}, p < {p_anova_dc:.2e}
   - Effect size: η² = {eta2_dc:.3f}
   - Interpretation: Groups differ {'significantly' if p_anova_dc < ALPHA else 'not significantly'}
   
   Minkowski Dimension (Dm):
   - ANOVA: F = {f_dm:.2f}, p < {p_anova_dm:.2e}
   - Effect size: η² = {eta2_dm:.3f}
   - Interpretation: Groups differ {'significantly' if p_anova_dm < ALPHA else 'not significantly'}

2. PAIRWISE COMPARISONS:
   ------------------------------------------------
   Total comparisons: {len(pairwise_df)} pairs
   Significant (Bonferroni-corrected): 
   - Dc: {pairwise_df['Significant_Dc_Bonferroni'].sum()} pairs
   - Dm: {pairwise_df['Significant_Dm_Bonferroni'].sum()} pairs
   
   Largest effect sizes:
   {pairwise_df_sorted_dc[['Comparison', 'Cohens_d_Dc']].head(3).to_string(index=False)}

3. TREND ANALYSIS:
   ------------------------------------------------
   - Dc: r = {r_dc:.3f}, p = {p_r_dc:.4e}
   - Dm: r = {r_dm:.3f}, p = {p_r_dm:.4e}

4. CLASSIFICATION PERFORMANCE:
   ------------------------------------------------
   Random Forest (5-fold CV):
   - Accuracy: {100*cv_scores_rf.mean():.2f}% ± {100*cv_scores_rf.std():.2f}%
   
   Linear Discriminant Analysis:
   - Accuracy: {100*cv_scores_lda.mean():.2f}% ± {100*cv_scores_lda.std():.2f}%

5. FEATURE IMPORTANCE:
   ------------------------------------------------
   - Dc: {100*feature_importance_rf[0]:.1f}%
   - Dm: {100*feature_importance_rf[1]:.1f}%

6. ROC-AUC SCORES (One-vs-Rest):
   ------------------------------------------------
{roc_summary}

================================================================================
INTERPRETATION
================================================================================

{anova_interpretation}

{classification_interpretation}

{trend_interpretation}

================================================================================
"""

with open(RESULTS_DIR / '00_RQ3_SUMMARY.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(summary_report)
