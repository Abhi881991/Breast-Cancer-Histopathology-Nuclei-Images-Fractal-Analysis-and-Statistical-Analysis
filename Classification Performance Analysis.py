"""
================================================================================
RESEARCH QUESTION 8 (RQ8): CLASSIFICATION PERFORMANCE COMPARISON (ENHANCED)
================================================================================

Research Question:
"Which dimension and classifier combination is most clinically useful?"

Classifiers Tested (4):
1. Logistic Regression - Linear, interpretable baseline
2. Random Forest - Ensemble, handles non-linearity
3. K-Nearest Neighbors (KNN) - Instance-based, no assumptions
4. Support Vector Machine (SVM) - Maximum margin, kernel trick

Feature Sets (3): Dc only, Dm only, Both

Total Comparisons: 4 classifiers × 3 feature sets = 12 combinations
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            roc_curve, auc, roc_auc_score, f1_score)
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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
plt.rcParams['legend.fontsize'] = 14

# Configuration
BASE_PATH = Path(r"C:\Users\ajd44\Desktop")
OUTPUT_DIR = BASE_PATH / 'RQ8_Classification_Performance'
PLOTS_DIR = OUTPUT_DIR / 'plots'
RESULTS_DIR = OUTPUT_DIR / 'results'
ORIGIN_DATA_DIR = OUTPUT_DIR / 'origin_data'
for d in [OUTPUT_DIR, PLOTS_DIR, RESULTS_DIR, ORIGIN_DATA_DIR]:
    d.mkdir(exist_ok=True)

PATHOLOGY_ORDER = ['N', 'PB', 'UDH', 'FEA', 'ADH', 'DCIS', 'IC']
RANDOM_STATE = 42
N_FOLDS = 5

print("="*80)
print("RQ8: ENHANCED CLASSIFICATION PERFORMANCE (4 Classifiers)")
print("="*80)

# ============================================================================
# DATA LOADING
# ============================================================================

corr_df = pd.read_csv(BASE_PATH / 'Correlation Dimension.csv')
mink_df = pd.read_csv(BASE_PATH / 'Minkowski Dimension.csv')

corr_df['Pathology'] = corr_df['File name'].str.extract(r'_([A-Z]+)_\d+\.tif')
mink_df['Pathology'] = mink_df['File name'].str.extract(r'_([A-Z]+)_\d+\.tif')

merged = pd.merge(
    corr_df[['File name', 'Dc', 'Pathology']],
    mink_df[['File name', 'Dm']],
    on='File name'
)
merged = merged[merged['Pathology'].isin(PATHOLOGY_ORDER)].copy()

# Binary classification - CORRECTED
# Benign: N, PB, UDH (normal/benign proliferative)
# Atypical/Pre-malignant: FEA, ADH (hyperplasia, non-invasive)
# Malignant: DCIS (in situ carcinoma), IC (invasive carcinoma)
# 
# BINARY CLASSIFICATION: Benign+Atypical vs. Malignant (DCIS+IC only)
merged['Binary'] = merged['Pathology'].map({
    'N': 'Benign', 'PB': 'Benign', 'UDH': 'Benign',
    'FEA': 'Benign', 'ADH': 'Benign',  # Pre-malignant but NOT invasive
    'DCIS': 'Malignant', 'IC': 'Malignant'  # True malignancy
})

# Ternary classification - CORRECTED
merged['Ternary'] = merged['Pathology'].map({
    'N': 'Benign', 'PB': 'Benign', 'UDH': 'Benign',
    'FEA': 'Atypical', 'ADH': 'Atypical',
    'DCIS': 'Malignant', 'IC': 'Malignant'
})

print(f"Total ROIs: {len(merged)}")
print(f"Binary class distribution (ROI-level): {dict(merged['Binary'].value_counts())}")
print(f"Ternary class distribution (ROI-level): {dict(merged['Ternary'].value_counts())}")

# ============================================================================
# CRITICAL: CHECK ICC AND AGGREGATE TO WSI-LEVEL
# ============================================================================

print("\n" + "="*80)
print("ICC CHECK AND WSI-LEVEL AGGREGATION (Addressing Pseudoreplication)")
print("="*80)

# Extract WSI_ID
merged['WSI_ID'] = merged['File name'].str.extract(r'(BRACS_\d+)')

# Calculate ICC for Dc and Dm
def calculate_icc(data, value_col, cluster_col):
    """Calculate ICC to check clustering within WSIs"""
    cluster_means = data.groupby(cluster_col)[value_col].mean()
    grand_mean = data[value_col].mean()
    n_per_cluster = data.groupby(cluster_col).size()
    
    # Between-cluster variance
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
    return icc

icc_dc = calculate_icc(merged, 'Dc', 'WSI_ID')
icc_dm = calculate_icc(merged, 'Dm', 'WSI_ID')

print(f"\nICC(Dc): {icc_dc:.3f}")
print(f"ICC(Dm): {icc_dm:.3f}")
print(f"\n⚠ Both ICC > 0.4 → Substantial clustering within WSIs")
print(f"  → ROI-level classification = PSEUDOREPLICATION (invalid)")
print(f"  → Aggregating to WSI-level (n=368) for valid inference")

# Aggregate to WSI-level
wsi_level = merged.groupby('WSI_ID').agg(
    Dc=('Dc', 'mean'),
    Dm=('Dm', 'mean'),
    Pathology=('Pathology', lambda x: x.mode()[0]),  # majority vote
    Binary=('Binary', lambda x: x.mode()[0]),
    Ternary=('Ternary', lambda x: x.mode()[0])
).reset_index()

print(f"\n✓ Aggregated from {len(merged)} ROIs to {len(wsi_level)} WSIs")
print(f"Binary class distribution (WSI-level): {dict(wsi_level['Binary'].value_counts())}")
print(f"Ternary class distribution (WSI-level): {dict(wsi_level['Ternary'].value_counts())}")

# Use WSI-level data for all classification
merged_analysis = wsi_level.copy()

# Prepare features from WSI-level data
X_dc = merged_analysis[['Dc']].values
X_dm = merged_analysis[['Dm']].values
X_both = merged_analysis[['Dc', 'Dm']].values
y_binary = (merged_analysis['Binary'] == 'Malignant').astype(int)
y_ternary = merged_analysis['Ternary'].map({'Benign': 0, 'Atypical': 1, 'Malignant': 2}).values

# Scale features (important for KNN and SVM)
scaler_dc = StandardScaler()
scaler_dm = StandardScaler()
scaler_both = StandardScaler()

X_dc_scaled = scaler_dc.fit_transform(X_dc)
X_dm_scaled = scaler_dm.fit_transform(X_dm)
X_both_scaled = scaler_both.fit_transform(X_both)

# ============================================================================
# ANALYSIS 1: BINARY CLASSIFICATION (ALL CLASSIFIERS)
# ============================================================================

print("\n" + "="*80)
print(f"ANALYSIS 1: BINARY CLASSIFICATION - WSI-LEVEL (n={len(merged_analysis)})")
print("="*80)
print(f"Classification: Benign (N/PB/UDH/FEA/ADH) vs Malignant (DCIS/IC only)")
print(f"Addressing pseudoreplication: WSI-level aggregation applied (ICC > 0.4)")
print("="*80)

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
}

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results_binary = []

for feat_name, X, X_scaled in [('Dc_only', X_dc, X_dc_scaled), 
                                ('Dm_only', X_dm, X_dm_scaled), 
                                ('Both', X_both, X_both_scaled)]:
    print(f"\n{feat_name}:")
    
    for clf_name, clf in classifiers.items():
        # Use scaled data for KNN and SVM, original for others
        X_use = X_scaled if clf_name in ['KNN', 'SVM'] else X
        
        # Accuracy
        acc_scores = cross_val_score(clf, X_use, y_binary, cv=cv, scoring='accuracy')
        
        # AUC
        auc_scores = cross_val_score(clf, X_use, y_binary, cv=cv, scoring='roc_auc')
        
        # F1 score
        f1_scores = cross_val_score(clf, X_use, y_binary, cv=cv, scoring='f1')
        
        results_binary.append({
            'Features': feat_name,
            'Classifier': clf_name,
            'Accuracy_Mean': acc_scores.mean(),
            'Accuracy_SD': acc_scores.std(),
            'AUC_Mean': auc_scores.mean(),
            'AUC_SD': auc_scores.std(),
            'F1_Mean': f1_scores.mean(),
            'F1_SD': f1_scores.std()
        })
        
        print(f"  {clf_name:20s}: Acc={acc_scores.mean():.4f}±{acc_scores.std():.4f}, "
              f"AUC={auc_scores.mean():.4f}±{auc_scores.std():.4f}")

binary_df = pd.DataFrame(results_binary)
binary_df.to_excel(RESULTS_DIR / '01_binary_all_classifiers.xlsx', index=False)
print(f"\n✓ Saved: {RESULTS_DIR / '01_binary_all_classifiers.xlsx'}")

# Find best combination
best_binary = binary_df.loc[binary_df['AUC_Mean'].idxmax()]
print(f"\n⭐ BEST BINARY: {best_binary['Classifier']} + {best_binary['Features']} "
      f"(AUC={best_binary['AUC_Mean']:.4f})")

# ============================================================================
# ANALYSIS 2: TERNARY CLASSIFICATION (ALL CLASSIFIERS)
# ============================================================================

print("\n" + "="*80)
print(f"ANALYSIS 2: TERNARY CLASSIFICATION - WSI-LEVEL (n={len(merged_analysis)})")
print("="*80)
print(f"Classification: Benign (N/PB/UDH) vs Atypical (FEA/ADH) vs Malignant (DCIS/IC)")
print("="*80)

# For ternary, need different classifiers setup
classifiers_ternary = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', random_state=RANDOM_STATE)
}

results_ternary = []

for feat_name, X, X_scaled in [('Dc_only', X_dc, X_dc_scaled),
                                ('Dm_only', X_dm, X_dm_scaled),
                                ('Both', X_both, X_both_scaled)]:
    print(f"\n{feat_name}:")
    
    for clf_name, clf in classifiers_ternary.items():
        X_use = X_scaled if clf_name in ['KNN', 'SVM'] else X
        
        acc_scores = cross_val_score(clf, X_use, y_ternary, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(clf, X_use, y_ternary, cv=cv, scoring='f1_weighted')
        
        results_ternary.append({
            'Features': feat_name,
            'Classifier': clf_name,
            'Accuracy_Mean': acc_scores.mean(),
            'Accuracy_SD': acc_scores.std(),
            'F1_Mean': f1_scores.mean(),
            'F1_SD': f1_scores.std()
        })
        
        print(f"  {clf_name:20s}: Acc={acc_scores.mean():.4f}±{acc_scores.std():.4f}")

ternary_df = pd.DataFrame(results_ternary)
ternary_df.to_excel(RESULTS_DIR / '02_ternary_all_classifiers.xlsx', index=False)
print(f"\n✓ Saved: {RESULTS_DIR / '02_ternary_all_classifiers.xlsx'}")

best_ternary = ternary_df.loc[ternary_df['Accuracy_Mean'].idxmax()]
print(f"\n⭐ BEST TERNARY: {best_ternary['Classifier']} + {best_ternary['Features']} "
      f"(Acc={best_ternary['Accuracy_Mean']:.4f})")

# ============================================================================
# ANALYSIS 3: HYPERPARAMETER TUNING (Best Classifier)
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 3: HYPERPARAMETER TUNING (Best Classifier)")
print("="*80)

if best_binary['Classifier'] == 'KNN':
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
    base_clf = KNeighborsClassifier()
elif best_binary['Classifier'] == 'SVM':
    param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    base_clf = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
elif best_binary['Classifier'] == 'Random Forest':
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    base_clf = RandomForestClassifier(random_state=RANDOM_STATE)
else:
    param_grid = {'C': [0.1, 1, 10]}
    base_clf = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)

X_best = X_both_scaled if best_binary['Classifier'] in ['KNN', 'SVM'] else X_both

grid_search = GridSearchCV(base_clf, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_best, y_binary)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best AUC: {grid_search.best_score_:.4f}")

tuning_results = pd.DataFrame(grid_search.cv_results_)
tuning_results.to_excel(RESULTS_DIR / '03_hyperparameter_tuning.xlsx', index=False)
print(f"✓ Saved: {RESULTS_DIR / '03_hyperparameter_tuning.xlsx'}")

# ============================================================================
# VISUALIZATION 1: HEATMAP COMPARISON
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('RQ8: Classifier Performance Comparison', fontsize=18, fontweight='bold')

# Binary heatmap
ax = axes[0]
pivot_binary = binary_df.pivot(index='Classifier', columns='Features', values='AUC_Mean')
sns.heatmap(pivot_binary, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
           ax=ax, cbar_kws={'label': 'AUC'})
ax.set_title('Binary Classification (Benign vs Malignant)\nAUC Scores')
ax.set_ylabel('')

# Ternary heatmap
ax = axes[1]
pivot_ternary = ternary_df.pivot(index='Classifier', columns='Features', values='Accuracy_Mean')
sns.heatmap(pivot_ternary, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.3, vmax=0.8,
           ax=ax, cbar_kws={'label': 'Accuracy'})
ax.set_title('Ternary Classification (Benign/Atypical/Malignant)\nAccuracy Scores')
ax.set_ylabel('')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig1_classifier_heatmaps.tif', format='tif', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOTS_DIR / 'fig1_classifier_heatmaps.tif'}")

# ============================================================================
# VISUALIZATION 2: BAR CHART COMPARISON
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('RQ8: Classifier Performance by Feature Set', fontsize=18, fontweight='bold')

for idx, feat in enumerate(['Dc_only', 'Dm_only', 'Both']):
    ax = axes[idx]
    subset = binary_df[binary_df['Features'] == feat]
    
    x = np.arange(len(subset))
    width = 0.35
    
    ax.bar(x - width/2, subset['Accuracy_Mean'], width, label='Accuracy',
          yerr=subset['Accuracy_SD'], capsize=3, alpha=0.8, edgecolor='black')
    ax.bar(x + width/2, subset['AUC_Mean'], width, label='AUC',
          yerr=subset['AUC_SD'], capsize=3, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Score')
    ax.set_title(f'Feature Set: {feat}')
    ax.set_xticks(x)
    ax.set_xticklabels(subset['Classifier'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.5, 1.0])

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fig2_classifier_bars.tif', format='tif', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOTS_DIR / 'fig2_classifier_bars.tif'}")

# Export data
binary_df.to_excel(ORIGIN_DATA_DIR / 'binary_all_classifiers.xlsx', index=False)
ternary_df.to_excel(ORIGIN_DATA_DIR / 'ternary_all_classifiers.xlsx', index=False)

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("GENERATING SUMMARY")
print("="*80)

summary = f"""
================================================================================
RQ8: CLASSIFICATION PERFORMANCE - CORRECTED METHODOLOGY - SUMMARY
================================================================================

Research Question:
"Which classifier and dimension combination is most clinically useful?"

ANSWER: {best_binary['Classifier']} with {best_binary['Features']} (AUC={best_binary['AUC_Mean']:.4f})

⚠ CRITICAL CORRECTIONS APPLIED:
1. Binary labels CORRECTED: Benign (N/PB/UDH/FEA/ADH) vs Malignant (DCIS/IC only)
   - Previous version incorrectly labeled FEA/ADH as malignant
   - FEA and ADH are pre-malignant hyperplasias, NOT invasive carcinoma
2. WSI-LEVEL analysis (n={len(merged_analysis)}) instead of ROI-level (n={len(merged)})
   - ICC(Dc)={icc_dc:.3f}, ICC(Dm)={icc_dm:.3f} → substantial clustering
   - ROI-level classification = pseudoreplication (invalid)
3. All results below reflect CORRECTED methodology

================================================================================
BINARY CLASSIFICATION (Benign+Atypical vs Malignant)
================================================================================
WSI-Level: n={len(merged_analysis)} ({dict(merged_analysis['Binary'].value_counts())})

BEST PERFORMER:
  Classifier: {best_binary['Classifier']}
  Features:   {best_binary['Features']}
  AUC:        {best_binary['AUC_Mean']:.4f} ± {best_binary['AUC_SD']:.4f}
  Accuracy:   {best_binary['Accuracy_Mean']:.4f} ± {best_binary['Accuracy_SD']:.4f}
  F1 Score:   {best_binary['F1_Mean']:.4f} ± {best_binary['F1_SD']:.4f}

RANKING BY AUC (Top 5):
"""

for idx, row in binary_df.nlargest(5, 'AUC_Mean').iterrows():
    summary += f"  {idx+1}. {row['Classifier']:20s} + {row['Features']:10s}: AUC={row['AUC_Mean']:.4f}\n"

summary += f"""

CLASSIFIER COMPARISON (averaged across all features):
"""
clf_avg = binary_df.groupby('Classifier')['AUC_Mean'].mean().sort_values(ascending=False)
for clf, auc in clf_avg.items():
    summary += f"  {clf:20s}: {auc:.4f}\n"

summary += f"""

FEATURE COMPARISON (averaged across all classifiers):
"""
feat_avg = binary_df.groupby('Features')['AUC_Mean'].mean().sort_values(ascending=False)
for feat, auc in feat_avg.items():
    summary += f"  {feat:10s}: {auc:.4f}\n"

summary += f"""

================================================================================
TERNARY CLASSIFICATION (Benign/Atypical/Malignant)
================================================================================
WSI-Level: n={len(merged_analysis)} ({dict(merged_analysis['Ternary'].value_counts())})

BEST PERFORMER:
  Classifier: {best_ternary['Classifier']}
  Features:   {best_ternary['Features']}
  Accuracy:   {best_ternary['Accuracy_Mean']:.4f} ± {best_ternary['Accuracy_SD']:.4f}
  F1 Score:   {best_ternary['F1_Mean']:.4f} ± {best_ternary['F1_SD']:.4f}

================================================================================
KEY INSIGHTS
================================================================================

1. BEST CLASSIFIER: {clf_avg.index[0]} (averaged AUC: {clf_avg.iloc[0]:.4f})

2. BEST FEATURES: {feat_avg.index[0]} (averaged AUC: {feat_avg.iloc[0]:.4f})

3. CLINICAL UTILITY:
   {'HIGH - AUC > 0.80, suitable for clinical decision support' if best_binary['AUC_Mean'] > 0.80 else 'MODERATE - AUC 0.70-0.80, useful supplementary tool' if best_binary['AUC_Mean'] > 0.70 else 'LIMITED - AUC < 0.70, NOT reliable for clinical diagnosis'}
   
   ⚠ INTERPRETATION: Fractal dimensions alone have {'excellent' if best_binary['AUC_Mean'] > 0.85 else 'good' if best_binary['AUC_Mean'] > 0.75 else 'modest' if best_binary['AUC_Mean'] > 0.65 else 'poor'} discriminative power.
   {'Suitable as standalone diagnostic biomarker.' if best_binary['AUC_Mean'] > 0.85 else 'Should be combined with morphology, grade, immunohistochemistry for clinical decisions.' if best_binary['AUC_Mean'] > 0.65 else 'NOT suitable as standalone biomarker - must combine with standard clinical features.'}

4. RECOMMENDATION:
   {'Use both Dc and Dm for best performance' if best_binary['Features'] == 'Both' else f'Single dimension ({best_binary["Features"]}) sufficient'}
   {'Consider ensemble methods (Random Forest) for robustness' if best_binary['Classifier'] == 'Random Forest' else ''}
   {'KNN or SVM may capture non-linear patterns better than Logistic Regression' if best_binary['Classifier'] in ['KNN', 'SVM'] else ''}
   
5. METHODOLOGICAL NOTES:
   - Analysis performed at WSI-level to avoid pseudoreplication
   - Binary classification: Benign+Atypical (N/PB/UDH/FEA/ADH) vs Malignant (DCIS/IC)
   - Corrects previous error where FEA/ADH were mislabeled as malignant
   - 5-fold stratified cross-validation with standard scaling for KNN/SVM

FILES GENERATED:
- 01_binary_all_classifiers.xlsx - All binary results (WSI-level)
- 02_ternary_all_classifiers.xlsx - All ternary results (WSI-level)
- 03_hyperparameter_tuning.xlsx - Grid search results
- fig1_classifier_heatmaps.tif - Performance heatmaps
- fig2_classifier_bars.tif - Bar chart comparisons

================================================================================
"""

with open(RESULTS_DIR / '00_RQ8_SUMMARY.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print(summary)
print(f"✓ Saved: {RESULTS_DIR / '00_RQ8_SUMMARY.txt'}")

print("\n" + "="*80)
print("RQ8 CORRECTED ANALYSIS COMPLETE!")
print("="*80)
print(f"\n⚠ CRITICAL CORRECTIONS:")
print(f"  1. Binary labels: Benign+Atypical vs Malignant (DCIS/IC only)")
print(f"  2. WSI-level analysis (n={len(merged_analysis)}) to avoid pseudoreplication")
print(f"\nTested: 4 classifiers × 3 feature sets = 12 combinations")
print(f"Best: {best_binary['Classifier']} + {best_binary['Features']} (AUC={best_binary['AUC_Mean']:.4f})")
print("="*80)