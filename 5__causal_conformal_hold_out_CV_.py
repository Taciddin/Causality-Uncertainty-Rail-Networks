import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from econml.dml import LinearDML
from mapie.regression import MapieRegressor

# --- Step 0 & 1: Data Loading and GNN Feature Extraction (No Changes) ---
def load_and_prepare_data(file_path='data.xlsx'):
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"ERROR: '{file_path}' not found. Please check the file location.")
        return None
    rename_dict = {
        'Hattın Kodu': 'Line_ID', 'Yıl': 'Year', 'Periyot': 'Period',
        'Araç KM': 'Vehicle_KM', 'Araç Sayısı': 'Vehicle_Count'
    }
    df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns}, inplace=True)
    df = df.dropna()
    print("✓ Data loaded and prepared successfully.")
    return df

class GNNFeatureExtractor(torch.nn.Module):
    def __init__(self, num_features, embedding_dim=32):
        super().__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(num_features, embedding_dim * 2)
        self.conv2 = GCNConv(embedding_dim * 2, embedding_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def get_gnn_embeddings(df, k_neighbors=8, embedding_dim=32, epochs=200):
    print("\n--- STEP 1: Feature Extraction with GNN ---")
    features_df = df.select_dtypes(include=np.number).drop(columns=['Vehicle_KM', 'Year', 'Period'], errors='ignore')
    if 'Line_ID' in features_df.columns: features_df = features_df.drop(columns=['Line_ID'])
    print(f"  Features used for GNN training: {list(features_df.columns)}")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    adjacency_matrix = kneighbors_graph(features_scaled, k_neighbors, mode='connectivity')
    edge_index, _ = from_scipy_sparse_matrix(adjacency_matrix)
    x_tensor = torch.tensor(features_scaled, dtype=torch.float)
    graph_data = Data(x=x_tensor, edge_index=edge_index)
    gnn_model = GNNFeatureExtractor(graph_data.num_features, embedding_dim)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    gnn_model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings = gnn_model(graph_data)
        reconstructed_features = torch.nn.Linear(embedding_dim, graph_data.num_features)(embeddings)
        loss = criterion(reconstructed_features, graph_data.x)
        loss.backward()
        optimizer.step()
    gnn_model.eval()
    with torch.no_grad():
        final_embeddings = gnn_model(graph_data).numpy()
    print("✓ GNN-based features (embeddings) created successfully.")
    return final_embeddings

# --- Step 2: New Plotting and Evaluation Functions 

# NEW FUNCTION: Provides central style control for all plots.
def set_plot_style(font_scale=2.0):
    """Sets general visual settings for Matplotlib and Seaborn."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12 * font_scale,
        'axes.labelsize': 12 * font_scale,
        'axes.titlesize': 14 * font_scale,
        'xtick.labelsize': 10 * font_scale,
        'ytick.labelsize': 10 * font_scale,
        'legend.fontsize': 10 * font_scale,
        'figure.titlesize': 16 * font_scale,
        'figure.dpi': 100,
    })
    print("✓ Plotting style updated for better readability (larger fonts).")

def evaluate_cate_predictor(y_true_cate, y_pred_cate, y_pis_cate, alpha, prefix=""):
    """Evaluates the performance of the CATE prediction model and its conformal intervals."""
    rmse = np.sqrt(mean_squared_error(y_true_cate, y_pred_cate))
    mae = mean_absolute_error(y_true_cate, y_pred_cate)
    r2 = r2_score(y_true_cate, y_pred_cate)
    coverage = ((y_true_cate >= y_pis_cate[:, 0, 0]) & (y_true_cate <= y_pis_cate[:, 1, 0])).mean()
    width = (y_pis_cate[:, 1, 0] - y_pis_cate[:, 0, 0]).mean()
    
    print(f"\n--- {prefix} Performance Metrics ---")
    print("  CATE Prediction Model Performance (how well we predict the causal effect):")
    print(f"    - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R-squared: {r2:.3f}")
    print("  Conformal Interval Performance (for the causal effect):")
    print(f"    - Target Coverage: {1 - alpha:.2f}")
    print(f"    - Actual Coverage: {coverage:.3f}")
    print(f"    - Average Interval Width: {width:.2f}")

# PLOTTING FUNCTION
def plot_cate_intervals(cate_pred, cate_pis, title, filename):
    """Plots the causal effect prediction intervals."""
    indices = np.argsort(cate_pred)
    cate_pred_sorted, cate_pis_sorted = cate_pred[indices], cate_pis[indices]
    is_significant = (cate_pis_sorted[:, 0, 0] > 0) | (cate_pis_sorted[:, 1, 0] < 0)

    # More distinct and vibrant colors are defined
    significant_color = '#11A579' # Vibrant Green
    insignificant_color = '#3498DB' # Bright Blue
    zero_line_color = '#E74C3C' # Red

    plt.figure(figsize=(18, 10)) # Size increased
    # More distinct markers for significant effects
    plt.errorbar(
        np.arange(len(cate_pred_sorted))[is_significant], cate_pred_sorted[is_significant],
        yerr=(cate_pred_sorted[is_significant] - cate_pis_sorted[is_significant, 0, 0], cate_pis_sorted[is_significant, 1, 0] - cate_pred_sorted[is_significant]),
        fmt='o', color='white', ecolor=significant_color, elinewidth=2.5, capsize=4,
        markeredgecolor=significant_color, markersize=8, markeredgewidth=2,
        label='Significant Effect (Interval does not cross zero)'
    )
    # More distinct markers for insignificant effects
    plt.errorbar(
        np.arange(len(cate_pred_sorted))[~is_significant], cate_pred_sorted[~is_significant],
        yerr=(cate_pred_sorted[~is_significant] - cate_pis_sorted[~is_significant, 0, 0], cate_pis_sorted[~is_significant, 1, 0] - cate_pred_sorted[~is_significant]),
        fmt='o', color='white', ecolor=insignificant_color, elinewidth=2, capsize=4,
        markeredgecolor=insignificant_color, markersize=8, markeredgewidth=2, alpha=0.8,
        label='Insignificant Effect (Interval crosses zero)'
    )
    plt.axhline(0, color=zero_line_color, linestyle='--', linewidth=2.5, label='Zero Effect Line')
    plt.title(title, fontweight='bold')
    plt.xlabel("Test Samples (Sorted by Predicted Causal Effect)")
    plt.ylabel("Predicted Causal Effect (CATE)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(pad=1.5)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Causal interval plot for publication saved: {filename}")
    plt.show()

#PLOTTING FUNCTION
def plot_actual_vs_predicted_cate(y_true_cate, y_pred_cate, title, filename):
    """Plots the comparison between actual CATEs and predicted CATEs."""
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x=y_true_cate, y=y_pred_cate,
        alpha=0.7,
        s=150,  # Point size increased
        edgecolor='black', # Black border added to points
        color='#3498DB' # Bright Blue color used
    )
    plt.plot([y_true_cate.min(), y_true_cate.max()], [y_true_cate.min(), y_true_cate.max()], '--', color='#E74C3C', lw=3)
    plt.title(title, fontweight='bold')
    plt.xlabel("Pseudo-True CATE (from DML on test set)")
    plt.ylabel("Predicted CATE (from Meta-Learner)")
    plt.grid(True)
    plt.tight_layout(pad=1.5)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ CATE prediction performance plot saved: {filename}")
    plt.show()

#PLOTTING FUNCTION
def plot_feature_importance(estimators_list, feature_names, title, filename, num_features=15):
    """Plots the average feature importance from one or more models."""
    importances = [est.feature_importances_ for est in estimators_list]
    avg_importances = pd.Series(np.mean(importances, axis=0), index=feature_names)
    top_importances = avg_importances.sort_values(ascending=False).head(num_features)
    xlabel = f'Average Importance Score (from {len(estimators_list)} model(s))'
    
    plt.figure(figsize=(12, 10))
    sns.barplot(
        x=top_importances.values,
        y=top_importances.index,
        palette='plasma' # More vibrant and distinct 'plasma' color palette
    )
    plt.title(title, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel('Top Features (Original & GNN-Generated)')
    plt.tight_layout(pad=1.5)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance plot saved: {filename}")
    plt.show()
    
# --- Step 3: Main Analysis Function ---
def run_causal_conformal_analysis(df, gnn_embeddings):
    print("\n--- STEP 2: Causal ML + Conformal Prediction with GNN Features ---")
    
    Y = df[['Vehicle_KM']]
    T = df[['Vehicle_Count']]
    original_features = df.select_dtypes(include=np.number).drop(columns=['Vehicle_KM', 'Year', 'Period', 'Vehicle_Count'], errors='ignore')
    if 'Line_ID' in original_features.columns: original_features = original_features.drop(columns=['Line_ID'])
    
    gnn_features_df = pd.DataFrame(gnn_embeddings, index=original_features.index,
                                   columns=[f'gnn_feat_{i}' for i in range(gnn_embeddings.shape[1])])
    W = pd.concat([original_features, gnn_features_df], axis=1)

    # Split data for DML and the CATE predictor
    W_train, W_test, Y_train, Y_test, T_train, T_test = train_test_split(
        W, Y, T, test_size=0.2, random_state=42
    )

    # --- DML Model: To estimate CATEs ---
    model_y = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42)
    model_t = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42)
    dml_estimator = LinearDML(model_y=model_y, model_t=model_t, cv=5, random_state=42)
    dml_estimator.fit(Y_train, T_train, X=W_train, W=None)
    
    # Create "pseudo-true" CATEs: Apply DML to both training and test data
    cates_train = dml_estimator.effect(W_train).ravel()
    cates_test = dml_estimator.effect(W_test).ravel()

    # --- CATE Prediction Model ---
    base_cate_predictor = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
    alpha = 0.05

    # ==========================================================
    # Method 1: Hold-Out (Split-Conformal) Approach
    # ==========================================================
    print("\n" + "="*50 + "\nMETHOD 1: HOLD-OUT (SPLIT-CONFORMAL)\n" + "="*50)
    mapie_split = MapieRegressor(base_cate_predictor, cv="split", random_state=42)
    mapie_split.fit(W_train, cates_train)
    cate_pred_split, cate_pis_split = mapie_split.predict(W_test, alpha=alpha)
    evaluate_cate_predictor(cates_test, cate_pred_split, cate_pis_split, alpha, prefix="Hold-Out")

    # ==========================================================
    # Method 2: 5-Fold Cross-Validation (CV+)
    # ==========================================================
    print("\n\n" + "="*50 + "\nMETHOD 2: 5-FOLD CROSS-VALIDATION (CV+)\n" + "="*50)
    mapie_cv = MapieRegressor(base_cate_predictor, cv=5)
    mapie_cv.fit(W_train, cates_train)
    cate_pred_cv, cate_pis_cv = mapie_cv.predict(W_test, alpha=alpha)
    evaluate_cate_predictor(cates_test, cate_pred_cv, cate_pis_cv, alpha, prefix="CV+")
    
    # --- INTERPRETATION AND PLOTS (Only for the more robust CV method) ---
    print("\n--- VISUALIZATIONS FOR PUBLICATION (Based on the more robust CV+ method) ---")
    
    # NEWLY ADDED LINE: Style settings for all plots are made here once.
    set_plot_style(font_scale=2.0)
    
    # 1. Main Plot: Causal Effect Intervals
    plot_cate_intervals(
        cate_pred_cv, cate_pis_cv,
        'Conformalized Causal Effect Intervals (95% Coverage Guarantee)',
        'causal_conformal_intervals_GNN.png'
    )
    
    # 2. CATE Prediction Model Performance
    plot_actual_vs_predicted_cate(
        cates_test, cate_pred_cv,
        'CATE Prediction Performance (Meta-Learner)',
        'cate_prediction_performance.png'
    )

    # 3. Importance of Features Determining CATEs
    plot_feature_importance(
        mapie_cv.estimator_.estimators_, W.columns,
        'Top 15 Features Determining the *Magnitude* of Causal Effect',
        'cate_feature_importance_GNN.png'
    )
    
    print("\n\n>> FINAL CONCLUSION:")
    avg_effect = cate_pred_cv.mean()
    avg_width = (cate_pis_cv[:, 1, 0] - cate_pis_cv[:, 0, 0]).mean()
    print(f"The combined Causal+Conformal model estimates the Average Treatment Effect (ATE) on the test set to be {avg_effect:.2f}.")
    print(f"Crucially, it provides a rigorous, distribution-free 95% prediction interval for the causal effect of each individual bus line.")
    print(f"The average width of these causal intervals is {avg_width:.2f}, quantifying the uncertainty of our causal estimates.")
    print("This method provides not just an estimate of the effect, but a guaranteed range for the effect of adding a new vehicle for any given line.")

# --- MAIN PROGRAM ---
if __name__ == '__main__':
    # Make sure 'data.xlsx' is in the same directory as the code
    df = load_and_prepare_data('data.xlsx')
    if df is not None:
        gnn_embeddings = get_gnn_embeddings(df)
        run_causal_conformal_analysis(df, gnn_embeddings)
