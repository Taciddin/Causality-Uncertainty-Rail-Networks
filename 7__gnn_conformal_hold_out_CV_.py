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
from mapie.regression import MapieRegressor

# --- Step 0: Data Loading and Preparation ---
def load_and_prepare_data(file_path='data.xlsx'):
    """Loads the data, translates column names to English, and cleans it."""
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

# --- Step 1: Feature Extraction with GNN (Creating Embeddings) ---
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
        if (epoch + 1) % 50 == 0:
            print(f"  GNN Training Epoch: {epoch+1}/{epochs}, Reconstruction Loss: {loss.item():.4f}")
    
    gnn_model.eval()
    with torch.no_grad():
        final_embeddings = gnn_model(graph_data).numpy()
    
    print("✓ GNN-based features (embeddings) created successfully.")
    return final_embeddings

# --- Step 2: Performance and Enhanced Plotting Functions ---

def evaluate_predictions(y_true, y_pred, y_pis, alpha):
    """Evaluates the performance of both point and interval predictions."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    coverage = ((y_true >= y_pis[:, 0, 0]) & (y_true <= y_pis[:, 1, 0])).mean()
    width = (y_pis[:, 1, 0] - y_pis[:, 0, 0]).mean()
    
    print("\n--- Model Performance Metrics ---")
    print("  Point Prediction Performance:")
    print(f"    - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R-squared: {r2:.3f}")
    print("  Prediction Interval Performance:")
    print(f"    - Target Coverage: {1 - alpha:.2f}")
    print(f"    - Actual Coverage: {coverage:.3f}")
    print(f"    - Average Interval Width: {width:.2f}")
    return coverage

def plot_prediction_intervals(y_true, y_pred, y_pis, coverage, title, filename):
    """Plots a visually enhanced graph showing prediction intervals and coverage status."""
    indices = np.argsort(y_true)
    y_true_sorted, y_pred_sorted, y_pis_sorted = y_true[indices], y_pred[indices], y_pis[indices]
    is_covered = (y_true_sorted >= y_pis_sorted[:, 0, 0]) & (y_true_sorted <= y_pis_sorted[:, 1, 0])

    plt.style.use('seaborn-v0_8-whitegrid') 
    plt.figure(figsize=(18, 10)) 

   
    covered_color = '#007ACC' # Vivid Blue
    not_covered_color = '#D62728' # Vivid Red
    actual_color = 'black'
    
   
    plt.errorbar(
        np.arange(len(y_true_sorted))[is_covered], y_pred_sorted[is_covered], 
        yerr=(y_pred_sorted[is_covered] - y_pis_sorted[is_covered, 0, 0], y_pis_sorted[is_covered, 1, 0] - y_pred_sorted[is_covered]),
        fmt='o', color=covered_color, ecolor=covered_color, elinewidth=2.5, 
        markersize=6, alpha=0.7, label='Covered' 
    )
   
    plt.errorbar(
        np.arange(len(y_true_sorted))[~is_covered], y_pred_sorted[~is_covered], 
        yerr=(y_pred_sorted[~is_covered] - y_pis_sorted[~is_covered, 0, 0], y_pis_sorted[~is_covered, 1, 0] - y_pred_sorted[~is_covered]),
        fmt='o', color=not_covered_color, ecolor=not_covered_color, elinewidth=2.5, 
        markersize=8, alpha=1.0, label='Not Covered' 
    )
    # Actual Values
    plt.plot(np.arange(len(y_true_sorted)), y_true_sorted, 'o', color=actual_color, 
             markersize=5, label='Actual Value') 

   
    plt.title(f"{title}\nActual Coverage: {coverage:.3f}", fontsize=32, weight='bold')
    plt.xlabel("Test Samples (Sorted by Actual Value)", fontsize=24)
    plt.ylabel("Vehicle_KM", fontsize=24)
    plt.xticks(fontsize=18) 
    plt.yticks(fontsize=18)
    
    plt.legend(fontsize=20) 
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight') # Margins optimized with bbox_inches='tight'
    print(f"✓ Critical plot for publication saved (visually enhanced): {filename}")
    plt.show()

def plot_feature_importance(estimators_list, feature_names, title, filename, num_features=15):
    """Plots a visually enhanced graph of average feature importance from multiple models."""
    importances = [est.feature_importances_ for est in estimators_list]
    avg_importances = pd.Series(np.mean(importances, axis=0), index=feature_names)
    top_importances = avg_importances.sort_values(ascending=False).head(num_features)
    
    xlabel = f'Average Importance Score (from {len(estimators_list)} model(s))'

    plt.style.use('seaborn-v0_8-whitegrid') 
    plt.figure(figsize=(14, 12)) 
    
   
    sns.barplot(x=top_importances, y=top_importances.index, palette='plasma')
    
   
    plt.title(title, fontsize=32, weight='bold')
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel('Top Features (Original & GNN-Generated)', fontsize=24)
    plt.xticks(fontsize=18) 
    plt.yticks(fontsize=18)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance plot saved (visually enhanced): {filename}")
    plt.show()
    
# --- Step 3: Main Analysis Function ---
def run_conformal_analysis(df, gnn_embeddings):
    """Runs the Conformal Prediction analysis with GNN-enhanced features."""
    print("\n--- STEP 2: Conformal Prediction with GNN-enhanced Features ---")
    
    y = df['Vehicle_KM'].values
    original_features = df.select_dtypes(include=np.number).drop(columns=['Vehicle_KM', 'Year', 'Period'], errors='ignore')
    if 'Line_ID' in original_features.columns: original_features = original_features.drop(columns=['Line_ID'])
    
    gnn_features_df = pd.DataFrame(gnn_embeddings, index=original_features.index,
                                   columns=[f'gnn_feat_{i}' for i in range(gnn_embeddings.shape[1])])
    
    X = pd.concat([original_features, gnn_features_df], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nData split: {len(X_train)} training samples, {len(X_test)} test samples.")
    print(f"Total features used for prediction: {X.shape[1]} ({len(original_features.columns)} original + {len(gnn_features_df.columns)} GNN)")

    base_regressor = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
    alpha = 0.05

    # ==========================================================
    # METHOD 1: HOLD-OUT (SPLIT-CONFORMAL) APPROACH
    # ==========================================================
    print("\n" + "="*50 + "\nMETHOD 1: HOLD-OUT (SPLIT-CONFORMAL)\n" + "="*50)
    
    mapie_split = MapieRegressor(base_regressor, cv="split", random_state=42)
    mapie_split.fit(X_train, y_train)
    y_pred_split, y_pis_split = mapie_split.predict(X_test, alpha=alpha)
    
    print("\n[Hold-Out] Evaluating predictions on the test set...")
    coverage_split = evaluate_predictions(y_test, y_pred_split, y_pis_split, alpha)
    
    # ==========================================================
    # METHOD 2: 5-FOLD CROSS-VALIDATION (CV+) - MORE ROBUST
    # ==========================================================
    print("\n\n" + "="*50 + "\nMETHOD 2: 5-FOLD CROSS-VALIDATION (CV+)\n" + "="*50)
    
    mapie_cv = MapieRegressor(base_regressor, cv=5)
    mapie_cv.fit(X_train, y_train)
    y_pred_cv, y_pis_cv = mapie_cv.predict(X_test, alpha=alpha)

    print("\n[CV+] Evaluating predictions on the test set...")
    coverage_cv = evaluate_predictions(y_test, y_pred_cv, y_pis_cv, alpha)
    
    # --- INTERPRETATION AND VISUALIZATIONS (Based on the more robust CV+ method only) ---
    print("\n--- VISUALIZATIONS FOR PUBLICATION (Based on the more robust CV+ method) ---")
    
    plot_prediction_intervals(
        y_test, y_pred_cv, y_pis_cv, coverage_cv,
        'CV+: GNN-Enhanced Conformal Prediction Intervals',
        'cv_conformal_intervals_GNN_enhanced.png'
    )
    
    plot_feature_importance(
        mapie_cv.estimator_.estimators_, X.columns,
        'CV+: Top 15 Features Influencing Predictions',
        'cv_feature_importance_GNN_enhanced.png'
    )
    
    print("\n\n>> FINAL CONCLUSION:")
    print("Comparing the two methods, the 5-Fold CV+ approach is generally more robust as it utilizes the training data more efficiently.")
    print(f"The CV+ model achieved an actual coverage of {coverage_cv:.3f} (target: {1-alpha:.2f}) with an average interval width of {(y_pis_cv[:, 1, 0] - y_pis_cv[:, 0, 0]).mean():.2f}.")
    print("This provides a reliable uncertainty estimate for predictions enhanced by GNN-derived structural features.")

# --- MAIN PROGRAM ---
if __name__ == '__main__':
    df = load_and_prepare_data('data.xlsx')
    if df is not None:
        gnn_embeddings = get_gnn_embeddings(df)
        run_conformal_analysis(df, gnn_embeddings)
