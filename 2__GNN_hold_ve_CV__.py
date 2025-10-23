import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- Step 1: Global Graph Creation ---
def create_global_graph(file_path='data.xlsx', k_neighbors=10):
    try:
        df = pd.read_excel(file_path)
        df.rename(columns={'Yolcu Sayısı': 'Passenger_Count', 'Araç KM': 'Vehicle_KM'}, inplace=True)
    except FileNotFoundError:
        print(f"ERROR: '{file_path}' not found.")
        return None, None
    
    for col in ['Passenger_Count', 'Vehicle_KM']:
        if col in df.columns:
            df[col] = np.log1p(df[col])
            
    features_df = df.drop(columns=['Vehicle_KM', 'Line_ID', 'Yıl', 'Periyot'], errors='ignore')
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    adjacency_matrix_sparse = kneighbors_graph(features_scaled, k_neighbors, mode='connectivity', include_self=False)
    edge_index, _ = from_scipy_sparse_matrix(adjacency_matrix_sparse)
    x_tensor = torch.tensor(features_scaled, dtype=torch.float)
    y_tensor = torch.tensor(df['Vehicle_KM'].values, dtype=torch.float).unsqueeze(1)
    graph_data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
    print(f"Data loaded. A global graph with {graph_data.num_nodes} nodes has been created.")
    return graph_data, df

# --- Step 2: GNN Model Definition ---
class ImprovedGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels=64):
        super(ImprovedGNN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels // 2)
        self.out = nn.Linear(hidden_channels // 2, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index); x = self.bn1(x); x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index); x = self.bn2(x); x = x.relu()
        x = self.conv3(x, edge_index); x = x.relu()
        return self.out(x)

# --- Step 3: Model Training Function ---
def train_model(model, data, epochs=1000, verbose=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = torch.sqrt(criterion(out[data.train_mask], data.y[data.train_mask]))
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 200 == 0:
            print(f'  Epoch: {epoch+1:04d}, Training Loss (RMSE): {loss.item():.4f}')
    return model

# --- Step 4: Performance Evaluation Function ---
def evaluate_performance(model, data):
    model.eval()
    with torch.no_grad():
        all_predictions = model(data.x, data.edge_index)
    y_true_log = data.y.cpu().numpy().flatten()
    y_pred_log = all_predictions.cpu().numpy().flatten()
    test_mask = data.test_mask.cpu().numpy()
    y_true_original = np.expm1(y_true_log[test_mask])
    y_pred_original = np.expm1(y_pred_log[test_mask])
    rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
    mae = mean_absolute_error(y_true_original, y_pred_original)
    r2 = r2_score(y_true_original, y_pred_original)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}, y_true_original, y_pred_original

# --- Step 5: Plotting Function (REVISED FOR COLORS AND FONTS) ---
def plot_actual_vs_predicted(y_true, y_pred, title, filename):
    """
    Plots a publication-quality, high-resolution Actual vs. Predicted graph 
    with more distinct colors and doubled font sizes.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 10))

    # Scatter plot with new distinct colors
    sns.scatterplot(
        x=y_true, y=y_pred, alpha=0.7, edgecolor='black', s=100, color="#007ACC", ax=ax
    )
    
    # Perfect prediction line (y=x) with a vibrant, contrasting color
    perfect_line = np.linspace(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()), 100)
    ax.plot(perfect_line, perfect_line, color='#D95319', linestyle='--', linewidth=3, label='Perfect Prediction')
    
    # Title and labels with doubled font sizes
    ax.set_title(title, fontsize=48, weight='bold', pad=25)
    ax.set_xlabel('Actual Values (Vehicle KM)', fontsize=36, weight='bold', labelpad=20)
    ax.set_ylabel('Predicted Values (Vehicle KM)', fontsize=36, weight='bold', labelpad=20)
    
    # Increase tick label font size (doubled)
    ax.tick_params(axis='both', which='major', labelsize=32)
    
    # Add metrics to the plot with doubled font sizes
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'RMSE: {rmse:,.0f}\n$R^2$:      {r2:.4f}',
            transform=ax.transAxes, fontsize=36, weight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='lightgray', alpha=0.7))
    
    ax.legend(fontsize=32)
    plt.tight_layout()
    # Save with high resolution
    plt.savefig(filename, dpi=300)
    print(f"High-resolution graph saved as '{filename}'")
    plt.show()

# --- Main Program Execution ---
if __name__ == '__main__':
    graph_data, original_df = create_global_graph('data.xlsx', k_neighbors=10)
    if graph_data:
        num_nodes = graph_data.num_nodes
        
        # ======================================================================
        # METHOD 1: HOLD-OUT
        # ======================================================================
        print("\n" + "="*50 + "\n          METHOD 1: HOLD-OUT VALIDATION\n" + "="*50)
        indices = torch.randperm(num_nodes)
        train_size = int(num_nodes * 0.8)
        train_idx, test_idx = indices[:train_size], indices[train_size:]
        graph_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        graph_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        graph_data.train_mask[train_idx] = True
        graph_data.test_mask[test_idx] = True
        print(f"Data split into {len(train_idx)} training and {len(test_idx)} test nodes.")
        
        gnn_model_holdout = ImprovedGNN(num_features=graph_data.num_node_features, hidden_channels=128)
        print("\nTraining Hold-Out model...")
        train_model(gnn_model_holdout, graph_data, epochs=1000, verbose=True)
        print("Training complete.")
        
        holdout_results, y_true_holdout, y_pred_holdout = evaluate_performance(gnn_model_holdout, graph_data)
        print("\n--- HOLD-OUT TEST RESULTS ---")
        print(f"RMSE: {holdout_results['RMSE']:.2f}, MAE: {holdout_results['MAE']:.2f}, R-squared: {holdout_results['R2']:.4f}")
        plot_actual_vs_predicted(y_true_holdout, y_pred_holdout, 
                                 'Actual vs. Predicted Values (Hold-Out)', 
                                 'holdout_results_plot.png')
        
        # ======================================================================
        # METHOD 2: 5-FOLD CROSS-VALIDATION
        # ======================================================================
        print("\n" + "="*50 + "\n      METHOD 2: 5-FOLD CROSS-VALIDATION (CV)\n" + "="*50)
        fold_metrics = []
        all_true_values, all_pred_values = [], []
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        node_indices = np.arange(num_nodes)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(node_indices)):
            print(f"\n--- FOLD {fold + 1}/5 ---")
            graph_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool); graph_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            graph_data.train_mask[train_idx] = True; graph_data.test_mask[test_idx] = True
            gnn_model_cv = ImprovedGNN(num_features=graph_data.num_node_features, hidden_channels=128)
            print("Training CV model...")
            train_model(gnn_model_cv, graph_data, epochs=1000, verbose=False)
            
            results, y_true_fold, y_pred_fold = evaluate_performance(gnn_model_cv, graph_data)
            fold_metrics.append(results)
            all_true_values.extend(y_true_fold); all_pred_values.extend(y_pred_fold)
            print(f"Fold {fold + 1} Results -> RMSE: {results['RMSE']:.2f}, MAE: {results['MAE']:.2f}, R2: {results['R2']:.4f}")

        # --- Final CV Results and Combined Plot ---
        print("\n\n--- CROSS-VALIDATION SUMMARY RESULTS ---")
        results_df = pd.DataFrame(fold_metrics)
        mean_results = results_df.mean(); std_results = results_df.std()
        print(f"Average RMSE: {mean_results['RMSE']:.2f} (± {std_results['RMSE']:.2f})")
        print(f"Average MAE : {mean_results['MAE']:.2f} (± {std_results['MAE']:.2f})")
        print(f"Average R^2 : {mean_results['R2']:.4f} (± {std_results['R2']:.4f})")
        
        plot_actual_vs_predicted(np.array(all_true_values), np.array(all_pred_values), 
                                 'Actual vs. Predicted Values (5-Fold CV Combined)', 
                                 'cross_validation_results_plot.png')
