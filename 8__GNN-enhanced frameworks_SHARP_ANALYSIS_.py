import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import traceback
import time 
import shap
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix, to_networkx
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Step 0: Data Loading and Preparation ---
def load_and_prepare_data(file_path='data.xlsx'):
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"ERROR: '{file_path}' not found.")
        return None
    
    rename_dict = {
        'Hattın Kodu': 'Line_ID', 'Yıl': 'Year', 'Periyot': 'Period',
        'Araç KM': 'Vehicle_KM', 'Araç Sayısı': 'Vehicle_Count'
    }
    df.rename(columns=rename_dict, inplace=True)
    df = df.dropna().reset_index(drop=True)
    print("✓ Data successfully loaded and prepared.")
    return df

# --- Step 1: Feature Extraction with GNN ---
class GNNFeatureExtractor(torch.nn.Module):
    def __init__(self, num_features, embedding_dim=16): 
        super().__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(num_features, embedding_dim)
        self.out = torch.nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        return self.out(x)

def get_gnn_embeddings_and_topology(df, k_neighbors=5, embedding_dim=16, epochs=100): 
    print("\n--- STEP 1: GNN and Topology (Accelerated) ---")
    features_df = df.drop(columns=['Vehicle_KM', 'Line_ID', 'Year', 'Period', 'Vehicle_Count'], errors='ignore')
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
        # Simplified loss
        loss = criterion(embeddings, torch.zeros_like(embeddings)) 
        loss.backward()
        optimizer.step()
    
    gnn_model.eval()
    with torch.no_grad():
        final_embeddings = gnn_model(graph_data).numpy()
    print("✓ GNN embeddings created.")

    nx_graph = to_networkx(graph_data, to_undirected=True)
    topology_df = pd.DataFrame(index=df.index)
    topology_df['degree_centrality'] = pd.Series(nx.degree_centrality(nx_graph))
    topology_df['betweenness_centrality'] = pd.Series(nx.betweenness_centrality(nx_graph))
    topology_df['clustering_coefficient'] = pd.Series(nx.clustering(nx_graph))
    print("✓ Topological metrics calculated.")
    
    return final_embeddings, topology_df

# --- Step 2: INTERPRETATION-FOCUSED MAIN FUNCTION ---

def generate_interpretability_evidence(df, gnn_embeddings, topology_df):
    print("\n--- STEP 2: Generating Interpretability Evidence ---")
    
    Y = df['Vehicle_KM']
    original_features = df.drop(columns=['Vehicle_KM', 'Line_ID', 'Year', 'Period'], errors='ignore')
    gnn_features_df = pd.DataFrame(gnn_embeddings, index=original_features.index,
                                   columns=[f'gnn_feat_{i}' for i in range(gnn_embeddings.shape[1])])
    X = pd.concat([original_features, gnn_features_df], axis=1)

    # --- EVIDENCE 1: CORRELATION TABLE 
    temp_model = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, Y)
    importances = pd.Series(temp_model.feature_importances_, index=X.columns)
    top_gnn_features = importances[importances.index.str.startswith('gnn_feat_')].nlargest(5).index
    correlation_df = pd.concat([gnn_features_df[top_gnn_features], topology_df], axis=1).corr()
    correlation_subset = correlation_df.loc[top_gnn_features, topology_df.columns]
    
    print("\n" + "="*60)
    print("  EVIDENCE 1: GNN FEATURES AND TOPOLOGY CORRELATION TABLE")
    print("="*60)
    try:
        print(correlation_subset.to_markdown(floatfmt=".2f"))
    except ImportError:
        print("\n(Note: 'tabulate' library is not installed, the table is displayed in standard format.)")
        print(correlation_subset.round(2))
    print("="*60)
    
    # --- EVIDENCE 2: SHAP ANALYSIS 
    print("\n" + "="*60)
    print("  EVIDENCE 2: GNN INTERPRETABILITY WITH SHAP VALUES")
    print("="*60)  
    print("  Training a simpler Random Forest model for SHAP analysis...")
    final_model_for_shap = RandomForestRegressor(
        n_estimators=50,       
        max_depth=10,          
        random_state=42,
        n_jobs=-1              
    )
    final_model_for_shap.fit(X, Y)
    print("✓ Final model trained.")
    

    print("\n  Using a small subset of data to test SHAP...")
    if len(X) > 100:
        X_sample = X.sample(n=100, random_state=42)
    else:
        X_sample = X

    print(f"  Sample size: {len(X_sample)} rows, {len(X_sample.columns)} columns")
    
    print("  Creating SHAP explainer...")
    try:
        start_time = time.time()
        explainer = shap.TreeExplainer(final_model_for_shap)
        print("✓ SHAP explainer successfully created.")
        
        print("  Calculating SHAP values... (This process may take a few minutes)")
        shap_values = explainer.shap_values(X_sample)
        end_time = time.time()
        print(f"✓ SHAP values successfully calculated. Duration: {end_time - start_time:.2f} seconds.")
        
        print("  Creating and saving the plot...")
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, feature_names=X_sample.columns, show=False, plot_size=None)
        plt.title('SHAP Analysis: Feature Impact on Predicting Vehicle_KM', fontsize=18, weight='bold', pad=20)
        plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=14)
        plt.subplots_adjust(left=0.35, right=0.95, top=0.9, bottom=0.1) # Increase the left margin
        
        filename = 'shap_summary_FINAL.png'
        plt.savefig(filename, dpi=300)
        
        print("\n" + "#"*80)
        print(f"# ✓✓✓ SHAP PLOT SUCCESSFULLY SAVED: {filename} ✓✓✓")
        print("#"*80 + "\n")
        
        plt.show()
        
    except Exception as e:
        print("\n" + "!"*80)
        print("!!! A CRITICAL ERROR OCCURRED DURING SHAP ANALYSIS !!!")
        print("Please check the full error traceback below:")
        traceback.print_exc()
        print("!"*80 + "\n")

# --- MAIN PROGRAM ---
if __name__ == '__main__':
    df = load_and_prepare_data('data.xlsx')
    if df is not None:
        gnn_embeddings, topology_df = get_gnn_embeddings_and_topology(df)
        generate_interpretability_evidence(df, gnn_embeddings, topology_df)
