import xgboost as xgb
print(f"XGBOOST VERSION USED BY THE CODE: {xgb.__version__}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone
import catboost as cb

# Set seeds for reproducibility
np.random.seed(42)

# --- Step 1: Data Loading and Tabulation ---
def prepare_tabular_data(file_path='data.xlsx'):
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"ERROR: '{file_path}' not found.")
        return None, None

    # Apply log transformation to reduce the extreme right skew in the target variable
    if 'Araç KM' in df.columns:
        df['Araç KM'] = np.log1p(df['Araç KM'])
    # Separate features and target
    features_df = df.drop(columns=['Araç KM', 'Hattın Kodu', 'Yıl', 'Periyot'], errors='ignore')
    y = df['Araç KM'].values
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df)
    print(f"Data loaded. Table created with {X.shape[0]} samples and {X.shape[1]} features.")
    return X, y, features_df.columns.tolist()

# --- Step 2: Performance Evaluation Function ---
def evaluate_performance(y_true_log, y_pred_log):
    """Converts values from logarithmic scale back to original scale and calculates metrics."""
    y_true_original = np.expm1(y_true_log)
    y_pred_original = np.expm1(y_pred_log)

    rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
    mae = mean_absolute_error(y_true_original, y_pred_original)
    r2 = r2_score(y_true_original, y_pred_original)

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}, y_true_original, y_pred_original

# --- Step 3: Visualization Function ---
def plot_actual_vs_predicted(y_true, y_pred, title, filename):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor='k', s=80, color="#005A9C", ax=ax)
    perfect_line = np.linspace(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()), 100)
    ax.plot(perfect_line, perfect_line, color='#E41A1C', linestyle='--', linewidth=2.5, label='Perfect Prediction')
    ax.set_title(title, fontsize=24, weight='bold', pad=20)
    ax.set_xlabel('Actual Values (Vehicle KM)', fontsize=18, weight='bold', labelpad=15)
    ax.set_ylabel('Predicted Values (Vehicle KM)', fontsize=18, weight='bold', labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=16)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'RMSE: {rmse:,.0f}\n$R^2$:      {r2:.4f}',
            transform=ax.transAxes, fontsize=18, weight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='lightgray', alpha=0.7))
    ax.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"High-resolution plot saved as '{filename}'.")
    plt.show()

# --- Main Program ---
if __name__ == '__main__':
 
    X_data, y_data, feature_names = prepare_tabular_data('data.xlsx')

    if X_data is not None:
        # Define models
        models = {
            'XGBoost': xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                # FINAL SOLUTION: The early stopping parameter is passed DIRECTLY to the model itself.
                early_stopping_rounds=50
            ),
            'CatBoost': cb.CatBoostRegressor(
                iterations=1500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                loss_function='RMSE',
                random_seed=42,
                verbose=0
            )
        }

        # ======================================================================
        # METHOD 1: HOLD-OUT VALIDATION
        # ======================================================================
        print("\n" + "="*60 + "\n          METHOD 1: HOLD-OUT VALIDATION (80/20 SPLIT)\n" + "="*60)
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
        print(f"Data split into {len(X_train)} training and {len(X_test)} test samples.")

        for name, model in models.items():
            print(f"\n--- Training Model: {name} ---")

            if name == 'XGBoost':
                # FINAL SOLUTION: Only the data and validation set are passed to the .fit() method.
                # The model knows how to use early stopping from its own configuration.
                model.fit(X_train, y_train,
                          eval_set=[(X_test, y_test)],
                          verbose=False)
            else:
                # This method was already correct for CatBoost.
                model.fit(X_train, y_train)

            print("Training complete.")

            y_pred = model.predict(X_test)
            holdout_results, y_true_orig, y_pred_orig = evaluate_performance(y_test, y_pred)

            print(f"\n--- {name} HOLD-OUT TEST RESULTS ---")
            print(f"RMSE: {holdout_results['RMSE']:.2f}, MAE: {holdout_results['MAE']:.2f}, R-squared: {holdout_results['R2']:.4f}")
            plot_actual_vs_predicted(y_true_orig, y_pred_orig,
                                     f'Actual vs. Predicted ({name} - Hold-Out)',
                                     f'{name.lower()}_holdout_results_plot.png')

        # ======================================================================
        # METHOD 2: 5-FOLD CROSS-VALIDATION (CV)
        # ======================================================================
        print("\n" + "="*60 + "\n      METHOD 2: 5-FOLD CROSS-VALIDATION (CV)\n" + "="*60)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            # For the CV loop, it's more correct to use a model without early_stopping_rounds.
            # Therefore, we will clone the model definition and remove that parameter.
            if 'early_stopping_rounds' in model.get_params():
                 cv_params = model.get_params()
                 del cv_params['early_stopping_rounds']
                 base_model_for_cv = xgb.XGBRegressor(**cv_params)
            else:
                 base_model_for_cv = model
            
            print(f"\n--- Starting Cross-Validation: {name} ---")
            fold_metrics = []
            all_true_values, all_pred_values = [], []

            for fold, (train_idx, test_idx) in enumerate(kf.split(X_data)):
                print(f"  - Processing Fold {fold + 1}/5...")
                
                cloned_model = clone(base_model_for_cv)

                X_train_fold, X_test_fold = X_data[train_idx], X_data[test_idx]
                y_train_fold, y_test_fold = y_data[train_idx], y_data[test_idx]
                
                cloned_model.fit(X_train_fold, y_train_fold)

                y_pred_fold = cloned_model.predict(X_test_fold)
                results, y_true_fold_orig, y_pred_fold_orig = evaluate_performance(y_test_fold, y_pred_fold)

                fold_metrics.append(results)
                all_true_values.extend(y_true_fold_orig)
                all_pred_values.extend(y_pred_fold_orig)

            # --- Final CV Results and Combined Plot ---
            print(f"\n--- {name} CROSS-VALIDATION SUMMARY RESULTS ---")
            results_df = pd.DataFrame(fold_metrics)
            mean_results = results_df.mean()
            std_results = results_df.std()
            print(f"Average RMSE: {mean_results['RMSE']:.2f} (± {std_results['RMSE']:.2f})")
            print(f"Average MAE : {mean_results['MAE']:.2f} (± {std_results['MAE']:.2f})")
            print(f"Average R^2 : {mean_results['R2']:.4f} (± {std_results['R2']:.4f})")

            plot_actual_vs_predicted(np.array(all_true_values), np.array(all_pred_values),
                                     f'Actual vs. Predicted ({name} - 5-Fold CV Combined)',
                                     f'{name.lower()}_cross_validation_results_plot.png')
