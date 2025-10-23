import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from econml.dml import LinearDML

# 1. DATA LOADING AND PREPARATION
def load_and_prepare_data(file_path='data.xlsx'):
    """Loads, preprocesses, and returns the data from the specified path."""
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found. Please check the file path.")
        return None
    
    # Remove columns that will not be used or could cause leakage
    df = df.drop(columns=['Hattın Kodu', 'Yıl', 'Periyot'], errors='ignore')
    df = df.dropna()
    
    print("✓ Data successfully loaded and preprocessed.")
    return df

# 2. EVALUATING THE PERFORMANCE OF NUISANCE MODELS
def evaluate_nuisance_model(model_name, y_true, y_pred):
    """
    Calculates and prints RMSE, MAE, and R-squared metrics for the given predictions and true values.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"  - Model Performance ({model_name}):")
    print(f"    - RMSE (Root Mean Square Error): {rmse:.2f}")
    print(f"    - MAE (Mean Absolute Error):   {mae:.2f}")
    print(f"    - R-squared (Coefficient of Determination): {r2:.3f}")

# 3. INTERPRETING THE CAUSAL EFFECT
def interpret_causal_summary(summary):
    """
    Analyzes the EconML model summary and outputs a user-friendly interpretation.
    """
    try:
        # Dynamically get the required values from the summary table
        headers = summary.tables[0].data[0]
        data_row = summary.tables[0].data[1]
        
        point_estimate_idx = headers.index('point_estimate')
        pvalue_idx = headers.index('pvalue')
        ci_lower_idx = headers.index('ci_lower')
        ci_upper_idx = headers.index('ci_upper')

        point_estimate = float(data_row[point_estimate_idx])
        p_value = float(data_row[pvalue_idx])
        conf_int = [float(data_row[ci_lower_idx]), float(data_row[ci_upper_idx])]

        print("\n--- INTERPRETATION ---")
        print(f"According to these results, holding all other factors constant, EACH NEW VEHICLE added to the fleet,")
        print(f"increases the total 'Vehicle KM' by approximately {point_estimate:.2f} units on average.")
        
        if p_value < 0.05:
            print(f"This result is statistically SIGNIFICANT because the p-value ({p_value:.3f}) is less than 0.05.")
        else:
            print(f"This result is NOT statistically significant because the p-value ({p_value:.3f}) is greater than 0.05.")
        
        print(f"The true value of the effect is expected to be within the range [{conf_int[0]:.2f}, {conf_int[1]:.2f}] with 95% probability.")
    
    except (IndexError, ValueError) as e:
        print(f"An error occurred while interpreting the summary: {e}")
        print("Please check the format of the model summary.")


# 4. RUNNING THE CAUSAL ANALYSIS
def run_causal_analysis(df):
    """
    Runs the DML analysis on the data using both Hold-Out and Cross-Validation methods.
    """
    if df is None:
        return

    # --- Defining Variables ---
    Y = df[['Araç KM']]
    T = df[['Araç Sayısı']]
    W = df.drop(columns=['Araç KM', 'Araç Sayısı'])
    
    print("\nCausal Model Setup:")
    print(f"  - Outcome Variable (Y): {Y.columns[0]}")
    print(f"  - Treatment Variable (T): {T.columns[0]}")
    print(f"  - Control Variables (W/X): {list(W.columns)}")

    # Base model definitions
    model_y = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42)
    model_t = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42)
    
    #================================================================
    # METHOD 1: HOLD-OUT (Train/Test Split)
    #================================================================
    print("\n\n" + "="*50)
    print(" METHOD 1: HOLD-OUT (TRAIN/TEST SPLIT) APPROACH")
    print("="*50)
    
    # --- Splitting the Data ---
    Y_train, Y_test, T_train, T_test, W_train, W_test = train_test_split(
        Y, T, W, test_size=0.2, random_state=42
    )
    print(f"\nData split into {len(Y_train)} training and {len(Y_test)} test rows.")

    # --- Evaluating Nuisance Models (on Test Set) ---
    print("\nEvaluating Nuisance Models Performance (on Test Data)...")
    # Model Y (Vehicle KM prediction)
    model_y.fit(W_train, Y_train.values.ravel())
    Y_pred = model_y.predict(W_test)
    evaluate_nuisance_model('Vehicle KM Predictor (model_y)', Y_test, Y_pred)

    # Model T (Vehicle Count prediction)
    model_t.fit(W_train, T_train.values.ravel())
    T_pred = model_t.predict(W_test)
    evaluate_nuisance_model('Vehicle Count Predictor (model_t)', T_test, T_pred)

    # --- Setting Up and Training the DML Model ---
    dml_estimator_holdout = LinearDML(
        model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42),
        model_t=RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42),
        random_state=42
    )
    
    print("\nTraining causal model with TRAINING DATA...")
    dml_estimator_holdout.fit(Y_train, T_train, X=W_train, W=None) # W=None because X=W already
    print("✓ Model training completed.")
    
    summary_holdout = dml_estimator_holdout.summary(feature_names=W.columns.tolist())
    print("\n--- STATISTICAL SUMMARY OF THE CAUSAL MODEL (HOLD-OUT) ---")
    print(summary_holdout)
    interpret_causal_summary(summary_holdout)

    #================================================================
    # METHOD 2: CROSS-VALIDATION
    #================================================================
    print("\n\n" + "="*50)
    print(" METHOD 2: 5-FOLD CROSS-VALIDATION (CV) APPROACH")
    print("="*50)
    print("\nThis method uses the entire dataset to provide a more robust effect estimate.")

    # --- Evaluating Nuisance Models (with CV) ---
    print("\nEvaluating Nuisance Models Performance (with 5-Fold CV)...")
    
    # CV scores for Model Y (Vehicle KM prediction)
    r2_scores_y = cross_val_score(model_y, W, Y.values.ravel(), cv=5, scoring='r2')
    rmse_scores_y = np.sqrt(-cross_val_score(model_y, W, Y.values.ravel(), cv=5, scoring='neg_mean_squared_error'))
    mae_scores_y = -cross_val_score(model_y, W, Y.values.ravel(), cv=5, scoring='neg_mean_absolute_error') # NEWLY ADDED
    print(f"  - Model Performance (Vehicle KM Predictor - model_y):")
    print(f"    - Mean R-squared: {r2_scores_y.mean():.3f} (±{r2_scores_y.std():.3f})")
    print(f"    - Mean RMSE:      {rmse_scores_y.mean():.2f} (±{rmse_scores_y.std():.2f})")
    print(f"    - Mean MAE:       {mae_scores_y.mean():.2f} (±{mae_scores_y.std():.2f})") # NEWLY ADDED

    # CV scores for Model T (Vehicle Count prediction)
    r2_scores_t = cross_val_score(model_t, W, T.values.ravel(), cv=5, scoring='r2')
    rmse_scores_t = np.sqrt(-cross_val_score(model_t, W, T.values.ravel(), cv=5, scoring='neg_mean_squared_error'))
    mae_scores_t = -cross_val_score(model_t, W, T.values.ravel(), cv=5, scoring='neg_mean_absolute_error') # NEWLY ADDED
    print(f"  - Model Performance (Vehicle Count Predictor - model_t):")
    print(f"    - Mean R-squared: {r2_scores_t.mean():.3f} (±{r2_scores_t.std():.3f})")
    print(f"    - Mean RMSE:      {rmse_scores_t.mean():.2f} (±{rmse_scores_t.std():.2f})")
    print(f"    - Mean MAE:       {mae_scores_t.mean():.2f} (±{mae_scores_t.std():.2f})") # NEWLY ADDED

    # --- Setting Up and Training the DML Model (with CV) ---
    dml_estimator_cv = LinearDML(
        model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42),
        model_t=RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42),
        cv=5,  # 5-fold cross-validation is activated here
        random_state=42
    )

    print("\nTraining causal model with the ENTIRE DATA (using 5-fold CV)...")
    # When using CV, the `fit` method takes the entire dataset and handles the splitting internally.
    dml_estimator_cv.fit(Y, T, X=W, W=None)
    print("✓ Model training completed.")
    
    summary_cv = dml_estimator_cv.summary(feature_names=W.columns.tolist())
    print("\n--- STATISTICAL SUMMARY OF THE CAUSAL MODEL (5-FOLD CV) ---")
    print(summary_cv)
    interpret_causal_summary(summary_cv)

# --- MAIN PROGRAM ---
if __name__ == '__main__':
    # Make sure the 'data.xlsx' file is in the same directory where the code is run
    df = load_and_prepare_data('data.xlsx')
    run_causal_analysis(df)
