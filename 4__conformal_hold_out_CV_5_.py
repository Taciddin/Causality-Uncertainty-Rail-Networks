import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mapie.regression import MapieRegressor

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

# 2. EVALUATING POINT PREDICTION PERFORMANCE
def evaluate_point_predictions(method_name, y_true, y_pred):
    """
    Calculates and prints RMSE, MAE, and R-squared metrics for the given predictions and true values.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n--- POINT PREDICTION MODEL PERFORMANCE ({method_name}) ---")
    print(f"  - RMSE (Root Mean Square Error): {rmse:.2f}")
    print(f"  - MAE (Mean Absolute Error):   {mae:.2f}")
    print(f"  - R-squared (Coefficient of Determination): {r2:.3f}")

# 3. EVALUATING CONFORMAL PREDICTION RESULTS
def evaluate_conformal_results(y_test, y_pred, y_pis, alpha):
    """
    Evaluates the conformal prediction results (coverage, width) and prints a summary table.
    """
    # Calculate coverage rate and interval width
    coverage = (y_test >= y_pis[:, 0, 0]) & (y_test <= y_pis[:, 1, 0])
    coverage_score = coverage.mean()
    width = (y_pis[:, 1, 0] - y_pis[:, 0, 0]).mean()
    
    print("\n--- CONFORMAL PREDICTION INTERVAL METRICS ---")
    print(f"  - Target Coverage Rate: {1 - alpha:.2f}")
    print(f"  - Actual Coverage Rate: {coverage_score:.2f}")
    print(f"  - Average Prediction Interval Width: {width:.2f}")

    # Combine the results in a DataFrame
    results_df = pd.DataFrame({
        'Actual Value': y_test,
        'Point Prediction': y_pred,
        'Interval Lower Bound': y_pis[:, 0, 0],
        'Interval Upper Bound': y_pis[:, 1, 0],
        'Is Covered?': coverage
    })
    
    print("\n--- TEST SET PREDICTION RESULTS (First 15 Rows) ---")
    print(results_df.head(15).to_string())
    
    print("\n--- INTERPRETATION ---")
    print("This table shows the point prediction made by the model for each test data point and")
    print(f"the prediction interval that is guaranteed to contain the true value with {int((1-alpha)*100)}% probability.")
    if abs(coverage_score - (1 - alpha)) < 0.02:
         print(f"The fact that the actual coverage rate ({coverage_score:.2f}) is very close to the target ({1-alpha:.2f}) proves that the guarantee given by the model is reliable.")
    else:
         print(f"WARNING: The actual coverage rate ({coverage_score:.2f}) is significantly different from the target ({1-alpha:.2f}). The model calibration may not have worked as expected.")


# 4. RUNNING THE CONFORMAL PREDICTION ANALYSIS
def run_conformal_prediction(df):
    """
    Runs the conformal prediction analysis on the data using both Hold-Out (split) 
    and Cross-Validation methods.
    """
    if df is None:
        return

    # --- Defining Variables and the Model ---
    y = df['Araç KM']
    X = df.drop(columns=['Araç KM'])
    
    print("\nConformal Prediction Model Setup:")
    print(f"  - Target Variable (y): {y.name}")
    print(f"  - Features (X): {list(X.columns)}")

    # --- Splitting the Data ---
    # The split is done once at the beginning so that both methods can be evaluated on the same test data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nData split into {len(X_train)} training and {len(X_test)} test rows.")
    print("Both methods will be compared on the same test set.")

    # Base regression model
    base_regressor = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
    alpha = 0.05 # for a 95% confidence interval

    #================================================================
    # METHOD 1: HOLD-OUT (SPLIT-CONFORMAL) APPROACH
    #================================================================
    print("\n\n" + "="*60)
    print(" METHOD 1: HOLD-OUT (SPLIT-CONFORMAL) APPROACH")
    print("="*60)
    print("Description: In this method, `mapie` splits the training data (80%) into two again.")
    print("It trains the model with one part and learns the error margins with the other part (calibration set).")

    # Set up MAPIE with the "split" strategy
    mapie_regressor_split = MapieRegressor(base_regressor, cv="split", random_state=42)
    
    print("\nTraining and calibrating the model using the TRAINING DATA...")
    mapie_regressor_split.fit(X_train, y_train)
    print("✓ Model training and calibration completed.")
    
    # Make predictions on the test data
    y_pred_split, y_pis_split = mapie_regressor_split.predict(X_test, alpha=alpha)

    # Evaluate the Results
    evaluate_point_predictions("Hold-Out", y_test, y_pred_split)
    evaluate_conformal_results(y_test, y_pred_split, y_pis_split, alpha)

    #================================================================
    # METHOD 2: 5-FOLD CROSS-VALIDATION (CV+) APPROACH
    #================================================================
    print("\n\n" + "="*60)
    print(" METHOD 2: 5-FOLD CROSS-VALIDATION (CV+) APPROACH")
    print("="*60)
    print("Description: This method uses the entire training data more efficiently.")
    print("It splits the data into 5 folds, training on 4 parts and learning the error margin on 1 part each time.")
    print("This generally produces narrower and more stable prediction intervals.")

    # Set up MAPIE with the 5-fold CV strategy
    mapie_regressor_cv = MapieRegressor(base_regressor, cv=5)

    print("\nTraining and calibrating the model using the TRAINING DATA (with 5-fold CV)...")
    mapie_regressor_cv.fit(X_train, y_train)
    print("✓ Model training and calibration completed.")

    # Make predictions on the test data
    y_pred_cv, y_pis_cv = mapie_regressor_cv.predict(X_test, alpha=alpha)
    
    # Evaluate the Results
    evaluate_point_predictions("5-Fold CV", y_test, y_pred_cv)
    evaluate_conformal_results(y_test, y_pred_cv, y_pis_cv, alpha)


# --- MAIN PROGRAM ---
if __name__ == '__main__':
    # Make sure the 'data.xlsx' file is in the same directory where the code is run
    data = load_and_prepare_data('data.xlsx')
    if data is not None:
        run_conformal_prediction(data)
