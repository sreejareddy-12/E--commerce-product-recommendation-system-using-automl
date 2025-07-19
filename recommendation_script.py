import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from sklearn.model_selection import train_test_split # Although not used for H2O training, good to keep if you plan a manual split
from tkinter import Tk, filedialog, messagebox # Added messagebox for user feedback

# --- Step 1: Initialize H2O ---
# Initializes the H2O cluster. 'ip' and 'port' can be specified if connecting
# to a remote cluster. For local execution, no arguments are usually needed.
try:
    h2o.init()
    print("H2O cluster initialized successfully.")
except Exception as e:
    print(f"Error initializing H2O cluster: {e}")
    exit() # Exit if H2O can't be initialized

# --- Step 2: Upload Dataset ---
def upload_dataset():
    """
    Prompts the user to select a CSV dataset using a Tkinter file dialog.
    Returns a pandas DataFrame. Handles cases where no file is selected.
    """
    root = Tk()
    root.withdraw() # Hide the main Tkinter window

    file_path = filedialog.askopenfilename(
        title="Select Dataset",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )

    root.destroy() # Close the Tkinter root window after selection attempt

    if not file_path:
        messagebox.showwarning("No File Selected", "No dataset file was selected. Please run the script again and select a file.")
        return None # Return None if no file is selected

    try:
        df = pd.read_csv(file_path)
        print(f"Dataset '{file_path}' loaded successfully.")
        return df
    except Exception as e:
        messagebox.showerror("Error Loading File", f"An error occurred while loading the dataset: {e}")
        return None

df = upload_dataset()

if df is None: # Exit if no dataset was loaded
    h2o.shutdown(prompt=False)
    exit()

# Check if dataset has required columns
required_columns = ['productid', 'userid', 'score']
if not all(column in df.columns for column in required_columns):
    messagebox.showerror("Column Error", f"Dataset must contain the following columns: {required_columns}")
    h2o.shutdown(prompt=False)
    exit()

# Rename columns for consistency
column_mapping = {
    'productid': 'product_id',
    'userid': 'user_id',
    'score': 'rating'
}
df.rename(columns=column_mapping, inplace=True)
print("Columns renamed for consistency.")

# --- Step 3: Handle Small Datasets (e.g., 10 rows) ---
if len(df) <= 10:
    print("Warning: Dataset is very small. Results may not be meaningful.")
    messagebox.showinfo("Small Dataset Warning", "The dataset is very small. Model performance and recommendations might not be reliable.")

# --- Step 4: Clean and Validate Data ---
# Remove rows with missing values. Consider more sophisticated imputation
# strategies for larger, real-world datasets.
initial_rows = len(df)
df.dropna(inplace=True)
cleaned_rows = len(df)
print(f"Removed {initial_rows - cleaned_rows} rows with missing values.")
print("Data after cleaning (first 5 rows):")
print(df.head())

if df.empty:
    messagebox.showerror("Empty Dataset", "Dataset is empty after cleaning. Cannot proceed.")
    h2o.shutdown(prompt=False)
    exit()

# Convert to H2O Frame
# This step transfers the pandas DataFrame to the H2O cluster's memory.
h2o_df = h2o.H2OFrame(df)
print("Dataset converted to H2OFrame.")

# --- Step 5: Specify Features and Target ---
x = ['user_id', 'product_id'] # Features for the model
y = 'rating'                  # Target variable (what we want to predict)

# Ensure target column is numerical for regression
if h2o_df[y].isstring():
    h2o_df[y] = h2o_df[y].asnumeric()
    print(f"Converted target column '{y}' to numeric.")

# Convert 'user_id' and 'product_id' to categorical/enum type if they are not already,
# as this is often beneficial for recommendation systems in H2O.
for col in x:
    if not h2o_df[col].isfactor():
        h2o_df[col] = h2o_df[col].asfactor()
        print(f"Converted feature column '{col}' to factor (categorical).")


# --- Step 6: Train AutoML Model with Increased Runtime ---
# H2O AutoML automates the machine learning workflow, including algorithm selection
# and hyperparameter tuning. max_runtime_secs sets the maximum training time.
print(f"Starting AutoML training for {y} based on {x}...")
aml = H2OAutoML(max_runtime_secs=300, seed=42, nfolds=5, sort_metric='rmse') # Increased folds for robustness
aml.train(x=x, y=y, training_frame=h2o_df)
print("AutoML training complete.")

# --- Step 7: Evaluate the Best Model ---
# The 'leader' model is the best performing model found by AutoML based on the
# chosen 'sort_metric' (default is AUC for classification, RMSE for regression).
leader = aml.leader
print("\n--- Best Model from AutoML ---")
print(leader.model_id) # Print the ID of the best model

# Evaluate Model Performance on the training data.
# For a true test of generalization, you would split your initial pandas DataFrame
# into train and test sets *before* converting to H2OFrame, and then pass
# 'test_data=h2o_test_df' here.
perf = leader.model_performance(test_data=h2o_df)
print("\n--- Model Performance on Training Data ---")
print(perf)

print("\n--- Leaderboard of Models ---")
# Display the leaderboard, showing how different models performed.
print(aml.leaderboard)

# --- Step 8: Generate Recommendations for All Users ---
print("\n--- Generating Recommendations for all users ---")
all_users = df['user_id'].unique()    # Get all unique users from the original DataFrame
all_products = df['product_id'].unique() # Get all unique products

# Create a DataFrame to store recommendations
recommendations = pd.DataFrame(columns=['user_id', 'product_id', 'predicted_rating'])

for user_id in all_users:
    # Create User-Product Pairs for all products for the current user
    # This generates potential ratings for all products the user hasn't rated yet (or has)
    user_data = pd.DataFrame({
        'user_id': [user_id] * len(all_products),
        'product_id': all_products
    })

    # Convert user_data to H2O Frame for prediction
    # Ensure column types match the training frame's column types
    user_h2o = h2o.H2OFrame(user_data)
    for col in x:
        if not user_h2o[col].isfactor():
            user_h2o[col] = user_h2o[col].asfactor()

    # Predict ratings for the user-product pairs using the best model
    # The .as_data_frame() method converts the H2OFrame prediction back to a pandas DataFrame.
    # 'use_multi_thread=True' is not a valid argument for as_data_frame; removed it.
    user_predictions = leader.predict(user_h2o).as_data_frame()

    # Add predictions to the User Data DataFrame
    user_data['predicted_rating'] = user_predictions['predict']

    # Filter out products already rated by the user (optional, but common for recommendations)
    # This requires knowing which products the user has already rated in the original df
    # For simplicity, we'll keep all for now, but you might want to add this logic:
    # rated_products = df[df['user_id'] == user_id]['product_id'].tolist()
    # user_data = user_data[~user_data['product_id'].isin(rated_products)]

    # Get Top-N Recommendations for the User (e.g., Top 5 highest predicted ratings)
    top_recommendations = user_data.sort_values(by='predicted_rating', ascending=False).head(5)
    recommendations = pd.concat([recommendations, top_recommendations], ignore_index=True)

# Save All Recommendations to a CSV file
output_filename = "all_user_recommendations.csv"
try:
    recommendations.to_csv(output_filename, index=False)
    print(f"\nRecommendations saved to {output_filename}")
except Exception as e:
    messagebox.showerror("Save Error", f"Could not save recommendations to CSV: {e}")

# Debug Outputs: Sample Recommendations
print("\n--- Sample Recommendations (First 10 rows) ---")
print(recommendations.head(10))

# --- Step 9: Shutdown H2O ---
# Always shut down the H2O cluster when you are done to free up resources.
h2o.shutdown(prompt=False) # prompt=False prevents the "Shutting down H2O?" confirmation
print("\nH2O cluster shut down.")