import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow
import os

# MLflow Setup
mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("SuperKart-Sales-Prediction")

# Hugging Face API authentication
api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "cjigar/superkart" # Use your specific HF ID here

# Load datasets directly from Hugging Face
Xtrain = pd.read_csv(f"hf://datasets/{repo_id}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{repo_id}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{repo_id}/ytrain.csv").values.ravel()
ytest = pd.read_csv(f"hf://datasets/{repo_id}/ytest.csv").values.ravel()

# Define features based on SuperKart Data Description
numeric_features = [
    'Product_Weight',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Establishment_Year'
]

categorical_features = [
    'Product_Id',
    'Product_Sugar_Content',
    'Product_Type',
    'Store_Id',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type'
]

# Preprocessing Pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define XGBoost Regressor (Changed from Classifier for Sales Total)
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbregressor__n_estimators': [100, 150],
    'xgbregressor__max_depth': [3, 5],
    'xgbregressor__learning_rate': [0.01, 0.1]
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_reg)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Store the best model
    best_model = grid_search.best_estimator_

    # Predictions
    ypred = best_model.predict(Xtest)

    # Metrics
    mae = mean_absolute_error(ytest, ypred)
    r2 = r2_score(ytest, ypred)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)

    # Save and Upload Model
    model_filename = "superkart_sales_model.joblib"
    joblib.dump(best_model, model_filename)


    # Log the model artifact
    mlflow.log_artifact(model_filename)
    print(f"Model saved as artifact at: {model_filename}")

    # Upload to Hugging Face
    repo_id = "cjigar/superkart-package-model"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_superkart_package_model_v1.joblib",
        path_in_repo="best_superkart_package_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
