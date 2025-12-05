# ==========================================
# IMPOR LIBRARY
# ==========================================

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import GridSearchCV
import warnings
import joblib

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv('Exam_Score_Prediction.csv')

# ==========================================
# PREPROCESSING
# ==========================================

print("Starting data preprocessing setup...")

# Drop ID
df_processed = df.drop('student_id', axis=1)

# Define X (Fitur MENTAH) dan y (Target)
y = df_processed['exam_score']
X = df_processed.drop('exam_score', axis=1)
# CATATAN: X di sini masih berisi teks (Male, b.com, dll). Kita biarkan begitu.

# Define columns
ordinal_cols = ['sleep_quality', 'facility_rating', 'exam_difficulty']
onehot_cols = ['gender', 'course', 'internet_access', 'study_method']
numeric_cols_to_scale = ['study_hours', 'class_attendance', 'age']

# Define categories
sleep_quality_categories = ['poor', 'average', 'good']
facility_rating_categories = ['low', 'medium', 'high']
exam_difficulty_categories = ['easy', 'moderate', 'hard']

categories_list = [
    sleep_quality_categories,
    facility_rating_categories,
    exam_difficulty_categories
]

# Create Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OrdinalEncoder(categories=categories_list), ordinal_cols),
        ('onehot', OneHotEncoder(handle_unknown='ignore'), onehot_cols),
        ('scaler', StandardScaler(), numeric_cols_to_scale)
    ],
    remainder='passthrough'
)

print("Preprocessing setup complete.")

# ==========================================
# DATA SPLITTING
# ==========================================

print("Splitting data...")
# Kita split data yang masih MENTAH. Biarkan Pipeline yang mengurus transformasinya nanti.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape} (Masih mengandung data teks)")
print(f"y_train shape: {y_train.shape}")

# ==========================================
# MODELING DENGAN PIPELINE (DIPERBAIKI)
# ==========================================

# Pipeline = Preprocessor + Model

# 1. Linear Regression Pipeline
pipeline_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 2. Random Forest Pipeline
pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
])

# 3. XGBoost Pipeline
pipeline_xgb = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist'))
])

models = {
    'Linear Regression': pipeline_lr,
    'Random Forest': pipeline_rf,
    'XGBoost': pipeline_xgb
}

print("\nTraining & Evaluating Pipelines...")
results = {}

for name, pipeline in models.items():
    print(f"\n--- Training {name} ---")

    # Fit Pipeline
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2 Score': r2}

    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'R2 Score: {r2:.4f}')

print("\n\n--- Perbandingan Hasil Akhir ---")
df_results = pd.DataFrame(results).T
print(df_results)

# ==========================================
# HYPERPARAMETER TUNING
# ==========================================

warnings.filterwarnings('ignore')

print("\nStarting Hyperparameter Tuning...")

# Karena model ada di dalam pipeline dengan nama 'regressor',
# nama parameter harus diberi prefix 'regressor__'

# --- Random Forest Tuning ---
param_grid_rf = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [10, 20],
    'regressor__min_samples_split': [2, 5]
}

grid_search_rf = GridSearchCV(
    pipeline_rf,
    param_grid_rf,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
grid_search_rf.fit(X_train, y_train)
print(f"Best RF R2: {grid_search_rf.best_score_:.4f}")


# --- XGBoost Tuning ---
param_grid_xgb = {
    'regressor__n_estimators': [100, 200],
    'regressor__learning_rate': [0.05, 0.1],
    'regressor__max_depth': [5, 10]
}

grid_search_xgb = GridSearchCV(
    pipeline_xgb,
    param_grid_xgb,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
grid_search_xgb.fit(X_train, y_train)
print(f"Best XGB R2: {grid_search_xgb.best_score_:.4f}")

# ==========================================
# SAVING FINAL MODEL
# ==========================================

print("\n--- Saving Models ---")

# Simpan Model Linear Regression (Pipeline)
# Ini yang akan dipakai di app.py
joblib_file = 'Linear_Regression_(Default)_model.joblib'
joblib.dump(pipeline_lr, joblib_file)
print(f"Model Linear Regression berhasil disimpan sebagai: {joblib_file}")

print("\nSELESAI. Silakan download file .joblib dan upload ke folder Streamlit.")

import sklearn
print(sklearn.__version__)
# Contoh output: 1.5.2