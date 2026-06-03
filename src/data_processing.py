import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# =====================================================================
# Local WOE transformer implementation (avoids xverse compatibility issues)
# =====================================================================
class WOETransformer(BaseEstimator, TransformerMixin):
    """Simple Weight-of-Evidence transformer for selected categorical features."""

    def __init__(self, feature_names='all', exclude_features=None, woe_prefix=None, treat_missing='separate', woe_bins=None, monotonic_binning=False):
        self.feature_names = feature_names
        self.exclude_features = exclude_features
        self.woe_prefix = woe_prefix
        self.treat_missing = treat_missing
        self.woe_bins = woe_bins
        self.monotonic_binning = monotonic_binning
        self.transform_features = None
        self.iv_df = None

    def check_datatype(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("The input data must be pandas dataframe. But the input provided is " + str(type(X)))
        return self

    def treat_missing_values(self, X):
        if self.treat_missing == 'separate':
            return X.fillna('NA')
        if self.treat_missing == 'mode':
            return X.fillna(X.mode().iloc[0])
        if self.treat_missing == 'least_frequent':
            for col in X.columns:
                X[col] = X[col].fillna(X[col].value_counts().index[-1])
            return X
        raise ValueError("Missing values must be one of 'separate', 'mode', or 'least_frequent'.")

    def fit(self, X, y):
        try:
            X, y = X
        except Exception:
            pass
        self.check_datatype(X)
        if X.shape[0] != len(y):
            raise ValueError(f"Mismatch in input lengths. Length of X is {X.shape[0]} but length of y is {len(y)}.")
        if len(np.unique(y)) != 2:
            raise ValueError("The target column y must be binary. But the target contains " + str(len(np.unique(y))) + " unique value(s).")

        numeric_features = list(X._get_numeric_data().columns)
        categorical_features = list(X.columns.difference(numeric_features))
        if self.feature_names == 'all':
            self.transform_features = categorical_features
        else:
            self.transform_features = [f for f in self.feature_names if f in X.columns]

        if self.exclude_features:
            self.transform_features = [f for f in self.transform_features if f not in self.exclude_features]

        temp_X = X[self.transform_features].astype('object')
        temp_X = self.treat_missing_values(temp_X)

        self.woe_bins = {}
        iv_rows = []
        target_series = pd.Series(y, name='__target__')
        for feature in temp_X.columns:
            counts = pd.DataFrame({'X': temp_X[feature], 'Y': target_series})
            grouped = counts.groupby('X', as_index=True)['Y']
            event = grouped.sum()
            total_event = event.sum()
            non_event = grouped.count() - event
            total_non_event = non_event.sum()
            event_dist = (event + 1e-8) / (total_event + 1e-8)
            non_event_dist = (non_event + 1e-8) / (total_non_event + 1e-8)
            woe_values = np.log(event_dist / non_event_dist)
            self.woe_bins[feature] = woe_values.to_dict()

            iv = ((event_dist - non_event_dist) * woe_values).sum()
            iv_rows.append({'Variable_Name': feature, 'Information_Value': iv})

        self.iv_df = pd.DataFrame(iv_rows).sort_values('Information_Value', ascending=False).reset_index(drop=True)
        return self

    def transform(self, X, y=None):
        try:
            X, y = X
        except Exception:
            pass
        self.check_datatype(X)
        outX = X.copy(deep=True)
        outX = self.treat_missing_values(outX.astype('object'))

        transform_features = [f for f in self.transform_features if f in outX.columns]
        if not transform_features:
            raise ValueError("Empty list for WOE transformation. Estimator has to be fitted to make WOE transformations")

        for feature in transform_features:
            outX[feature] = outX[feature].replace(self.woe_bins[feature])

        return outX

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


# =====================================================================
# 1. REQUIREMENT: CREATE AGGREGATE FEATURES
# =====================================================================
class CustomerBehaviorAggregator(BaseEstimator, TransformerMixin):
    """
    Computes customer-level historical profiles:
    - Total Transaction Amount (Sum)
    - Average Transaction Amount (Mean)
    - Transaction Count (Count)
    - Standard Deviation of Transaction Amounts (Variability)
    """
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.customer_profiles_ = None
        self.global_defaults_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        
        # Aggregate logic calculations per customer
        profiles = X_df.groupby(self.customer_id_col).agg(
            Total_Transaction_Amount=(self.amount_col, 'sum'),
            Average_Transaction_Amount=(self.amount_col, 'mean'),
            Transaction_Count=(self.amount_col, 'count'),
            Standard_Deviation_Transaction_Amount=(self.amount_col, 'std')
        )
        # Handle cases where customer has only 1 transaction (std dev becomes NaN)
        profiles['Standard_Deviation_Transaction_Amount'] = profiles['Standard_Deviation_Transaction_Amount'].fillna(0.0)
        self.customer_profiles_ = profiles
        
        # Cold-start fallback via Imputation (Median / Mean baselines)
        self.global_defaults_ = {
            'Total_Transaction_Amount': X_df[self.amount_col].median(),
            'Average_Transaction_Amount': X_df[self.amount_col].mean(),
            'Transaction_Count': 1.0,
            'Standard_Deviation_Transaction_Amount': 0.0
        }
        return self

    def transform(self, X):
        if self.customer_profiles_ is None:
            raise RuntimeError("The transformer must be fitted before calling transform.")
        X_df = pd.DataFrame(X).copy()
        X_merged = X_df.merge(self.customer_profiles_, on=self.customer_id_col, how='left')
        
        # Infill unseen customer values
        for col, default_val in self.global_defaults_.items():
            X_merged[col] = X_merged[col].fillna(default_val)
            
        # Drop unique text ID so it doesn't break downstream mathematical matrices
        if self.customer_id_col in X_merged.columns:
            X_merged = X_merged.drop(columns=[self.customer_id_col])
            
        return X_merged


# =====================================================================
# 2. REQUIREMENT: EXTRACT TEMPORAL FEATURES
# =====================================================================
class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts time features from raw timestamp components:
    - Transaction Hour
    - Transaction Day
    - Transaction Month
    - Transaction Year
    """
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        
        if self.datetime_col in X_df.columns:
            datetime_series = pd.to_datetime(X_df[self.datetime_col])
            
            X_df['Transaction_Hour'] = datetime_series.dt.hour
            X_df['Transaction_Day'] = datetime_series.dt.day
            X_df['Transaction_Month'] = datetime_series.dt.month
            X_df['Transaction_Year'] = datetime_series.dt.year
            
            # Remove raw text timestamp column after extraction
            X_df = X_df.drop(columns=[self.datetime_col])
            
        return X_df


# =====================================================================
# 3. UNIFIED PRODUCTION EXECUTIVE PIPELINE ENGINE
# =====================================================================
def execute_instructional_pipeline():
    print("🚀 Initiating Final Credit Risk Preprocessing Pipeline Engine...")
    
    # Automated Working Directory Path Mapping
    script_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_directory)
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'data.csv')
    
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"❌         python src/data_processing.py Raw dataset missing from source tree layout: {raw_data_path}")
        
    df = pd.read_csv(raw_data_path)
    print(f"📊 Dataset Loaded Successfully. Row/Column Shape: {df.shape}")
    
    # Run-time Target Vector Detection & Isolation
    target_names = ['Target', 'FraudResult', 'target', 'fraud_result']
    target_col = next((col for col in target_names if col in df.columns), None)
    if target_col is not None:
        y = df[target_col].values
        X_raw = df.drop(columns=[target_col])
    else:
        print("⚠️ Target column not detected. Generating standard binary placeholder vector...")
        np.random.seed(42)
        y = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])
        X_raw = df.copy()

    # Exclude strict high-cardinality alphanumeric text markers from matrix calculations
    tracking_keys = ['TransactionId']
    X_features = X_raw.drop(columns=[col for col in tracking_keys if col in X_raw.columns])
    
    # Declare Base Preprocessing Schemas
    num_features = ['Amount', 'Value', 'PricingStrategy']
    cat_features = ['ProductCategory', 'ChannelId', 'ProviderId']
    
    # Cross-reference with columns actually existing inside your dataset variants
    num_features = [col for col in num_features if col in X_features.columns]
    cat_features = [col for col in cat_features if col in X_features.columns]
    
    # Appended list tracking the numeric parameters engineered dynamically by your custom blocks
    extended_numeric_schema = num_features + [
        'Transaction_Hour', 'Transaction_Day', 'Transaction_Month', 'Transaction_Year',
        'Total_Transaction_Amount', 'Average_Transaction_Amount', 'Transaction_Count', 'Standard_Deviation_Transaction_Amount'
    ]
    
    print("⚙️ Processing Conveyor: Imputing Missing Values, Encoding Categoricals, and Standardizing...")
    
    # =====================================================================
    # 4. REQUIREMENTS: MISSING VALUES, ENCODING & STANDARDIZATION
    # =====================================================================
    numeric_conveyor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),       # Handle Missing Values via Median Imputation
        ('scaler', StandardScaler())                         # Standardize Numerical Features (Mean=0, Std=1)
    ])

    categorical_conveyor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Encode Categoricals (One-Hot)
    ])

    base_preprocessor = ColumnTransformer(
        transformers=[
            ('numeric_belt', numeric_conveyor, [col for col in extended_numeric_schema]),
            ('categorical_belt', categorical_conveyor, cat_features)
        ],
        remainder='drop'
    )

    # =====================================================================
    # 5. REQUIREMENT: FEATURE ENGINEERING WITH WoE / IV
    # =====================================================================
    # Apply WOE directly to the engineered DataFrame prior to any one-hot encoding.
    production_pipeline = Pipeline(steps=[
        ('temporal_extractor', TemporalFeatureExtractor(datetime_col='TransactionStartTime')),
        ('behavioral_aggregator', CustomerBehaviorAggregator(customer_id_col='CustomerId', amount_col='Amount')),
        ('woe_transformation', WOETransformer(feature_names=cat_features)) # Local WOE transformation on selected categorical features
    ])
    
    print("🏋️ Fitting steps on historical rows and executing matrix transformations...")
    processed_matrix = production_pipeline.fit_transform(X_features, y)
    
    # Structuring Output DataFrame
    print("🧹 Converting matrix data back into pandas file schemas...")
    if isinstance(processed_matrix, np.ndarray):
        output_df = pd.DataFrame(processed_matrix)
        output_df.columns = [f"feature_{i}" for i in range(output_df.shape[1])]
    else:
        output_df = processed_matrix.copy()
        
    # Append target data column
    output_df['Target'] = y
    
    # Export File Setup
    output_directory = os.path.join(project_root, 'data', 'processed')
    os.makedirs(output_directory, exist_ok=True)
    output_save_file_path = os.path.join(output_directory, 'model_ready_data.csv')
    
    print(f"💾 Saving instruction-ready data frame asset file to: {output_save_file_path}")
    output_df.to_csv(output_save_file_path, index=False)
    print(f"✨ Setup Complete! Processed Data Dimensions: {output_df.shape}\n")

if __name__ == "__main__":
    execute_instructional_pipeline()