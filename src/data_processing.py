import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans

# =====================================================================
# LOCAL WOE TRANSFORMER IMPLEMENTATION
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
# REVISED UPDATED TRANSFORMER: CALCULATES RFM & VOLATILITY METRICS
# =====================================================================
class CustomerBehaviorAggregator(BaseEstimator, TransformerMixin):
    """
    Computes customer-level historical RFM profiles:
    - Recency: Days since last transaction relative to global snapshot date
    - Frequency: Total transaction count per customer
    - Monetary: Sum of all transaction amounts per customer
    - Standard_Deviation_Transaction_Amount: Variability of transaction amounts
    """
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount', datetime_col='TransactionStartTime'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.datetime_col = datetime_col
        self.customer_profiles_ = None
        self.global_defaults_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        X_df[self.datetime_col] = pd.to_datetime(X_df[self.datetime_col])
        
        # Consistent Data-Driven Snapshot Date: 1 day after the global maximum transaction date
        snapshot_date = X_df[self.datetime_col].max() + pd.Timedelta(days=1)
        
        # Calculate full suite of RFM + Volatility Metrics
        profiles = X_df.groupby(self.customer_id_col).agg(
            Recency=(self.datetime_col, lambda x: (snapshot_date - x.max()).days),
            Frequency=(self.amount_col, 'count'),
            Monetary=(self.amount_col, 'sum'),
            Standard_Deviation_Transaction_Amount=(self.amount_col, 'std')
        )
        profiles['Standard_Deviation_Transaction_Amount'] = profiles['Standard_Deviation_Transaction_Amount'].fillna(0.0)
        self.customer_profiles_ = profiles
        
        # Safe Imputation fallbacks for cold starts
        self.global_defaults_ = {
            'Recency': profiles['Recency'].median(),
            'Frequency': 1.0,
            'Monetary': X_df[self.amount_col].median(),
            'Standard_Deviation_Transaction_Amount': 0.0
        }
        return self

    def transform(self, X):
        if self.customer_profiles_ is None:
            raise RuntimeError("The transformer must be fitted before calling transform.")
        X_df = pd.DataFrame(X).copy()
        X_merged = X_df.merge(self.customer_profiles_, on=self.customer_id_col, how='left')
        
        for col, default_val in self.global_defaults_.items():
            X_merged[col] = X_merged[col].fillna(default_val)
            
        if self.customer_id_col in X_merged.columns:
            X_merged = X_merged.drop(columns=[self.customer_id_col])
            
        return X_merged


# =====================================================================
# TEMPORAL FEATURE EXTRACTOR
# =====================================================================
class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts hour, day, month, and year from datetime column text."""
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
            X_df = X_df.drop(columns=[self.datetime_col])
        return X_df


# =====================================================================
# PIPELINE ARCHITECTURE ENGINE
# =====================================================================
def execute_instructional_pipeline():
    print("Initiating Labeled Credit Risk Preprocessing Pipeline Engine...")
    
    script_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_directory)
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'data.csv')
    
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"❌ Raw dataset missing from path layout: {raw_data_path}")
        
    df = pd.read_csv(raw_data_path)
    print(f"📊 Raw Dataset Loaded. Shape: {df.shape}")
    
    # -----------------------------------------------------------------
    # STEP 1: CALCULATE RFM AND GENERATE TARGET PROXY VIA K-MEANS
    # -----------------------------------------------------------------
    print("Compiling global snapshot date and standalone RFM metrics...")
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    global_snapshot = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    rfm_profiles = df.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda x: (global_snapshot - x.max()).days),
        Frequency=('Amount', 'count'),
        Monetary=('Amount', 'sum')
    ).reset_index()
    
    # Scale features uniformly to prevent unit bias distortion in distance tracking
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm_profiles[['Recency', 'Frequency', 'Monetary']])
    
    print("Segmenting unique customers using K-Means clustering (k=3)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm_profiles['Cluster'] = kmeans.fit_predict(scaled_rfm)
    
    # Automated cluster inspection to tag the unengaged cluster group
    cluster_analysis = rfm_profiles.groupby('Cluster').agg({'Frequency': 'mean'})
    high_risk_cluster_id = cluster_analysis['Frequency'].idxmin()
    print(f"Automated Target Resolution: Identified Cluster {high_risk_cluster_id} as High-Risk Proxy.")
    
    # Generate and map the binary label vector
    rfm_profiles['is_high_risk'] = (rfm_profiles['Cluster'] == high_risk_cluster_id).astype(int)
    target_map = dict(zip(rfm_profiles['CustomerId'], rfm_profiles['is_high_risk']))
    
    # Establish our final target variable column
    df['is_high_risk'] = df['CustomerId'].map(target_map)
    y = df['is_high_risk'].values
    print(f" Proxy target 'is_high_risk' constructed! Default Class Distribution: {np.bincount(y)}")

    # -----------------------------------------------------------------
    # STEP 2: ORCHESTRATE THE PRIMARY FEATURES PIPELINE
    # -----------------------------------------------------------------
    X_features = df.drop(columns=['TransactionId', 'is_high_risk'])
    
    num_features = ['Amount', 'Value', 'PricingStrategy']
    cat_features = ['ProductCategory', 'ChannelId', 'ProviderId']
    
    num_features = [col for col in num_features if col in X_features.columns]
    cat_features = [col for col in cat_features if col in X_features.columns]
    
    # Dynamic schema tracking pipeline extensions
    extended_numeric_schema = num_features + [
        'Transaction_Hour', 'Transaction_Day', 'Transaction_Month', 'Transaction_Year',
        'Recency', 'Frequency', 'Monetary', 'Standard_Deviation_Transaction_Amount'
    ]
    
    # Standard numerical conveyor belt
    numeric_conveyor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Standard categorical conveyor belt
    categorical_conveyor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    base_preprocessor = ColumnTransformer(
        transformers=[
            ('numeric_belt', numeric_conveyor, extended_numeric_schema),
            ('categorical_belt', categorical_conveyor, cat_features)
        ],
        remainder='drop'
    )

    production_pipeline = Pipeline(steps=[
        ('rfm_and_volatility', CustomerBehaviorAggregator()),
        ('temporal_extractor', TemporalFeatureExtractor(datetime_col='TransactionStartTime')),
        ('woe_transformation', WOETransformer(feature_names=cat_features)), # Local WOE transformer
        ('math_preprocessor', base_preprocessor)
    ])
    
    print("⚙️ Executing fit_transform matrix engine across features...")
    processed_matrix = production_pipeline.fit_transform(X_features, y)
    
    # Standardize array mapping back to clean pandas schemas
    output_df = pd.DataFrame(processed_matrix) if isinstance(processed_matrix, np.ndarray) else processed_matrix.copy()
    output_df['is_high_risk'] = y
    
    # Export Labeled, Engineered Files to Processed Directory Layout
    output_directory = os.path.join(project_root, 'data', 'processed')
    os.makedirs(output_directory, exist_ok=True)
    output_save_file_path = os.path.join(output_directory, 'model_ready_data.csv')
    
    print(f" Saving model-ready analytical framework asset to: {output_save_file_path}")
    output_df.to_csv(output_save_file_path, index=False)
    print(f"Setup Complete! Operational Matrix Properties: {output_df.shape}\n")
    
if __name__ == "__main__":
    execute_instructional_pipeline()