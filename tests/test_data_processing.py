import pytest
import pandas as pd
import numpy as np
from src.data_processing import TemporalFeatureExtractor, CustomerBehaviorAggregator

def test_temporal_feature_extractor_columns():
    """Unit Test 1: Verifies that the Temporal extractor creates all 4 required time columns."""
    # Create mock transactional data
    fake_data = pd.DataFrame({
        'TransactionStartTime': ['2026-06-01 14:30:00', '2026-06-02 09:15:00']
    })
    
    extractor = TemporalFeatureExtractor(datetime_col='TransactionStartTime')
    processed_df = extractor.fit_transform(fake_data)
    
    # Assert checks to verify structural changes
    assert 'Transaction_Hour' in processed_df.columns
    assert 'Transaction_Day' in processed_df.columns
    assert 'Transaction_Month' in processed_df.columns
    assert 'Transaction_Year' in processed_df.columns
    assert 'TransactionStartTime' not in processed_df.columns  # Original should be dropped

def test_customer_behavior_aggregator_output():
    """Unit Test 2: Verifies that the behavioral aggregator handles customer pooling correctly."""
    fake_transactions = pd.DataFrame({
        'CustomerId': ['Cust_A', 'Cust_A', 'Cust_B'],
        'Amount': [1000.0, 2000.0, 500.0],
        'TransactionStartTime': ['2026-01-01', '2026-01-02', '2026-01-03']
    })
    
    aggregator = CustomerBehaviorAggregator(customer_id_col='CustomerId', amount_col='Amount', datetime_col='TransactionStartTime')
    processed_df = aggregator.fit_transform(fake_transactions)
    
    # Assert that engineered columns exist
    assert 'Recency' in processed_df.columns
    assert 'Frequency' in processed_df.columns
    assert 'Monetary' in processed_df.columns
    assert 'Standard_Deviation_Transaction_Amount' in processed_df.columns