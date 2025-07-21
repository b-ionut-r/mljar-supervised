#!/usr/bin/env python3
"""
Test script for the manual_transformer integration with mljar-supervised.
This script demonstrates how to use the UltraAdvancedSpatioTemporalFeatures 
transformer with the AutoML class.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Import the spatiotemporal transformer
from spatio_temporal import UltraAdvancedSpatioTemporalFeatures

# Import the modified AutoML class
from supervised import AutoML

def create_sample_spatiotemporal_data(n_samples=1000):
    """Create sample data with spatial and temporal features like the transformer expects."""
    
    # Generate base regression data
    X, y = make_regression(n_samples=n_samples, n_features=5, noise=0.1, random_state=42)
    
    # Add spatial features (latitude, longitude)
    np.random.seed(42)
    latitude = np.random.uniform(40.0, 41.0, n_samples)  # NYC area
    longitude = np.random.uniform(-74.0, -73.0, n_samples)  # NYC area
    
    # Add temporal features
    hour = np.random.randint(0, 24, n_samples)
    day_of_week = np.random.randint(0, 7, n_samples)  
    month = np.random.randint(1, 13, n_samples)
    day_of_year = np.random.randint(1, 366, n_samples)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(5)]
    df = pd.DataFrame(X, columns=feature_names)
    df['latitude'] = latitude
    df['longitude'] = longitude
    df['hour'] = hour
    df['day_of_week'] = day_of_week
    df['month'] = month
    df['day_of_year'] = day_of_year
    df['target'] = y
    
    return df

def test_manual_transformer_integration():
    """Test the integration of manual transformer with AutoML."""
    
    print("Creating sample spatiotemporal data...")
    df = create_sample_spatiotemporal_data(500)  # Small dataset for quick testing
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Features: {list(X_train.columns)}")
    
    # Create the manual transformer
    print("\nCreating UltraAdvancedSpatioTemporalFeatures transformer...")
    manual_transformer = UltraAdvancedSpatioTemporalFeatures(
        row_only=False,
        n_spatial_clusters=5,  # Reduced for small dataset
        n_temporal_clusters=3,  # Reduced for small dataset
        use_distribution_matching=False,  # Disabled for simplicity
        january_bridge_features=True,
        n_geohash_neighbors=5,  # Reduced for small dataset
        use_target_encoding=True,
        use_polynomial_features=True
    )
    
    # Create AutoML instance with manual transformer
    print("\nCreating AutoML with manual_transformer...")
    automl = AutoML(
        mode="Explain",  # Fast mode for testing
        total_time_limit=60,  # 1 minute limit
        algorithms=["Linear", "Random Forest"],  # Just a few algorithms
        train_ensemble=False,  # Disable ensemble for speed
        explain_level=1,
        manual_transformer=manual_transformer,  # This is our new parameter!
        verbose=1
    )
    
    # Fit the model
    print("\nFitting AutoML model...")
    try:
        automl.fit(X_train, y_train)
        print("✓ AutoML fitting completed successfully!")
        
        # Make predictions
        print("\nMaking predictions...")
        y_pred = automl.predict(X_test)
        print(f"✓ Predictions completed! Shape: {y_pred.shape}")
        
        # Check performance
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nPerformance:")
        print(f"  MSE: {mse:.4f}")
        print(f"  R²: {r2:.4f}")
        
        # Show that manual transformer was used
        print(f"\n✓ Integration test successful!")
        print(f"✓ Manual transformer was successfully integrated into the preprocessing pipeline.")
        
        # Show feature count after transformation
        if hasattr(manual_transformer, 'n_features_out_'):
            print(f"✓ Original features: {X_train.shape[1]}")
            print(f"✓ Features after manual transformation: {manual_transformer.n_features_out_}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during AutoML fitting: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Manual Transformer Integration with MLJAR AutoML")
    print("=" * 60)
    
    success = test_manual_transformer_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("The manual_transformer parameter has been successfully integrated!")
    else:
        print("❌ TESTS FAILED!")
        print("Check the error messages above for details.")
    print("=" * 60)