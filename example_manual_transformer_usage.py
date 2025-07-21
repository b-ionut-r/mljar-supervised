#!/usr/bin/env python3
"""
Example usage of the manual_transformer integration with mljar-supervised.

This example shows how to use the UltraAdvancedSpatioTemporalFeatures transformer
as a preprocessing step in the AutoML pipeline.
"""

import pandas as pd
import numpy as np
from spatio_temporal import UltraAdvancedSpatioTemporalFeatures
from supervised import AutoML

# Example: Air pollution prediction with spatial-temporal features
def example_usage():
    """Example of using manual_transformer with AutoML"""
    
    # Load your spatiotemporal data
    # Your data should have columns: latitude, longitude, hour, day_of_week, month, day_of_year
    # For this example, we'll create some sample data
    
    # Sample data creation (replace with your actual data loading)
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'latitude': np.random.uniform(40.0, 41.0, n_samples),
        'longitude': np.random.uniform(-74.0, -73.0, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'day_of_year': np.random.randint(1, 366, n_samples),
        'temperature': np.random.normal(20, 10, n_samples),
        'humidity': np.random.uniform(30, 90, n_samples),
        'wind_speed': np.random.exponential(5, n_samples),
        # Target variable (e.g., air pollution level)
        'pollution_level': np.random.normal(50, 20, n_samples)
    }
    
    df = pd.DataFrame(data)
    X = df.drop('pollution_level', axis=1)
    y = df['pollution_level']
    
    # Create the advanced spatiotemporal feature transformer
    manual_transformer = UltraAdvancedSpatioTemporalFeatures(
        row_only=False,                    # Generate features across rows
        n_spatial_clusters=30,             # Number of spatial clusters
        n_temporal_clusters=10,            # Number of temporal clusters  
        use_distribution_matching=True,    # Use distribution matching
        test_distribution=None,            # Will be set automatically
        january_bridge_features=True,      # Add January bridge features
        n_geohash_neighbors=20,            # Number of geohash neighbors
        use_target_encoding=True,          # Use target encoding
        use_polynomial_features=True       # Add polynomial features
    )
    
    # Create AutoML with the manual transformer
    automl = AutoML(
        mode="Compete",                    # Use Compete mode for best performance
        total_time_limit=3600,             # 1 hour time limit
        algorithms=["Linear", "Random Forest", "Extra Trees", "LightGBM", "Xgboost"],
        train_ensemble=True,               # Enable ensemble
        explain_level=2,                   # Full explanations
        manual_transformer=manual_transformer,  # ← This is the new parameter!
        verbose=1
    )
    
    # Fit the model
    # The manual_transformer will be applied as the FIRST preprocessing step
    automl.fit(X, y)
    
    # Make predictions
    predictions = automl.predict(X)
    
    # The manual_transformer is automatically applied during prediction too!
    
    print("Manual transformer integration successful!")
    return automl

# Alternative usage patterns:

def example_with_custom_transformer():
    """Example with a simpler custom transformer"""
    
    class SimpleFeatureEngineer:
        def fit(self, X, y=None):
            # Learn any patterns from training data
            self.feature_means_ = X.select_dtypes(include=[np.number]).mean()
            return self
            
        def transform(self, X):
            # Add some simple engineered features
            X_new = X.copy()
            
            # Add interaction features
            if 'latitude' in X_new.columns and 'longitude' in X_new.columns:
                X_new['lat_lon_interaction'] = X_new['latitude'] * X_new['longitude']
                
            # Add distance from city center (example for NYC)
            if 'latitude' in X_new.columns and 'longitude' in X_new.columns:
                nyc_lat, nyc_lon = 40.7128, -74.0060
                X_new['distance_from_nyc'] = np.sqrt(
                    (X_new['latitude'] - nyc_lat)**2 + 
                    (X_new['longitude'] - nyc_lon)**2
                )
            
            # Add temporal features
            if 'hour' in X_new.columns:
                X_new['is_rush_hour'] = X_new['hour'].apply(
                    lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0
                )
                
            return X_new
    
    # Use it with AutoML
    custom_transformer = SimpleFeatureEngineer()
    
    # Your data loading here...
    # X, y = load_your_data()
    
    automl = AutoML(
        mode="Perform",
        manual_transformer=custom_transformer,  # Use your custom transformer
        total_time_limit=1800
    )
    
    # automl.fit(X, y)
    # predictions = automl.predict(X_test)
    
    print("Custom transformer example ready!")

def example_conditional_transformer():
    """Example showing conditional transformer usage"""
    
    # You can conditionally set the transformer
    use_advanced_features = True  # Set this based on your needs
    
    manual_transformer = None
    if use_advanced_features:
        manual_transformer = UltraAdvancedSpatioTemporalFeatures(
            n_spatial_clusters=20,
            use_target_encoding=True
        )
    
    automl = AutoML(
        mode="Explain",
        manual_transformer=manual_transformer,  # Can be None
        total_time_limit=600
    )
    
    print("Conditional transformer example ready!")

if __name__ == "__main__":
    print("Manual Transformer Usage Examples")
    print("=" * 50)
    
    print("\n1. Advanced Spatiotemporal Transformer Example:")
    try:
        example_usage()
    except Exception as e:
        print(f"Note: To run this example, install required dependencies: {e}")
    
    print("\n2. Custom Transformer Example:")
    example_with_custom_transformer()
    
    print("\n3. Conditional Transformer Example:")
    example_conditional_transformer()
    
    print("\nIntegration complete! You can now use manual transformers with AutoML.")