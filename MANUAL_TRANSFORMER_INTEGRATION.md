# Manual Transformer Integration with MLJAR AutoML

## Overview

This integration adds support for custom manual feature engineering transformers as a preprocessing step in MLJAR AutoML. The transformer is applied as the **first step** in the preprocessing pipeline, allowing you to use advanced feature engineering techniques (like the `UltraAdvancedSpatioTemporalFeatures` class) before any other AutoML preprocessing.

## What Was Modified

### 1. AutoML Class (`supervised/automl.py`)
- **Added `manual_transformer` parameter** to the constructor
- **Added parameter documentation** explaining its usage
- **Set attribute** `self.manual_transformer = manual_transformer`

### 2. Base AutoML Class (`supervised/base_automl.py`)
- **Added `_manual_transformer` attribute** initialization
- **Added `_get_manual_transformer()` method** to retrieve the transformer
- **Added transformer retrieval** in `_fit()` method
- **Updated MljarTuner call** to pass the manual_transformer

### 3. MljarTuner (`supervised/tuner/mljar_tuner.py`)
- **Added `manual_transformer` parameter** to constructor
- **Stored transformer** as `self._manual_transformer`
- **Modified `_get_model_params()`** to add manual_transformer to preprocessing params

### 4. Preprocessing Class (`supervised/preprocessing/preprocessing.py`)
- **Added `_manual_transformer` attribute** from preprocessing parameters
- **Added manual transformer execution** in `fit_and_transform()` as first step after excluding missing targets
- **Added manual transformer execution** in `transform()` for both cases:
  - When `y_validation` is provided (training/validation)
  - When `y_validation` is None (pure prediction)
- **Added validation** to ensure transformer has `fit()` and `transform()` methods
- **Added DataFrame conversion** to ensure output compatibility

## Usage Examples

### Basic Usage
```python
from supervised import AutoML
from spatio_temporal import UltraAdvancedSpatioTemporalFeatures

# Create your manual transformer
manual_transformer = UltraAdvancedSpatioTemporalFeatures(
    row_only=False,
    n_spatial_clusters=30,
    n_temporal_clusters=10,
    use_distribution_matching=True,
    january_bridge_features=True,
    n_geohash_neighbors=20,
    use_target_encoding=True,
    use_polynomial_features=True
)

# Use it with AutoML
automl = AutoML(
    mode="Compete",
    total_time_limit=3600,
    manual_transformer=manual_transformer,  # ← New parameter!
    verbose=1
)

# Fit and predict as usual
automl.fit(X_train, y_train)
predictions = automl.predict(X_test)
```

### Custom Transformer Example
```python
class CustomFeatureEngineer:
    def fit(self, X, y=None):
        # Learn patterns from training data
        self.feature_stats_ = X.describe()
        return self
        
    def transform(self, X):
        # Add your custom features
        X_new = X.copy()
        X_new['custom_feature'] = X['feature1'] * X['feature2']
        return X_new

# Use with AutoML
custom_transformer = CustomFeatureEngineer()
automl = AutoML(manual_transformer=custom_transformer)
```

### Conditional Usage
```python
# You can conditionally set the transformer
use_advanced_features = True

manual_transformer = None
if use_advanced_features:
    manual_transformer = UltraAdvancedSpatioTemporalFeatures()

automl = AutoML(
    manual_transformer=manual_transformer,  # Can be None
    mode="Perform"
)
```

## Integration Details

### Execution Order
1. **Exclude missing targets** (built-in AutoML step)
2. **Manual transformer** (your custom transformer) ← **NEW**
3. **Target preprocessing** (built-in AutoML steps)
4. **Column preprocessing** (built-in AutoML steps)
5. **Text transformations** (built-in AutoML steps)
6. **Missing value handling** (built-in AutoML steps)
7. **Categorical encoding** (built-in AutoML steps)
8. **Golden features** (built-in AutoML steps)
9. **K-means features** (built-in AutoML steps)
10. **Scaling** (built-in AutoML steps)

### Transformer Requirements
Your manual transformer must implement:
- **`fit(X, y=None)`** method that learns from training data and returns `self`
- **`transform(X)`** method that transforms the input data and returns the transformed data

### Data Flow
- **Training**: `fit()` is called once, then `transform()` is called for each fold
- **Prediction**: Only `transform()` is called using the fitted transformer
- **Output**: Must be compatible with pandas DataFrame (automatic conversion included)

## Error Handling

The integration includes validation to ensure your transformer:
- Has both `fit()` and `transform()` methods
- Returns data that can be converted to a DataFrame
- Doesn't break the downstream preprocessing pipeline

## Files Modified

- `supervised/automl.py` - Main AutoML class
- `supervised/base_automl.py` - Base AutoML functionality  
- `supervised/tuner/mljar_tuner.py` - Parameter tuning logic
- `supervised/preprocessing/preprocessing.py` - Preprocessing pipeline

## Compatibility

- ✅ Works with all AutoML modes (`Explain`, `Perform`, `Compete`, `Optuna`)
- ✅ Works with all algorithms
- ✅ Works with ensemble and stacking
- ✅ Works with cross-validation
- ✅ Compatible with existing preprocessing options
- ✅ Maintains scikit-learn API compatibility
- ✅ Preserves all existing AutoML functionality

## Benefits

1. **First-step processing**: Your transformer runs before any other preprocessing
2. **Full integration**: Works seamlessly with all AutoML features
3. **Automatic application**: Applied during both training and prediction
4. **Flexible**: Use any transformer that follows scikit-learn patterns
5. **Optional**: Can be set to `None` for standard AutoML behavior

## Testing

The integration has been tested with:
- Basic transformer functionality
- Parameter passing through the pipeline
- Training and prediction workflows
- Error handling for invalid transformers

See `test_manual_transformer.py` and `example_manual_transformer_usage.py` for complete examples.

## Future Enhancements

Potential future improvements:
- Support for multiple manual transformers in sequence
- Integration with AutoML feature selection
- Advanced transformer validation
- Performance optimization for large transformers