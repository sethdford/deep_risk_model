# Testing Guide for Deep Risk Model

This guide explains how to write and use tests for the Deep Risk Model, with a focus on making it accessible for junior engineers.

## Test Structure

The test suite is organized into three main categories:

1. **Unit Tests** - Test individual components in isolation
2. **Integration Tests** - Test component interactions
3. **End-to-End Tests** - Test complete workflows

## Test Utilities

The `test_utils.rs` module provides helper functions for testing:

### 1. Data Generation

```rust
pub fn generate_test_data(num_stocks: usize, num_features: usize, noise_level: f64) -> MarketData {
    // Generates synthetic market data for testing
    // Returns: MarketData with random features and returns
}
```

Usage example:
```rust
#[test]
fn test_data_generation() {
    let market_data = generate_test_data(100, 10, 0.1);
    assert_eq!(market_data.features.shape()[0], 100);
    assert_eq!(market_data.features.shape()[2], 10);
}
```

### 2. Factor Quality Verification

```rust
pub fn verify_factor_quality(
    t_stats: &ArrayD<f64>,
    vif: &ArrayD<f64>,
    auto_correlation: &ArrayD<f64>
) -> bool {
    // Verifies the quality of generated risk factors
    // Returns: true if all quality metrics pass
}
```

Usage example:
```rust
#[test]
fn test_factor_quality() {
    let t_stats = Array::from_vec(vec![2.5, -2.8, 3.2]);
    let vif = Array::from_vec(vec![1.2, 2.3, 3.1]);
    let auto_correlation = Array::from_vec(vec![0.3, -0.5, 0.7]);
    
    assert!(verify_factor_quality(
        &t_stats.into_dyn(),
        &vif.into_dyn(),
        &auto_correlation.into_dyn()
    ));
}
```

### 3. Covariance Matrix Verification

```rust
pub fn verify_covariance_matrix(covariance: &ArrayD<f64>, num_stocks: usize) -> bool {
    // Verifies the properties of a covariance matrix
    // Returns: true if matrix is valid
}
```

Usage example:
```rust
#[test]
fn test_covariance_verification() {
    let num_stocks = 30;
    let covariance = Array::random((num_stocks, num_stocks), StandardNormal);
    let covariance = covariance.dot(&covariance.t()); // Make it positive definite
    
    assert!(verify_covariance_matrix(&covariance.into_dyn(), num_stocks));
}
```

## Writing Tests

### 1. Unit Tests

Unit tests should be placed in the same file as the code they're testing:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_specific_function() {
        // Arrange
        let input = /* setup test data */;
        
        // Act
        let result = function_under_test(input);
        
        // Assert
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), expected_value);
    }
}
```

### 2. Integration Tests

Integration tests go in the `tests` directory:

```rust
#[test]
fn test_component_interaction() {
    // Setup
    let config = ModelConfig {
        input_size: 10,
        hidden_size: 64,
        num_heads: 4,
        head_dim: 16,
        num_layers: 2,
        output_size: 10,
    };
    let mut model = DeepRiskModel::new(&config)?;
    let data = generate_test_data(100, 10, 0.1);
    
    // Test interaction
    model.train(&data).await.unwrap();
    let factors = model.generate_factors(&data).await.unwrap();
    
    // Verify
    assert!(verify_factor_quality(
        &factors.t_stats,
        &factors.vif,
        &factors.auto_correlation
    ));
}
```

### 3. End-to-End Tests

E2E tests test complete workflows:

```rust
#[tokio::test]
async fn test_model_workflow() -> Result<(), ModelError> {
    // Initialize
    let config = ModelConfig {
        input_size: 10,
        hidden_size: 64,
        num_heads: 4,
        head_dim: 16,
        num_layers: 2,
        output_size: 10,
    };
    let mut model = DeepRiskModel::new(&config)?;
    
    // Generate test data
    let data = generate_test_data(100, 10, 0.1);
    
    // Train model
    model.train(&data).await?;
    
    // Generate factors
    let factors = model.generate_factors(&data).await?;
    
    // Verify results
    assert!(verify_factor_quality(
        &factors.t_stats,
        &factors.vif,
        &factors.auto_correlation
    ));
    
    // Estimate covariance
    let covariance = model.estimate_covariance(&data).await?;
    assert!(verify_covariance_matrix(&covariance, 100));
    
    Ok(())
}
```

## Best Practices

1. **Test Naming**
   - Use descriptive names that explain what's being tested
   - Follow the pattern `test_<component>_<behavior>`

2. **Test Organization**
   - Group related tests in the same module
   - Use `#[cfg(test)]` for test-only code
   - Keep test setup code in helper functions

3. **Assertions**
   - Use specific assertions (`assert_eq!`, `assert_ne!`)
   - Include descriptive messages for failures
   - Test both success and failure cases

4. **Async Testing**
   - Use `#[tokio::test]` for async tests
   - Handle errors appropriately
   - Use `Result` return types

5. **Test Data**
   - Use `generate_test_data` for consistent test data
   - Create edge cases and boundary conditions
   - Use realistic data shapes and values

## Running Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_model_workflow

# Run tests with output
cargo test -- --nocapture

# Run tests in parallel
cargo test -- --test-threads=4
```

## Debugging Tests

1. **Print Debug Information**
   ```rust
   #[test]
   fn test_with_debug() {
       let result = function_under_test();
       println!("Debug: {:?}", result);
       assert!(result.is_ok());
   }
   ```

2. **Use Test Attributes**
   ```rust
   #[test]
   #[should_panic(expected = "Invalid input")]
   fn test_error_case() {
       function_under_test(invalid_input);
   }
   ```

3. **Test Timeouts**
   ```rust
   #[tokio::test]
   #[timeout(5000)] // 5 second timeout
   async fn test_with_timeout() {
       // Test code
   }
   ```

## Common Pitfalls

1. **Async/Await Issues**
   - Remember to use `#[tokio::test]` for async tests
   - Use `.await` for async operations
   - Handle errors properly

2. **Array Dimension Mismatches**
   - Verify array shapes before operations
   - Use `into_dyn()` for dynamic arrays
   - Check array dimensions in assertions

3. **Test Isolation**
   - Each test should be independent
   - Clean up resources after tests
   - Don't rely on test execution order

## Further Reading

- [Rust Testing Documentation](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [Tokio Testing Guide](https://tokio.rs/tokio/tutorial/testing)
- [ndarray Documentation](https://docs.rs/ndarray/latest/ndarray/) 