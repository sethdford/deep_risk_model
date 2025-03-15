use ndarray::ShapeError;
use std::io;
use thiserror::Error;

/// Custom error types for the Deep Risk Model.
///
/// This enum defines all possible errors that can occur during model operations,
/// providing detailed context and appropriate error handling for each case.
///
/// # Examples
///
/// ```rust
/// use deep_risk_model::error::ModelError;
///
/// fn initialize_model() -> Result<(), ModelError> {
///     if invalid_config {
///         return Err(ModelError::InvalidConfig("Invalid hidden size".to_string()));
///     }
///     Ok(())
/// }
/// ```
#[derive(Error, Debug)]
pub enum ModelError {
    /// Error during model initialization (e.g., invalid parameters)
    #[error("Initialization error: {0}")]
    InitializationError(String),
    
    /// Error due to invalid dimensions in tensor operations
    #[error("Invalid dimension: {0}")]
    InvalidDimension(String),
    
    /// Error due to invalid configuration parameters
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    
    /// Error during model training (e.g., gradient explosion)
    #[error("Training error: {0}")]
    TrainingError(String),
    
    /// Error during data preprocessing or validation
    #[error("Data error: {0}")]
    DataError(String),
    
    /// Error during model inference or prediction
    #[error("Inference error: {0}")]
    InferenceError(String),
    
    /// Error during I/O operations (e.g., loading/saving model)
    #[error("IO error: {0}")]
    IO(#[from] io::Error),
    
    /// Error due to shape errors in ndarray
    #[error("Shape error: {0}")]
    Shape(#[from] ShapeError),
    
    /// Error due to parse errors
    #[error("Parse error: {0}")]
    Parse(#[from] std::num::ParseFloatError),
    
    /// Error due to other errors
    #[error("Other error: {0}")]
    Other(String),
    
    /// Error due to anyhow errors
    #[error(transparent)]
    AnyhowError(#[from] anyhow::Error),
}

impl From<String> for ModelError {
    fn from(error: String) -> Self {
        ModelError::Other(error)
    }
}