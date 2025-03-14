use ndarray::ShapeError;
use std::error::Error as StdError;
use std::fmt;

#[derive(Debug)]
pub enum ModelError {
    InvalidDimension(String),
    ShapeError(ShapeError),
    Network(String),
    InvalidConfig(String),
    Training(String),
    Prediction(String),
    Other(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ModelError::InvalidDimension(msg) => write!(f, "Invalid dimension: {}", msg),
            ModelError::ShapeError(err) => write!(f, "Shape error: {}", err),
            ModelError::Network(msg) => write!(f, "Network error: {}", msg),
            ModelError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            ModelError::Training(msg) => write!(f, "Training error: {}", msg),
            ModelError::Prediction(msg) => write!(f, "Prediction error: {}", msg),
            ModelError::Other(msg) => write!(f, "Other error: {}", msg),
        }
    }
}

impl StdError for ModelError {}

impl From<ShapeError> for ModelError {
    fn from(err: ShapeError) -> Self {
        ModelError::ShapeError(err)
    }
}

impl From<anyhow::Error> for ModelError {
    fn from(err: anyhow::Error) -> Self {
        ModelError::Other(err.to_string())
    }
} 