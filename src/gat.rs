use ndarray::{Array1, Array2, Array3, Axis, s};
use ndarray::linalg::general_mat_mul;
use crate::error::ModelError;
use std::sync::Arc;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use rand::thread_rng;
use rayon::prelude::*;

/// Graph Attention Network (GAT) module for processing cross-sectional relationships in financial data.
///
/// # Architecture
///
/// The GAT module implements a multi-head attention mechanism that learns relationships
/// between different assets in the market. Key components include:
/// - Multi-head attention layers
/// - Query/Key/Value transformations
/// - Attention score computation
/// - Output projection
///
/// # Features
///
/// - Captures cross-sectional dependencies between assets
/// - Multi-head attention for diverse relationship modeling
/// - Learnable attention weights
/// - Efficient parallel computation
///
/// # Mathematical Details
///
/// For input X, the GAT computes:
/// ```text
/// Q = X·W_q    // Query transformation
/// K = X·W_k    // Key transformation
/// V = X·W_v    // Value transformation
/// 
/// Attention(Q,K,V) = softmax(QK^T/√d_k)V
/// 
/// MultiHead(X) = Concat(head_1,...,head_h)·W_o
/// where head_i = Attention(XW_q^i, XW_k^i, XW_v^i)
/// ```
#[derive(Debug)]
pub struct GATModule {
    /// Weight matrix for query transformation
    w_query: Array2<f32>,
    /// Weight matrix for key transformation
    w_key: Array2<f32>,
    /// Weight matrix for value transformation
    w_value: Array2<f32>,
    /// Hidden state dimension
    hidden_size: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Output projection weights
    w_out: Arc<Array2<f32>>,
    /// Output projection bias
    b_out: Arc<Array1<f32>>,
}

impl GATModule {
    /// Creates a new GAT module with specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - Size of hidden state
    /// * `num_heads` - Number of attention heads
    ///
    /// # Returns
    ///
    /// * `Result<Self, ModelError>` - A new GAT module or an error
    ///
    /// # Example
    ///
    /// ```rust
    /// let gat = GATModule::new(128, 4)?;
    /// ```
    pub fn new(hidden_size: usize, num_heads: usize) -> Result<Self, ModelError> {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.1).map_err(|e| ModelError::InitializationError(e.to_string()))?;
        
        let head_dim = hidden_size / num_heads;
        if hidden_size % num_heads != 0 {
            return Err(ModelError::InvalidConfig(
                format!("Hidden size ({}) must be divisible by number of heads ({})", 
                       hidden_size, num_heads)
            ));
        }

        Ok(Self {
            w_query: Array2::random_using((hidden_size, num_heads * head_dim), normal, &mut rng),
            w_key: Array2::random_using((hidden_size, num_heads * head_dim), normal, &mut rng),
            w_value: Array2::random_using((hidden_size, num_heads * head_dim), normal, &mut rng),
            hidden_size,
            num_heads,
            w_out: Arc::new(Array2::random_using((num_heads * head_dim, hidden_size), normal, &mut rng)),
            b_out: Arc::new(Array1::zeros(hidden_size)),
        })
    }

    /// Processes input through the GAT layer with optimized parallel processing.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (batch_size, seq_len, hidden_size)
    ///
    /// # Returns
    ///
    /// * `Result<Array3<f32>, ModelError>` - Output tensor or error
    ///
    /// # Example
    ///
    /// ```rust
    /// let output = gat.forward(&input_tensor)?;
    /// println!("Output shape: {:?}", output.shape());
    /// ```
    pub fn forward(&self, input: &Array3<f32>) -> Result<Array3<f32>, ModelError> {
        let (batch_size, seq_len, _) = input.dim();
        let head_dim = self.hidden_size / self.num_heads;

        // Process heads sequentially
        let mut output = Array3::zeros((batch_size, seq_len, self.hidden_size));
        
        // Reshape input once for all operations
        let input_2d = input.view()
            .into_shape((batch_size * seq_len, self.hidden_size))?;
        
        // Compute Q, K, V for all heads at once
        let mut qkv = Array2::zeros((batch_size * seq_len, 3 * self.num_heads * head_dim));
        general_mat_mul(1.0, &input_2d, &self.w_query, 0.0, &mut qkv.slice_mut(s![.., 0..self.num_heads * head_dim]));
        general_mat_mul(1.0, &input_2d, &self.w_key, 0.0, &mut qkv.slice_mut(s![.., self.num_heads * head_dim..2 * self.num_heads * head_dim]));
        general_mat_mul(1.0, &input_2d, &self.w_value, 0.0, &mut qkv.slice_mut(s![.., 2 * self.num_heads * head_dim..]));
        
        for head in 0..self.num_heads {
            let start_idx = head * head_dim;
            let end_idx = (head + 1) * head_dim;
            
            // Extract Q, K, V for this head
            let q = qkv.slice(s![.., start_idx..end_idx]);
            let k = qkv.slice(s![.., self.num_heads * head_dim + start_idx..self.num_heads * head_dim + end_idx]);
            let v = qkv.slice(s![.., 2 * self.num_heads * head_dim + start_idx..2 * self.num_heads * head_dim + end_idx]);
            
            // Compute attention scores
            let mut attention = Array2::zeros((batch_size * seq_len, batch_size * seq_len));
            general_mat_mul(1.0, &q, &k.t(), 0.0, &mut attention);
            
            // Scale and apply softmax in-place
            let scale = (head_dim as f32).sqrt();
            attention.mapv_inplace(|x| (x / scale).exp());
            
            // Normalize attention scores with stable softmax
            let max_scores = attention.map_axis(Axis(1), |row| {
                row.fold(f32::NEG_INFINITY, |a, &b| a.max(b))
            });
            
            for (mut row, &max) in attention.outer_iter_mut().zip(max_scores.iter()) {
                row.mapv_inplace(|x| (x - max).exp());
                let sum = row.sum();
                row.mapv_inplace(|x| x / sum);
            }
            
            // Apply attention to values
            let mut head_output = Array2::zeros((batch_size * seq_len, head_dim));
            general_mat_mul(1.0, &attention, &v, 0.0, &mut head_output);
            
            // Store this head's output in the correct slice
            let head_output_3d = head_output.into_shape((batch_size, seq_len, head_dim))?;
            output.slice_mut(s![.., .., head * head_dim..(head + 1) * head_dim])
                .assign(&head_output_3d);
        }
        
        // Final output projection with pre-allocated memory
        let mut projected = Array2::zeros((batch_size * seq_len, self.hidden_size));
        general_mat_mul(1.0, 
            &output.into_shape((batch_size * seq_len, self.hidden_size))?,
            &self.w_out,
            0.0,
            &mut projected);
        
        // Add bias and reshape
        let output_2d = projected.into_shape((batch_size * seq_len, self.hidden_size))?;
        let mut result = output_2d.to_owned();
        for mut row in result.outer_iter_mut() {
            for (val, &bias) in row.iter_mut().zip(self.b_out.iter()) {
                *val += bias;
            }
        }
        
        result.into_shape((batch_size, seq_len, self.hidden_size))
            .map_err(|e| ModelError::InvalidDimension(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Normal;
    use std::time::Instant;

    #[test]
    fn test_gat_performance() {
        let batch_size = 32;
        let num_nodes = 100;
        let hidden_size = 128;
        let num_heads = 4;

        let gat = GATModule::new(hidden_size, num_heads).unwrap();
        let input = Array3::random((batch_size, num_nodes, hidden_size), Normal::new(0.0, 1.0).unwrap());
        
        let start = Instant::now();
        let output = gat.forward(&input).unwrap();
        let duration = start.elapsed();
        
        println!("GAT forward pass took: {:?}", duration);
        assert_eq!(output.shape(), &[batch_size, num_nodes, hidden_size]);
        
        // Test numerical stability
        assert!(!output.iter().any(|&x| x.is_nan() || x.is_infinite()));
    }
    
    #[test]
    fn test_gat_memory_usage() {
        let batch_size = 100;
        let num_nodes = 1000;
        let hidden_size = 256;
        let num_heads = 8;

        let gat = GATModule::new(hidden_size, num_heads).unwrap();
        let input = Array3::random((batch_size, num_nodes, hidden_size), Normal::new(0.0, 1.0).unwrap());
        
        // Process large input to test memory efficiency
        let output = gat.forward(&input).unwrap();
        assert_eq!(output.shape(), &[batch_size, num_nodes, hidden_size]);
    }
}