use ndarray::{Array1, Array2, Array3, Axis, s};
use ndarray::linalg::general_mat_mul;
use std::ops::AddAssign;
use crate::error::ModelError;
use std::sync::Arc;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use rand::thread_rng;

/// Graph Attention Network module for cross-sectional feature extraction.
/// 
/// This module implements a simplified version of the Graph Attention Network
/// that processes cross-sectional relationships between assets. It uses
/// multi-head attention to capture different aspects of the relationships.
pub struct GATModule {
    w_query: Array2<f32>,
    w_key: Array2<f32>,
    w_value: Array2<f32>,
    hidden_size: usize,
    num_heads: usize,
    w_out: Arc<Array2<f32>>,    // Output transformation
    b_out: Arc<Array1<f32>>,    // Output bias
}

impl GATModule {
    /// Creates a new GAT module.
    /// 
    /// # Arguments
    /// * `hidden_size` - Dimension of hidden state per head
    /// * `num_heads` - Number of attention heads
    /// 
    /// # Returns
    /// * `Result<Self, ModelError>` - The initialized GAT module
    pub fn new(hidden_size: usize, num_heads: usize) -> Result<Self, ModelError> {
        let head_dim = hidden_size / num_heads;
        if head_dim * num_heads != hidden_size {
            return Err(ModelError::InvalidDimension("hidden_size must be divisible by num_heads".to_string()));
        }

        let normal = Normal::new(0.0, 0.1).map_err(|e| ModelError::InvalidDimension(e.to_string()))?;
        let mut rng = thread_rng();

        let w_query = Array2::random_using((hidden_size, hidden_size), normal.clone(), &mut rng).t().to_owned();
        let w_key = Array2::random_using((hidden_size, hidden_size), normal.clone(), &mut rng).t().to_owned();
        let w_value = Array2::random_using((hidden_size, hidden_size), normal.clone(), &mut rng).t().to_owned();
        let w_out = Arc::new(Array2::random_using((hidden_size, hidden_size), normal.clone(), &mut rng).t().to_owned());
        let b_out = Arc::new(Array1::zeros(hidden_size));

        Ok(Self {
            w_query,
            w_key,
            w_value,
            hidden_size,
            num_heads,
            w_out,
            b_out,
        })
    }

    fn compute_attention(&self, input: &Array3<f32>) -> Result<Array3<f32>, ModelError> {
        let (batch_size, num_nodes, _) = input.dim();
        let mut attention = Array3::zeros((batch_size, num_nodes, num_nodes));
        
        for b in 0..batch_size {
            let input_slice = input.slice(s![b, .., ..]);
            let mut q = Array2::zeros((num_nodes, self.hidden_size));
            let mut k = Array2::zeros((num_nodes, self.hidden_size));
            
            // Project input to query and key spaces
            general_mat_mul(1.0, &input_slice, &self.w_query, 0.0, &mut q);
            general_mat_mul(1.0, &input_slice, &self.w_key, 0.0, &mut k);
            
            // Compute attention scores
            let mut scores = Array2::zeros((num_nodes, num_nodes));
            general_mat_mul(1.0, &q, &k.t(), 0.0, &mut scores);
            
            let scale = (self.hidden_size as f32).sqrt();
            attention.slice_mut(s![b, .., ..]).assign(&(&scores / scale));
        }
        
        Ok(attention)
    }

    fn apply_attention(&self, input: &Array3<f32>, attention: &Array3<f32>) -> Result<Array3<f32>, ModelError> {
        let (batch_size, num_nodes, _) = input.dim();
        let mut output = Array3::zeros((batch_size, num_nodes, self.hidden_size));
        
        for b in 0..batch_size {
            let input_slice = input.slice(s![b, .., ..]);
            let att_slice = attention.slice(s![b, .., ..]);
            let mut v = Array2::zeros((num_nodes, self.hidden_size));
            
            // Project input to value space
            general_mat_mul(1.0, &input_slice, &self.w_value, 0.0, &mut v);
            
            // Apply attention to values
            general_mat_mul(1.0, &att_slice, &v, 0.0, &mut output.slice_mut(s![b, .., ..]));
        }
        
        Ok(output)
    }

    /// Forward pass through the GAT module.
    /// 
    /// # Arguments
    /// * `input` - Input tensor of shape [batch_size, num_nodes, hidden_size]
    /// 
    /// # Returns
    /// * `Result<Array3<f32>, ModelError>` - Output tensor or error
    pub fn forward(&self, input: &Array3<f32>) -> Result<Array3<f32>, ModelError> {
        let attention = self.compute_attention(input)?;
        
        // Apply softmax
        let attention = attention.mapv(|x| x.exp());
        let attention_sum = attention.sum_axis(Axis(2));
        let attention = &attention / &attention_sum.insert_axis(Axis(2));
        
        self.apply_attention(input, &attention)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Normal;

    #[test]
    fn test_gat_forward() {
        let batch_size = 2;
        let num_nodes = 3;
        let hidden_size = 64;
        let num_heads = 4;

        let gat = GATModule::new(hidden_size, num_heads).unwrap();
        let input = Array3::random((batch_size, num_nodes, hidden_size), Normal::new(0.0, 1.0).unwrap());
        
        let output = gat.forward(&input).unwrap();
        assert_eq!(output.shape(), &[batch_size, num_nodes, hidden_size]);
    }
}