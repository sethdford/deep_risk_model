use ndarray::{s, Array1, Array2, Array3, Axis};
use ndarray::concatenate;
use ndarray_rand::{RandomExt, rand_distr::Normal};
use std::sync::Arc;
use anyhow::Result;
use rand::thread_rng;
use rayon::prelude::*;

use crate::ModelError;

/// GRU (Gated Recurrent Unit) module implementation
pub struct GRUModule {
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    
    // Reset gate parameters
    w_ir: Arc<Array2<f32>>,  // Input to reset gate weights
    w_hr: Arc<Array2<f32>>,  // Hidden to reset gate weights
    b_r: Arc<Array1<f32>>,   // Reset gate bias
    
    // Update gate parameters
    w_iz: Arc<Array2<f32>>,  // Input to update gate weights
    w_hz: Arc<Array2<f32>>,  // Hidden to update gate weights
    b_z: Arc<Array1<f32>>,   // Update gate bias
    
    // New gate parameters
    w_in: Arc<Array2<f32>>,  // Input to new gate weights
    w_hn: Arc<Array2<f32>>,  // Hidden to new gate weights
    b_n: Arc<Array1<f32>>,   // New gate bias
}

impl GRUModule {
    /// Creates a new GRU module.
    /// 
    /// # Arguments
    /// * `input_size` - Dimension of input features
    /// * `hidden_size` - Dimension of hidden state
    /// * `num_layers` - Number of GRU layers
    /// 
    /// # Returns
    /// * `Result<Self>` - The initialized GRU module
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Result<Self> {
        let normal = Normal::new(0.0, 0.1)?;
        let mut rng = thread_rng();
        
        // Initialize reset gate parameters
        let w_ir = Arc::new(Array2::random_using((hidden_size, input_size), normal.clone(), &mut rng));
        let w_hr = Arc::new(Array2::random_using((hidden_size, hidden_size), normal.clone(), &mut rng));
        let b_r = Arc::new(Array1::zeros(hidden_size));
        
        // Initialize update gate parameters
        let w_iz = Arc::new(Array2::random_using((hidden_size, input_size), normal.clone(), &mut rng));
        let w_hz = Arc::new(Array2::random_using((hidden_size, hidden_size), normal.clone(), &mut rng));
        let b_z = Arc::new(Array1::zeros(hidden_size));
        
        // Initialize new gate parameters
        let w_in = Arc::new(Array2::random_using((hidden_size, input_size), normal.clone(), &mut rng));
        let w_hn = Arc::new(Array2::random_using((hidden_size, hidden_size), normal.clone(), &mut rng));
        let b_n = Arc::new(Array1::zeros(hidden_size));
        
        Ok(Self {
            input_size,
            hidden_size,
            num_layers,
            w_ir,
            w_hr,
            b_r,
            w_iz,
            w_hz,
            b_z,
            w_in,
            w_hn,
            b_n,
        })
    }

    /// Forward pass through the GRU module.
    /// 
    /// # Arguments
    /// * `input` - Input tensor of shape [batch_size, seq_len, input_size]
    /// * `hidden_state` - Optional initial hidden state
    /// 
    /// # Returns
    /// * `Result<Array3<f32>, ModelError>` - Output tensor
    pub fn forward(&self, input: &Array3<f32>, _hidden: Option<Array2<f32>>) -> Result<Array3<f32>, ModelError> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let hidden_size = self.hidden_size;
        
        // Pre-allocate output and hidden state
        let mut output = Array3::<f32>::zeros((batch_size, seq_len, hidden_size));
        let mut h = Array2::<f32>::zeros((batch_size, hidden_size));
        
        // Process sequence in chunks for better cache utilization
        const CHUNK_SIZE: usize = 16;
        for chunk_start in (0..seq_len).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(seq_len);
            
            // Process each timestep in the chunk
            for t in chunk_start..chunk_end {
                let x_t = input.slice(s![.., t, ..]);
                
                // Update gate
                let u = (x_t.dot(&self.w_iz.as_ref().to_owned()) + h.dot(&self.w_hz.as_ref().to_owned()) + &self.b_z.as_ref().to_owned())
                    .mapv(|x| 1.0 / (1.0 + (-x).exp()));
                
                // Reset gate
                let r = (x_t.dot(&self.w_ir.as_ref().to_owned()) + h.dot(&self.w_hr.as_ref().to_owned()) + &self.b_r.as_ref().to_owned())
                    .mapv(|x| 1.0 / (1.0 + (-x).exp()));
                
                // Candidate hidden state
                let h_candidate = (x_t.dot(&self.w_in.as_ref().to_owned()) + &(r * &h).dot(&self.w_hn.as_ref().to_owned()) + &self.b_n.as_ref().to_owned())
                    .mapv(|x| x.tanh());
                
                // Update hidden state using element-wise operations
                h = &u * &h_candidate + &(&Array2::from_elem(u.raw_dim(), 1.0) - &u) * &h;
                
                // Store output
                output.slice_mut(s![.., t, ..]).assign(&h);
            }
        }
        
        Ok(output)
    }

    /// Initialize hidden state for the GRU module.
    /// 
    /// # Arguments
    /// * `batch_size` - Batch size of the input data
    /// 
    /// # Returns
    /// * `Array2<f32>` - Initialized hidden state
    pub fn init_hidden(&self, batch_size: usize) -> Array2<f32> {
        Array2::zeros((batch_size, self.hidden_size))
    }
}

// Helper activation functions
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh(x: f32) -> f32 {
    x.tanh()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_gru_forward() {
        let gru = GRUModule::new(20, 20, 1).unwrap();
        
        let batch_size = 32;
        let seq_len = 50;
        let input_shape = [batch_size, seq_len, 20];
        
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let input = Array3::random_using(input_shape, Normal::new(0.0, 1.0).unwrap(), &mut rng);
        
        let output = gru.forward(&input, None).unwrap();
        
        assert_eq!(output.shape(), &[batch_size, seq_len, 20]);
    }
}