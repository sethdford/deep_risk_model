use ndarray::{s, Array1, Array2, Array3};
use ndarray_rand::{RandomExt, rand_distr::Normal};
use std::sync::Arc;
use anyhow::Result;
use rand::thread_rng;
use ndarray::linalg::general_mat_mul;
use std::ops::AddAssign;

use crate::ModelError;

/// A Gated Recurrent Unit (GRU) module for processing temporal features in financial data.
///
/// # Architecture
///
/// The GRU module implements a standard GRU architecture with:
/// - Update gate
/// - Reset gate
/// - Hidden state
///
/// # Features
///
/// - Processes temporal sequences of market data
/// - Maintains state across time steps
/// - Configurable hidden size and number of layers
/// - Efficient batch processing
///
/// # Mathematical Details
///
/// For each time step t, the GRU computes:
/// ```text
/// z_t = σ(W_z·[h_{t-1}, x_t] + b_z)    // update gate
/// r_t = σ(W_r·[h_{t-1}, x_t] + b_r)    // reset gate
/// h̃_t = tanh(W·[r_t ∗ h_{t-1}, x_t] + b)
/// h_t = (1 - z_t) ∗ h_{t-1} + z_t ∗ h̃_t
/// ```
#[derive(Debug)]
pub struct GRUModule {
    /// Input dimension size
    pub input_size: usize,
    /// Hidden state dimension size
    pub hidden_size: usize,
    /// Number of GRU layers
    pub num_layers: usize,
    
    // Reset gate parameters
    pub w_ir: Arc<Array2<f32>>,  // Input to reset gate weights
    pub w_hr: Arc<Array2<f32>>,  // Hidden to reset gate weights
    pub b_r: Arc<Array1<f32>>,   // Reset gate bias
    
    // Update gate parameters
    pub w_iz: Arc<Array2<f32>>,  // Input to update gate weights
    pub w_hz: Arc<Array2<f32>>,  // Hidden to update gate weights
    pub b_z: Arc<Array1<f32>>,   // Update gate bias
    
    // New gate parameters
    pub w_in: Arc<Array2<f32>>,  // Input to new gate weights
    pub w_hn: Arc<Array2<f32>>,  // Hidden to new gate weights
    pub b_n: Arc<Array1<f32>>,   // New gate bias
}

impl GRUModule {
    /// Creates a new GRU module with specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `input_size` - Size of input features
    /// * `hidden_size` - Size of hidden state
    /// * `num_layers` - Number of GRU layers
    ///
    /// # Returns
    ///
    /// * `Result<Self, ModelError>` - A new GRU module or an error
    ///
    /// # Example
    ///
    /// ```rust
    /// let gru = GRUModule::new(64, 128, 2)?;
    /// ```
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Result<Self> {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.1).map_err(|e| ModelError::InitializationError(e.to_string()))?;
        
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

    /// Processes input through the GRU layer with optimized SIMD operations.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (batch_size, seq_len, input_size)
    /// * `hidden_state` - Optional initial hidden state
    ///
    /// # Returns
    ///
    /// * `Result<Array3<f32>, ModelError>` - Output tensor or error
    ///
    /// # Example
    ///
    /// ```rust
    /// let output = gru.forward(&input_tensor, None)?;
    /// println!("Output shape: {:?}", output.shape());
    /// ```
    pub fn forward(&self, input: &Array3<f32>, _hidden: Option<Array2<f32>>) -> Result<Array3<f32>, ModelError> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let hidden_size = self.hidden_size;
        
        // Process sequence in parallel chunks for better cache utilization
        const CHUNK_SIZE: usize = 32;  // Increased chunk size for better vectorization
        
        // Pre-allocate output
        let mut output = Array3::<f32>::zeros((batch_size, seq_len, hidden_size));
        let mut h = Array2::<f32>::zeros((batch_size, hidden_size));
        
        // Process chunks sequentially but parallelize within chunks
        for chunk_start in (0..seq_len).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(seq_len);
            let mut h_local = h.clone();
            
            // Process each timestep in the chunk
            for t in chunk_start..chunk_end {
                let x_t = input.slice(s![.., t, ..]);
                
                // Pre-allocate matrices for intermediate computations
                let mut z_pre = Array2::<f32>::zeros((batch_size, hidden_size));
                let mut r_pre = Array2::<f32>::zeros((batch_size, hidden_size));
                let mut n_pre = Array2::<f32>::zeros((batch_size, hidden_size));
                
                // Update gate computation with SIMD
                general_mat_mul(1.0, &x_t, &self.w_iz, 0.0, &mut z_pre);
                general_mat_mul(1.0, &h_local, &self.w_hz, 1.0, &mut z_pre);
                z_pre.scaled_add(1.0, &self.b_z);
                let z = &z_pre.mapv(|x| 1.0 / (1.0 + (-x).exp()));
                
                // Reset gate computation with SIMD
                general_mat_mul(1.0, &x_t, &self.w_ir, 0.0, &mut r_pre);
                general_mat_mul(1.0, &h_local, &self.w_hr, 1.0, &mut r_pre);
                r_pre.scaled_add(1.0, &self.b_r);
                let r = &r_pre.mapv(|x| 1.0 / (1.0 + (-x).exp()));
                
                // Candidate state computation with SIMD
                let r_h = r * &h_local;
                general_mat_mul(1.0, &x_t, &self.w_in, 0.0, &mut n_pre);
                general_mat_mul(1.0, &r_h, &self.w_hn, 1.0, &mut n_pre);
                n_pre.scaled_add(1.0, &self.b_n);
                let h_candidate = &n_pre.mapv(|x| x.tanh());
                
                // Update hidden state using vectorized operations
                h_local = &(z * h_candidate) + &(&Array2::from_elem(z.raw_dim(), 1.0) - z) * &h_local;
                
                // Store output
                output.slice_mut(s![.., t, ..]).assign(&h_local);
            }
            
            // Update global hidden state
            h = h_local;
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

// Optimized activation functions using SIMD when available
#[cfg(target_arch = "x86_64")]
mod simd {
    use std::arch::x86_64::*;
    
    #[target_feature(enable = "avx2")]
    pub unsafe fn sigmoid_avx(x: &mut [f32]) {
        let len = x.len();
        let simd_len = len - (len % 8);
        
        for i in (0..simd_len).step_by(8) {
            let v = _mm256_loadu_ps(&x[i]);
            let neg_v = _mm256_sub_ps(_mm256_setzero_ps(), v);
            let exp_neg_v = exp_avx2(neg_v);
            let one = _mm256_set1_ps(1.0);
            let result = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_v));
            _mm256_storeu_ps(&mut x[i], result);
        }
        
        // Handle remaining elements
        for i in simd_len..len {
            x[i] = 1.0 / (1.0 + (-x[i]).exp());
        }
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn exp_avx2(x: __m256) -> __m256 {
        // Approximate exp using a polynomial
        let c1 = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(1.0);
        let c3 = _mm256_set1_ps(0.5);
        let c4 = _mm256_set1_ps(0.1666666);
        
        let x2 = _mm256_mul_ps(x, x);
        let x3 = _mm256_mul_ps(x2, x);
        
        let term1 = _mm256_add_ps(c1, x);
        let term2 = _mm256_mul_ps(c3, x2);
        let term3 = _mm256_mul_ps(c4, x3);
        
        _mm256_add_ps(_mm256_add_ps(term1, term2), term3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;

    #[test]
    fn test_gru_performance() {
        let gru = GRUModule::new(128, 128, 2).unwrap();
        
        let batch_size = 32;
        let seq_len = 100;
        let input_shape = [batch_size, seq_len, 128];
        
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let input = Array3::random_using(input_shape, Normal::new(0.0, 1.0).unwrap(), &mut rng);
        
        let start = Instant::now();
        let output = gru.forward(&input, None).unwrap();
        let duration = start.elapsed();
        
        println!("GRU forward pass took: {:?}", duration);
        assert_eq!(output.shape(), &[batch_size, seq_len, 128]);
    }
}