# Deep Risk Model: Theoretical Foundations

This document provides a detailed explanation of the theoretical foundations behind the Deep Risk Model, based on the research paper by [Lin et al. (2021)](https://arxiv.org/abs/2107.05201).

## 1. Problem Formulation

### 1.1 Traditional Risk Factor Models

Traditional factor models decompose asset returns \( r_t \) into systematic and idiosyncratic components:

\[ r_t = Bf_t + \epsilon_t \]

where:
- \( r_t \) is the vector of asset returns at time t
- \( B \) is the factor loading matrix
- \( f_t \) is the vector of factor returns
- \( \epsilon_t \) is the idiosyncratic return vector

### 1.2 Deep Learning Extension

Our model extends this by learning the factor structure through a deep neural network:

\[ f_t = \text{GAT}(\text{GRU}(X_t)) \]

where:
- \( X_t \) is the market data tensor
- \( \text{GRU}(\cdot) \) captures temporal dependencies
- \( \text{GAT}(\cdot) \) models cross-sectional relationships

## 2. Architecture Components

### 2.1 Gated Recurrent Unit (GRU)

The GRU processes temporal sequences with the following equations:

\[ z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z) \]
\[ r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r) \]
\[ \tilde{h}_t = \tanh(W_h x_t + U_h(r_t \odot h_{t-1}) + b_h) \]
\[ h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \]

where:
- \( z_t \) is the update gate
- \( r_t \) is the reset gate
- \( h_t \) is the hidden state
- \( \odot \) denotes element-wise multiplication

### 2.2 Graph Attention Network (GAT)

The GAT layer computes attention scores between assets:

\[ \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})} \]
\[ e_{ij} = a(Wh_i, Wh_j) \]

where:
- \( \alpha_{ij} \) is the attention coefficient
- \( h_i \) is the feature vector of asset i
- \( W \) is a learnable weight matrix
- \( a(\cdot) \) is a shared attention mechanism

## 3. Loss Function

The model is trained with a multi-component loss function:

\[ \mathcal{L} = \mathcal{L}_{\text{factor}} + \lambda_1 \mathcal{L}_{\text{ortho}} + \lambda_2 \mathcal{L}_{\text{stable}} \]

where:
- \( \mathcal{L}_{\text{factor}} \) measures factor explanatory power
- \( \mathcal{L}_{\text{ortho}} \) ensures factor orthogonality
- \( \mathcal{L}_{\text{stable}} \) promotes factor stability

### 3.1 Factor Loss

\[ \mathcal{L}_{\text{factor}} = \|r_t - Bf_t\|_2^2 \]

### 3.2 Orthogonality Loss

\[ \mathcal{L}_{\text{ortho}} = \|F^TF - I\|_F^2 \]

where \( F \) is the matrix of factor returns.

### 3.3 Stability Loss

\[ \mathcal{L}_{\text{stable}} = \|f_t - f_{t-1}\|_2^2 \]

## 4. Covariance Estimation

The covariance matrix is estimated as:

\[ \Sigma = B\Sigma_fB^T + D \]

where:
- \( \Sigma_f \) is the factor covariance matrix
- \( D \) is a diagonal matrix of idiosyncratic variances

## 5. Implementation Details

### 5.1 Hyperparameters

Our implementation uses the following default hyperparameters:
- Input size: 64 (market features)
- Hidden size: 64 (GRU state dimension)
- Number of attention heads: 4
- Head dimension: 16
- Number of GRU layers: 2
- Output size: 3 (risk factors)

### 5.2 Training Process

The model is trained using:
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Training epochs: 100
- Early stopping patience: 10

## 6. Performance Metrics

The model's performance is evaluated using:

1. Explained Variance (R²):
   \[ R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2} \]

2. Portfolio Risk Reduction:
   \[ \text{Risk Reduction} = \frac{\sigma_{\text{baseline}} - \sigma_{\text{model}}}{\sigma_{\text{baseline}}} \]

3. Factor Stability:
   \[ \text{Stability} = \text{corr}(f_t, f_{t-1}) \] 