/// Configuration struct for transformer model
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Dimensionality of token embeddings
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Feedforward hidden layer size
    pub ff_hidden_size: usize,
    /// Dropout rate for regularization
    pub dropout: f32,
    /// Whether to use weight sharing between input/output embeddings
    pub weight_sharing: bool,
    /// Whether to use layer normalization before residual connections (Pre-LN)
    pub pre_layer_norm: bool,
    /// Whether to tie positional encoding with learned embeddings
    pub learned_positional_encoding: bool,
    /// Whether to use quantized linear layers
    pub use_quantization: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            d_model: 128,
            n_heads: 8,
            n_layers: 4,
            max_seq_len: 128,
            vocab_size: 10000,
            ff_hidden_size: 512,
            dropout: 0.1,
            weight_sharing: false,
            pre_layer_norm: true,
            learned_positional_encoding: false,
            use_quantization: false,
            seed: Some(42),
        }
    }
}
