/// Config for building a transformer
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub max_position_embeddings: usize,
}
