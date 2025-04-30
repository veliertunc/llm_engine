pub mod attention;
pub mod feedforward;
pub mod layers;

use crate::config::ModelConfig;
use attention::MultiHeadAttention;
use layers::Linear;

pub struct SimpleTransformer {
    pub attention_layers: Vec<MultiHeadAttention>,
    pub ff_layers: Vec<Linear>,
}

impl SimpleTransformer {
    pub fn new(config: &ModelConfig) -> Self {
        let attention_layers = (0..config.num_layers)
            .map(|_| MultiHeadAttention::new(config.hidden_size, config.hidden_size))
            .collect();

        let ff_layers = (0..config.num_layers)
            .map(|_| Linear::new(config.hidden_size, config.hidden_size))
            .collect();

        Self {
            attention_layers,
            ff_layers,
        }
    }

    pub fn forward(&self, mut input: Vec<f32>) -> Vec<f32> {
        for (attn, ff) in self.attention_layers.iter().zip(self.ff_layers.iter()) {
            input = attn.forward(&input);
            input = ff.forward(&input); // simple FeedForward (can add LayerNorm later)
        }
        input
    }
}
