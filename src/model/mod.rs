pub mod attention;
pub mod feedforward;
pub mod layers;

use crate::config::ModelConfig;
use layers::Linear;

pub struct SimpleTransformer {
    pub layers: Vec<Linear>,
}

impl SimpleTransformer {
    pub fn new(config: &ModelConfig) -> Self {
        let layers = (0..config.num_layers)
            .map(|_| Linear::new(config.hidden_size, config.hidden_size))
            .collect();
        Self { layers }
    }

    pub fn forward(&self, input: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(input, |acc, layer| layer.forward(&acc))
    }
}
