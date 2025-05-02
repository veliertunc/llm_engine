/// Layer Normalization layer
pub struct LayerNorm {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub epsilon: f32,
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            epsilon: 1e-5,
        }
    }

    /// Forward pass of LayerNorm
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mean = input.iter().sum::<f32>() / input.len() as f32;
        let variance = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / input.len() as f32;
        input
            .iter()
            .enumerate()
            .map(|(i, x)| {
                self.gamma[i] * ((x - mean) / (variance + self.epsilon).sqrt()) + self.beta[i]
            })
            .collect()
    }
}
