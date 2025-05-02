use crate::model::{
    attention::MultiHeadAttention, layernorm::LayerNorm, linear::Linear,
    positional::PositionalEncoding,
};

use rayon::prelude::*;
/// Config for building a transformer
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub max_position_embeddings: usize,
}

/// Simplified Transformer model
pub struct SimpleTransformer {
    pub hidden_size: usize,
    pub pos_encoding: PositionalEncoding,
    pub attention_layers: Vec<MultiHeadAttention>,
    pub attn_norms: Vec<LayerNorm>,
    pub ff_layers: Vec<Linear>,
    pub ff_norms: Vec<LayerNorm>,
}

impl SimpleTransformer {
    pub fn new(config: &ModelConfig) -> Self {
        let pos_encoding =
            PositionalEncoding::new(config.max_position_embeddings, config.hidden_size);

        let attention_layers = (0..config.num_layers)
            .map(|_| MultiHeadAttention::new(config.hidden_size, config.num_heads))
            .collect();

        let attn_norms = (0..config.num_layers)
            .map(|_| LayerNorm::new(config.hidden_size))
            .collect();

        let ff_layers = (0..config.num_layers)
            .map(|_| Linear::new(config.hidden_size, config.hidden_size))
            .collect();

        let ff_norms = (0..config.num_layers)
            .map(|_| LayerNorm::new(config.hidden_size))
            .collect();

        Self {
            hidden_size: config.hidden_size,
            pos_encoding,
            attention_layers,
            attn_norms,
            ff_layers,
            ff_norms,
        }
    }

    /// Forward pass for a single sequence
    pub fn forward(&self, input: Vec<f32>) -> Vec<f32> {
        let seq_len = input.len() / self.hidden_size;
        let mut embedded_input = input
            .chunks(self.hidden_size)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();

        self.pos_encoding.add_encoding(&mut embedded_input);
        let mut input = embedded_input.into_iter().flatten().collect::<Vec<_>>();

        for ((attn, attn_norm), (ff, ff_norm)) in self
            .attention_layers
            .iter()
            .zip(self.attn_norms.iter())
            .zip(self.ff_layers.iter().zip(self.ff_norms.iter()))
        {
            let attn_out = attn.forward(&input, seq_len, true);
            let attn_out = attn_out
                .iter()
                .zip(&input)
                .map(|(a, i)| a + i)
                .collect::<Vec<_>>();
            let attn_normed = attn_norm.forward(&attn_out);

            let ff_out = ff.forward(&attn_normed);
            let ff_out = ff_out
                .iter()
                .zip(&attn_normed)
                .map(|(f, i)| f + i)
                .collect::<Vec<_>>();
            input = ff_norm.forward(&ff_out);
        }

        input
    }

    /// Forward pass for a batch of sequences
    pub fn forward_batch(&self, batch: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        batch.into_par_iter().map(|x| self.forward(x)).collect()
    }

    /// Placeholder for backward pass. In real impl, this computes gradients.
    pub fn backward(&mut self) {
        println!("[BACKWARD] Simulating gradient computation (not implemented yet)");
    }
}
