pub mod attention;
pub mod feedforward;
pub mod layers;
pub mod positional;

use crate::config::ModelConfig;
use attention::MultiHeadAttention;
use layers::{LayerNorm, Linear};
use positional::PositionalEncoding;

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
        let hidden_size = config.hidden_size;

        let attention_layers = (0..config.num_layers)
            .map(|_| MultiHeadAttention::new(config.hidden_size, config.num_heads))
            .collect();

        let ff_layers = (0..config.num_layers)
            .map(|_| Linear::new(config.hidden_size, config.hidden_size))
            .collect();

        let attn_norms = (0..config.num_layers)
            .map(|_| LayerNorm::new(config.hidden_size))
            .collect();

        let ff_norms = (0..config.num_layers)
            .map(|_| LayerNorm::new(config.hidden_size))
            .collect();

        let pos_encoding = PositionalEncoding::new(config.max_position_embeddings, config.hidden_size), 

        Self {
            hidden_size,
            pos_encoding,
            attention_layers,
            attn_norms,
            ff_layers,
            ff_norms,
        }
    }


    /// Forward pass through the full Transformer.
    /// Input is a flattened vector of shape [seq_len * hidden_size].
    pub fn forward(&self, input: Vec<f32>) -> Vec<f32> {
        // let seq_len = input.len() / self.hidden_size;

        // Reshape input into [seq_len][hidden_size]
        let mut embedded_input = input
            .chunks(self.hidden_size)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();

        // Add positional embeddings to token vectors
        self.pos_encoding.add_encoding(&mut embedded_input);

        // Flatten it back for processing
        let mut input = embedded_input.into_iter().flatten().collect::<Vec<_>>();

        for ((attn, attn_norm), (ff, ff_norm)) in 
            self.attention_layers.iter().zip(self.attn_norms.iter())
            .zip(self.ff_layers.iter().zip(self.ff_norms.iter())) {

            // Attention block with residual + norm
            let attn_out = attn.forward(&input);
            let attn_out = attn_out.iter().zip(&input).map(|(a, i)| a + i).collect::<Vec<_>>();
            let attn_normed = attn_norm.forward(&attn_out);

            // Feedforward block with residual + norm
            let ff_out = ff.forward(&attn_normed);
            let ff_out = ff_out.iter().zip(&attn_normed).map(|(f, i)| f + i).collect::<Vec<_>>();
            input = ff_norm.forward(&ff_out);
        }

        input
    }
}
