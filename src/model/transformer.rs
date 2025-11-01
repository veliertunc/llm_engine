use crate::model::layers::{
    attention::MultiHeadAttention, embedding::TokenEmbedding, linear::Linear, norm::LayerNorm,
    positional::PositionalEncoding,
};

use crate::model::consts::{HIDDEN_SIZE, NUM_LAYERS};

/// A simple transformer model with token and positional embeddings
pub struct SimpleTransformer {
    pub token_embedding: TokenEmbedding,
    pub pos_encoding: PositionalEncoding,
    pub hidden_size: usize,
    pub attention_layers: Vec<MultiHeadAttention>,
    pub attn_norms: Vec<LayerNorm>,
    pub ff_layers: Vec<Linear>,
    pub ff_norms: Vec<LayerNorm>,
}

impl SimpleTransformer {
    /// Initializes a new transformer model
    pub fn new() -> Self {
        let token_embedding = TokenEmbedding::new();
        let pos_encoding = PositionalEncoding::new();

        let mut attention_layers = Vec::new();
        let mut attn_norms = Vec::new();
        let mut ff_layers = Vec::new();
        let mut ff_norms = Vec::new();

        for _ in 0..NUM_LAYERS {
            attention_layers.push(MultiHeadAttention::new(HIDDEN_SIZE));
            attn_norms.push(LayerNorm::new(HIDDEN_SIZE));
            ff_layers.push(Linear::new(HIDDEN_SIZE, HIDDEN_SIZE));
            ff_norms.push(LayerNorm::new(HIDDEN_SIZE));
        }

        Self {
            token_embedding,
            pos_encoding,
            hidden_size: HIDDEN_SIZE,
            attention_layers,
            attn_norms,
            ff_layers,
            ff_norms,
        }
    }

    /// Forward pass from token IDs to final vector output
    pub fn forward(&self, token_ids: &[usize]) -> Vec<f32> {
        let seq_len = token_ids.len();
        let token_embeds = self.token_embedding.forward(token_ids);
        let pos_enc = self.pos_encoding.get_encoding(seq_len);

        // Add positional encoding to token embeddings
        let mut x: Vec<Vec<f32>> = token_embeds
            .iter()
            .zip(pos_enc.iter())
            .map(|(embed, pos)| embed.iter().zip(pos).map(|(a, b)| a + b).collect())
            .collect();

        // Apply each transformer layer
        for i in 0..self.attention_layers.len() {
            x = x
                .into_iter()
                .map(|vec| {
                    let attn_out = self.attention_layers[i].forward(&vec);
                    let normed = self.attn_norms[i].forward(&attn_out);
                    let ff_out = self.ff_layers[i].forward(&normed);
                    self.ff_norms[i].forward(&ff_out)
                })
                .collect();
        }

        // Mean pooling across sequence
        let mut final_vec = vec![0.0; self.hidden_size];
        for token_vec in &x {
            for i in 0..self.hidden_size {
                final_vec[i] += token_vec[i];
            }
        }
        for val in &mut final_vec {
            *val /= seq_len as f32;
        }

        final_vec
    }

    /// Clears all parameter gradients
    pub fn zero_grad(&mut self) {
        self.token_embedding.zero_grad();
        for attn in &mut self.attention_layers {
            attn.zero_grad();
        }
        for ff in &mut self.ff_layers {
            ff.zero_grad();
        }
    }

    /// Applies accumulated gradients to update parameters
    pub fn apply_grad(&mut self, lr: f32) {
        self.token_embedding.apply_grad(lr);
        for attn in &mut self.attention_layers {
            attn.apply_grad(lr);
        }
        for ff in &mut self.ff_layers {
            ff.apply_grad(lr);
        }
    }

    /// Simulates a backward pass by filling dummy gradients
    pub fn backward(&mut self, token_ids: &[usize]) {
        self.token_embedding.fill_dummy_grads(token_ids);
        for attn in &mut self.attention_layers {
            attn.fill_dummy_grads();
        }
        for ff in &mut self.ff_layers {
            ff.weight.grad.iter_mut().for_each(|g| *g = 0.001);
            ff.bias.grad.iter_mut().for_each(|g| *g = 0.001);
        }
    }
}
