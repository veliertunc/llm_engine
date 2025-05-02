use crate::model::layers::Linear;

/// Multi-head self-attention layer.
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub query_proj: Linear,
    pub key_proj: Linear,
    pub value_proj: Linear,
    pub output_proj: Linear,
}

impl MultiHeadAttention {
    /// Initialize a new multi-head attention layer.
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        let head_dim = embed_dim / num_heads;
        Self {
            num_heads,
            head_dim,
            query_proj: Linear::new(embed_dim, embed_dim),
            key_proj: Linear::new(embed_dim, embed_dim),
            value_proj: Linear::new(embed_dim, embed_dim),
            output_proj: Linear::new(embed_dim, embed_dim),
        }
    }

    /// Forward pass with optional causal mask.
    ///
    /// Input is a flattened vector of shape [seq_len * hidden_size].
    pub fn forward(&self, input: &[f32], seq_len: usize, use_mask: bool) -> Vec<f32> {
        let q = self.query_proj.forward(input);
        let k = self.key_proj.forward(input);
        let v = self.value_proj.forward(input);

        let hidden_size = self.num_heads * self.head_dim;
        let q_mat = q
            .chunks(hidden_size)
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();
        let k_mat = k
            .chunks(hidden_size)
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();
        let v_mat = v
            .chunks(hidden_size)
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();

        // Compute scaled dot-product attention scores
        let mut attention_output = vec![vec![0.0; hidden_size]; seq_len];
        for i in 0..seq_len {
            let mut scores = vec![0.0; seq_len];
            for j in 0..seq_len {
                if use_mask && j > i {
                    scores[j] = f32::NEG_INFINITY; // mask future
                } else {
                    scores[j] = Self::dot(&q_mat[i], &k_mat[j]) / (self.head_dim as f32).sqrt();
                }
            }

            let weights = Self::softmax(&scores);
            for j in 0..seq_len {
                for d in 0..hidden_size {
                    attention_output[i][d] += weights[j] * v_mat[j][d];
                }
            }
        }

        // Flatten and apply output projection
        let output_flat = attention_output.into_iter().flatten().collect::<Vec<_>>();
        self.output_proj.forward(&output_flat)
    }

    /// Compute dot product between two vectors.
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Numerically stable softmax over a 1D vector.
    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|x| x / sum).collect()
    }
}
