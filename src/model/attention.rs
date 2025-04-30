use crate::model::layers::Linear;

pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub query_proj: Linear,
    pub key_proj: Linear,
    pub value_proj: Linear,
    pub output_proj: Linear,
}

impl MultiHeadAttention {
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

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let queries = self.query_proj.forward(input);
        let keys = self.key_proj.forward(input);
        let values = self.value_proj.forward(input);

        let attention_scores = Self::scaled_dot_product(&queries, &keys, self.head_dim as f32);
        let attention_probs = Self::softmax(&attention_scores);

        let mut output = vec![0.0; values.len()];

        for (i, prob) in attention_probs.iter().enumerate() {
            output[i] = prob * values[i];
        }

        self.output_proj.forward(&output)
    }

    fn scaled_dot_product(q: &[f32], k: &[f32], scale: f32) -> Vec<f32> {
        q.iter()
            .zip(k)
            .map(|(qi, ki)| (qi * ki) / scale.sqrt())
            .collect()
    }

    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp.iter().sum();
        exp.iter().map(|x| x / sum_exp).collect()
    }
}
