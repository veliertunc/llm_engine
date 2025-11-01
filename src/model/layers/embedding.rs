use crate::model::{
    common::Param,
    consts::{HIDDEN_SIZE, VOCAB_SIZE},
};
/// Token embedding layer using a learnable lookup table
pub struct TokenEmbedding {
    pub embeddings: Vec<Param>, // One Param per token ID
}

impl TokenEmbedding {
    pub fn new() -> Self {
        let embeddings = (0..VOCAB_SIZE).map(|_| Param::new(HIDDEN_SIZE)).collect();
        Self { embeddings }
    }

    pub fn forward(&self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        token_ids
            .iter()
            .map(|&id| self.embeddings[id].value.clone())
            .collect()
    }

    pub fn zero_grad(&mut self) {
        for param in &mut self.embeddings {
            param.zero_grad();
        }
    }

    pub fn apply_grad(&mut self, lr: f32) {
        for param in &mut self.embeddings {
            param.apply_grad(lr);
        }
    }

    pub fn fill_dummy_grads(&mut self, token_ids: &[usize]) {
        for &id in token_ids {
            self.embeddings[id].grad.iter_mut().for_each(|g| *g = 0.001);
        }
    }
}
