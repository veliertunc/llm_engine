use crate::model::layers::linear::Linear;

/// A simplified multi-head attention block composed of projection layers
pub struct MultiHeadAttention {
    pub query_proj: Linear,
    pub key_proj: Linear,
    pub value_proj: Linear,
    pub out_proj: Linear,
}

impl MultiHeadAttention {
    /// Initialize a multi-head attention block
    pub fn new(hidden_size: usize) -> Self {
        Self {
            query_proj: Linear::new(hidden_size, hidden_size),
            key_proj: Linear::new(hidden_size, hidden_size),
            value_proj: Linear::new(hidden_size, hidden_size),
            out_proj: Linear::new(hidden_size, hidden_size),
        }
    }

    /// Forward pass stub: apply Q/K/V and output projection to token embedding
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let q = self.query_proj.forward(input);
        let k = self.key_proj.forward(input);
        let v = self.value_proj.forward(input);
        let mut combined = vec![0.0; q.len()];
        for i in 0..q.len() {
            combined[i] = q[i] + k[i] + v[i]; // simplified operation
        }
        self.out_proj.forward(&combined)
    }

    /// Forward pass stub: apply Q/K/V and output projection to token embedding
    /// Zero gradients in all projection layers
    pub fn zero_grad(&mut self) {
        self.query_proj.zero_grad();
        self.key_proj.zero_grad();
        self.value_proj.zero_grad();
        self.out_proj.zero_grad();
    }

    /// Apply parameter gradients using learning rate
    pub fn apply_grad(&mut self, lr: f32) {
        self.query_proj.apply_grad(lr);
        self.key_proj.apply_grad(lr);
        self.value_proj.apply_grad(lr);
        self.out_proj.apply_grad(lr);
    }

    /// Fill all gradients with dummy value for simulation
    pub fn fill_dummy_grads(&mut self) {
        for proj in [
            &mut self.query_proj,
            &mut self.key_proj,
            &mut self.value_proj,
            &mut self.out_proj,
        ] {
            proj.weight.grad.iter_mut().for_each(|g| *g = 0.001);
            proj.bias.grad.iter_mut().for_each(|g| *g = 0.001);
        }
    }
}
