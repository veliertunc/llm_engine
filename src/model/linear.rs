use super::common::Param;

/// Fully connected linear layer with weights and biases.
pub struct Linear {
    pub weight: Param,
    pub bias: Param,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weight: Param::new(in_features * out_features),
            bias: Param::new(out_features),
            in_features,
            out_features,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Simplified matrix-vector multiply (flattened weights)
        let mut output = vec![0.0; self.out_features];
        for o in 0..self.out_features {
            for i in 0..self.in_features {
                output[o] += input[i] * self.weight.value[o * self.in_features + i];
            }
            output[o] += self.bias.value[o];
        }
        output
    }

    pub fn zero_grad(&mut self) {
        self.weight.zero_grad();
        self.bias.zero_grad();
    }

    pub fn apply_grad(&mut self, lr: f32) {
        self.weight.apply_grad(lr);
        self.bias.apply_grad(lr);
    }
}
