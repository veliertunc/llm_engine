use crate::model::common::Param;

/// A simple linear (fully connected) layer with learnable parameters
pub struct Linear {
    pub weight: Param,
    pub bias: Param,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    /// Creates a new linear layer with randomized weights and zero biases
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = Param::new(in_features * out_features);
        let bias = Param::new(out_features);
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Forward pass: input is a flattened vector, returns linear output
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.in_features);
        let mut output = vec![0.0; self.out_features];
        for i in 0..self.out_features {
            for j in 0..self.in_features {
                output[i] += self.weight.value[i * self.in_features + j] * input[j];
            }
            output[i] += self.bias.value[i];
        }
        output
    }

    /// Zero out all gradients in weight and bias
    pub fn zero_grad(&mut self) {
        self.weight.zero_grad();
        self.bias.zero_grad();
    }

    /// Apply gradients to update weight and bias using learning rate
    pub fn apply_grad(&mut self, lr: f32) {
        self.weight.apply_grad(lr);
        self.bias.apply_grad(lr);
    }
}
