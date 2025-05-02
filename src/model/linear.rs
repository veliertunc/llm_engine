use rand::Rng;

/// Fully connected linear layer with weights and biases.
pub struct Linear {
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::rng();
        let weights = (0..input_dim * output_dim)
            .map(|_| rng.random_range(-0.1..0.1))
            .collect();
        let biases = vec![0.0; output_dim];
        Self {
            weights,
            biases,
            input_dim,
            output_dim,
        }
    }

    /// Forward pass through the linear layer
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            for j in 0..self.input_dim {
                output[i] += input[j] * self.weights[i * self.input_dim + j];
            }
            output[i] += self.biases[i];
        }
        output
    }
}
