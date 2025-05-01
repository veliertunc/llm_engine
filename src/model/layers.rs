use rand::Rng;

pub struct Linear {
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // let mut rng = rand::thread_rng();
        let mut rng = rand::rng();
        let weights = (0..out_features)
            .map(|_| {
                (0..in_features)
                    .map(|_| rng.random_range(-0.1..0.1))
                    .collect()
            })
            .collect();
        let bias = vec![0.0; out_features];
        Self { weights, bias }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.weights
            .iter()
            .map(|w_row| w_row.iter().zip(input).map(|(w, i)| w * i).sum::<f32>())
            .zip(self.bias.iter())
            .map(|(sum, b)| sum + b)
            .collect()
    }
}

pub struct LayerNorm {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub epsilon: f32,
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            epsilon: 1e-5,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mean: f32 = input.iter().copied().sum::<f32>() / input.len() as f32;
        let var: f32 = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / input.len() as f32;
        let std = (var + self.epsilon).sqrt();

        input
            .iter()
            .enumerate()
            .map(|(i, &x)| self.gamma[i] * ((x - mean) / std) + self.beta[i])
            .collect()
    }
}
