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
