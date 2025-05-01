/// A learnable positional embedding table.
pub struct PositionalEncoding {
    pub embeddings: Vec<Vec<f32>>, // [max_position_embeddings][hidden_dim]
}

impl PositionalEncoding {
    /// Create a new PositionalEncoding with random values.
    pub fn new(max_pos: usize, hidden_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::rng();
        let embeddings = (0..max_pos)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| rng.random_range(-0.1..0.1)) // random init
                    .collect()
            })
            .collect();
        Self { embeddings }
    }

    /// Add positional encoding to input embeddings.
    /// `inputs` is shape [sequence_length][hidden_dim]
    pub fn add_encoding(&self, inputs: &mut [Vec<f32>]) {
        for (i, input_vec) in inputs.iter_mut().enumerate() {
            let pos_vec = &self.embeddings[i];
            for (j, val) in input_vec.iter_mut().enumerate() {
                *val += pos_vec[j]; // element-wise addition
            }
        }
    }
}
