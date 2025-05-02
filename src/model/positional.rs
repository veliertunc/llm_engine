/// Precomputed sinusoidal positional encoding
pub struct PositionalEncoding {
    pub embeddings: Vec<Vec<f32>>, // shape: [max_position][hidden_dim]
}

impl PositionalEncoding {
    pub fn new(max_position: usize, hidden_dim: usize) -> Self {
        let mut embeddings = vec![vec![0.0; hidden_dim]; max_position];
        for pos in 0..max_position {
            for i in 0..hidden_dim {
                let angle =
                    (pos as f32) / (10000.0_f32).powf((2 * (i / 2)) as f32 / hidden_dim as f32);
                embeddings[pos][i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }
        Self { embeddings }
    }

    /// Add position encoding to input matrix in-place
    pub fn add_encoding(&self, input: &mut [Vec<f32>]) {
        for (i, row) in input.iter_mut().enumerate() {
            for j in 0..row.len() {
                row[j] += self.embeddings[i][j];
            }
        }
    }
}
