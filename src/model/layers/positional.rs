use crate::model::consts::{HIDDEN_SIZE, MAX_SEQ_LEN};

/// PositionalEncoding generates sinusoidal position encodings for sequences
pub struct PositionalEncoding {
    pub encoding: Vec<Vec<f32>>, // shape: [MAX_SEQ_LEN][HIDDEN_SIZE]
}

impl PositionalEncoding {
    pub fn new() -> Self {
        let mut encoding = vec![vec![0.0; HIDDEN_SIZE]; MAX_SEQ_LEN];

        for pos in 0..MAX_SEQ_LEN {
            for i in 0..HIDDEN_SIZE {
                let angle =
                    pos as f32 / f32::powf(10000.0, (2 * (i / 2)) as f32 / HIDDEN_SIZE as f32);
                encoding[pos][i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        Self { encoding }
    }

    /// Fetch positional encoding for a given sequence length
    pub fn get_encoding(&self, seq_len: usize) -> &[Vec<f32>] {
        &self.encoding[0..seq_len]
    }
}
