use std::collections::HashMap;

pub struct Tokenizer {
    vocab: HashMap<String, usize>,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer {
    pub fn new() -> Self {
        let mut vocab = std::collections::HashMap::new();
        vocab.insert("<pad>".to_string(), 0);
        vocab.insert("<unk>".to_string(), 1);
        Self { vocab }
    }

    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|word| *self.vocab.get(word).unwrap_or(&1)) // unknown = 1
            .collect()
    }

    /// Registers new words into the vocabulary
    pub fn register_tokens(&mut self, tokens: &[&str]) {
        for token in tokens {
            let next_index = self.vocab.len();
            self.vocab.entry(token.to_string()).or_insert(next_index);
        }
    }

    /// Pads tokenized input sequences to the `max_length`
    /// Uses <pad> token ID (0) for padding
    pub fn pad_sequences(&self, sequences: &mut Vec<Vec<usize>>, max_length: usize) {
        for seq in sequences.iter_mut() {
            while seq.len() < max_length {
                seq.push(0); // Add <pad> token (ID 0) for padding
            }
        }
    }
}
