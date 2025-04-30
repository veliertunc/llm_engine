pub struct Tokenizer {
    vocab: std::collections::HashMap<String, usize>,
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
}
