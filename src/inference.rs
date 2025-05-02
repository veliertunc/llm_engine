use crate::{model::transformer::SimpleTransformer, tokenizer::Tokenizer};

pub struct InferenceEngine {
    model: SimpleTransformer,
    tokenizer: Tokenizer,
}

impl InferenceEngine {
    pub fn new(model: SimpleTransformer, tokenizer: Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    pub fn generate(&self, prompt: &str) -> Vec<f32> {
        let tokens = self.tokenizer.tokenize(prompt);
        let input = tokens.iter().map(|&t| t as f32).collect::<Vec<_>>();
        self.model.forward(input)
    }
}
