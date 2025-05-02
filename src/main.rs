mod model;
mod training;

use model::transformer::{ModelConfig, SimpleTransformer};
use training::optimizer::SGD;
use training::trainer::Trainer;

fn main() {
    let config = ModelConfig {
        hidden_size: 64,
        num_heads: 8,
        num_layers: 2,
        max_position_embeddings: 16,
    };

    let mut model = SimpleTransformer::new(&config);
    let optimizer = SGD::new(0.01);
    let mut trainer = Trainer::new(&mut model, optimizer);

    // Generate synthetic training data
    let sequence_len = 10;
    let num_samples = 6;
    let batch_size = 2;

    let dummy_data: Vec<(Vec<f32>, Vec<f32>)> = (0..num_samples)
        .map(|_| {
            let input = vec![0.5; config.hidden_size * sequence_len];
            let target = vec![0.8; config.hidden_size * sequence_len];
            (input, target)
        })
        .collect();

    trainer.train(dummy_data, batch_size);
}
