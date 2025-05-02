mod model;

use model::transformer::{ModelConfig, SimpleTransformer};

fn main() {
    // Set up model configuration
    let config = ModelConfig {
        hidden_size: 64,             // dimension of each token
        num_heads: 8,                // number of attention heads
        num_layers: 2,               // number of transformer blocks
        max_position_embeddings: 16, // max sequence length
    };

    // Create model
    let model = SimpleTransformer::new(&config);

    // Simulate a single input sequence: 10 tokens, each with `hidden_size` dimensions
    let seq_len = 10;
    let single_input = vec![0.5; config.hidden_size * seq_len];

    println!("== Single Sequence Inference ==");
    let output = model.forward(single_input.clone());
    println!("Output (first 8 values): {:?}", &output[..8]);

    // Batched inference with two sequences
    let batch_input = vec![
        single_input.clone(),                    // sequence 1
        vec![0.1; config.hidden_size * seq_len], // sequence 2
    ];

    println!("\n== Batched Inference ==");
    let batch_output = model.forward_batch(batch_input);
    for (i, out) in batch_output.iter().enumerate() {
        println!("Sample {}: {:?}", i, &out[..8]);
    }
}
