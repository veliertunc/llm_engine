use crate::model::transformer::SimpleTransformer;
use crate::tokenizer::Tokenizer;
use crate::training::consts::{EPOCHS, LEARNING_RATE};

use crate::utils::checkpointing::{load_checkpoint, save_checkpoint};
use crate::utils::distributed::aggregate_gradients_distributed;
use crate::utils::gradient_clipping::clip_gradients;
use crate::utils::lr_scheduler::adjust_learning_rate;

pub struct Trainer {
    pub lr: f32,
    pub clip_value: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub num_workers: usize,
}

impl Trainer {
    /// Trains the transformer model using standard gradient descent with scheduling, clipping, and checkpointing
    pub fn train(
        &mut self,
        model: &mut SimpleTransformer,
        data: &[Vec<usize>],
        targets: &[Vec<usize>],
    ) {
        let mut step = 0;

        for epoch in 0..10 {
            println!("Epoch {epoch}");

            for (input, target) in data.iter().zip(targets.iter()) {
                step += 1;

                // Adjust learning rate with warmup/decay schedule
                let lr = adjust_learning_rate(self.lr, step, self.warmup_steps, self.total_steps);

                // Forward pass and compute loss
                let _output = model.forward(input); // Replace with actual loss computation
                let loss = model.mse_loss(target); // Implement mse_loss in model

                // Backward pass (returns gradients of all parameters as flat Vec<f32>)
                let mut grads = model.backward();

                // Clip gradients by global norm
                clip_gradients(&mut grads, self.clip_value);

                // Aggregate gradients across workers (simulated)
                aggregate_gradients_distributed(&mut grads, self.num_workers);

                // Update parameters
                model.apply_grads(&grads, lr);

                // Save periodic checkpoints
                if step % 100 == 0 {
                    save_checkpoint(model, epoch, &self.checkpoint_path);
                    println!("Checkpoint saved at step {step}");
                }

                if step % 10 == 0 {
                    println!("Step {step}, Loss: {loss:.5}");
                }
            }
        }
    }

    /// Resumes training from checkpoint
    pub fn resume_training(&mut self) -> Option<(SimpleTransformer, usize)> {
        if let Some(checkpoint) = load_checkpoint(&self.checkpoint_path) {
            println!("Resumed from checkpoint: epoch {}", checkpoint.epoch);
            Some((checkpoint.model, checkpoint.epoch))
        } else {
            println!("No checkpoint found, starting fresh");
            None
        }
    }
}
