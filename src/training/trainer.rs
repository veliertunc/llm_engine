use crate::model::transformer::SimpleTransformer;
use crate::training::loss::mse_loss;
use crate::training::optimizer::SGD;

pub struct Trainer<'a> {
    pub model: &'a mut SimpleTransformer,
    pub optimizer: SGD,
}

impl<'a> Trainer<'a> {
    pub fn new(model: &'a mut SimpleTransformer, optimizer: SGD) -> Self {
        Self { model, optimizer }
    }

    /// Train for one epoch over synthetic data in mini-batches
    pub fn train(&mut self, data: Vec<(Vec<f32>, Vec<f32>)>, batch_size: usize) {
        let mut total_loss = 0.0;
        let mut step = 0;

        for batch in data.chunks(batch_size) {
            let mut batch_loss = 0.0;

            for (input, target) in batch.iter() {
                let output = self.model.forward(input.clone());
                let loss = mse_loss(output.as_slice(), target.as_slice());
                batch_loss += loss;

                // 🔁 This is where a backward pass would occur
                println!(
                    "[Step {}] Sample loss: {:.4} (backward placeholder)",
                    step, loss
                );
                step += 1;
            }

            let avg_loss = batch_loss / batch.len() as f32;
            println!("== Batch complete. Avg batch loss = {:.4} ==", avg_loss);

            // 👟 Optimizer step (e.g., apply gradients)
            self.optimizer.step();

            total_loss += batch_loss;
        }

        println!(
            "Epoch complete. Avg loss = {:.4}",
            total_loss / data.len() as f32
        );
    }
}
