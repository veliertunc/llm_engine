// lr_scheduler.rs

/// Computes learning rate using linear warmup and linear decay
pub fn adjust_learning_rate(
    base_lr: f32,
    step: usize,
    warmup_steps: usize,
    total_steps: usize,
) -> f32 {
    if step < warmup_steps {
        base_lr * (step as f32 / warmup_steps as f32)
    } else {
        base_lr * (1.0 - (step - warmup_steps) as f32 / (total_steps - warmup_steps) as f32)
    }
}
