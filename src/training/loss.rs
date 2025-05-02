/// Mean Squared Error loss
pub fn mse_loss(output: &[f32], target: &[f32]) -> f32 {
    output
        .iter()
        .zip(target)
        .map(|(o, t)| (o - t).powi(2))
        .sum::<f32>()
        / output.len() as f32
}
