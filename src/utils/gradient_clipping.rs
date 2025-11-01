/// Clips gradients based on global norm to stabilize training
pub fn clip_gradients(grads: &mut [f32], clip_value: f32) {
    let total_norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
    if total_norm > clip_value {
        let scale = clip_value / total_norm;
        for g in grads.iter_mut() {
            *g *= scale;
        }
    }
}
