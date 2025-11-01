/// Computes efficient attention using dot-product sparsity (toy example)
pub fn sparse_attention(q: &[f32], k: &[f32], v: &[f32], threshold: f32) -> f32 {
    q.iter()
        .zip(k)
        .zip(v)
        .filter(|((&q_i, &k_i), _)| (q_i * k_i) > threshold)
        .map(|((_, _), &v_i)| v_i)
        .sum::<f32>()
}
