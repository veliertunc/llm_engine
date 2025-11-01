use packed_simd_2::f32x8;

/// Computes the dot product using SIMD acceleration
pub fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = f32x8::splat(0.0);
    for i in (0..a.len()).step_by(8) {
        let va = f32x8::from_slice_unaligned(&a[i..]);
        let vb = f32x8::from_slice_unaligned(&b[i..]);
        sum += va * vb;
    }
    sum.reduce_sum()
}
