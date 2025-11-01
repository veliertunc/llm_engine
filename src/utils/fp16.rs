use half::f16;

/// Converts a slice of f32 to a vector of f16
pub fn f32_to_f16_vec(input: &[f32]) -> Vec<f16> {
    input.iter().map(|&x| f16::from_f32(x)).collect()
}

/// Converts a slice of f16 to a vector of f32
pub fn f16_to_f32_vec(input: &[f16]) -> Vec<f32> {
    input.iter().map(|&x| x.to_f32()).collect()
}
