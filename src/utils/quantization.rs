/// Quantizes a slice of f32 values to i8 using a scale factor
pub fn quantize_to_i8(values: &[f32], scale: f32) -> Vec<i8> {
    values
        .iter()
        .map(|&x| (x * scale).round().clamp(-128.0, 127.0) as i8)
        .collect()
}

/// Dequantizes a slice of i8 values to f32 using a scale factor
pub fn dequantize_from_i8(values: &[i8], scale: f32) -> Vec<f32> {
    values.iter().map(|&x| x as f32 / scale).collect()
}
