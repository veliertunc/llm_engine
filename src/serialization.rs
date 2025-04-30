use crate::model::SimpleTransformer;
use std::fs::File;
use std::io::{Read, Write};

pub fn save_model(model: &SimpleTransformer, path: &str) {
    let mut file = File::create(path).unwrap();
    for layer in &model.layers {
        for row in &layer.weights {
            for w in row {
                file.write_all(&w.to_le_bytes()).unwrap();
            }
        }
    }
}
