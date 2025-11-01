use crate::model::transformer::SimpleTransformer;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub model: SimpleTransformer,
    pub epoch: usize,
}

/// Saves a checkpoint of the model and training state to disk
pub fn save_checkpoint(model: &SimpleTransformer, epoch: usize, path: &str) {
    let checkpoint = Checkpoint {
        model: model.clone(),
        epoch,
    };
    let file = File::create(path).expect("Failed to create checkpoint file");
    bincode::serialize_into(file, &checkpoint).expect("Failed to write checkpoint");
}

/// Loads a checkpoint from disk
pub fn load_checkpoint(path: &str) -> Option<Checkpoint> {
    let mut file = File::open(path).ok()?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).ok()?;
    bincode::deserialize(&buffer).ok()
}
