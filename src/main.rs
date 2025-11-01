mod inference;
mod model;
mod training;
mod utils;

use crate::api::inference::run_inference_server;

fn main() {
    println!("Starting inference server...");
    run_inference_server().unwrap();
}
