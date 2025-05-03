/// A learnable parameter with value and gradient
#[derive(Debug)]
pub struct Param {
    pub value: Vec<f32>,
    pub grad: Vec<f32>,
}

impl Param {
    pub fn new(size: usize) -> Self {
        Self {
            value: vec![0.01; size], // init small weights
            grad: vec![0.0; size],
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad.iter_mut().for_each(|g| *g = 0.0);
    }

    pub fn apply_grad(&mut self, lr: f32) {
        for (v, g) in self.value.iter_mut().zip(&self.grad) {
            *v -= lr * g;
        }
    }
}
