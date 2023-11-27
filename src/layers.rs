use nanoserde::{SerJson, DeJson};

use crate::activations::{Activation, ActivationFunction};

#[derive(Debug, Clone, Copy, SerJson, DeJson)]
pub struct Layer {
    pub size: usize,
    activation: Option<ActivationFunction>,
}

impl Layer {
    pub fn new(size: usize, activation: Option<ActivationFunction>) -> Self {
        Self { size, activation }
    }

    pub fn get_activation_function(&self) -> Option<Activation> {
        self.activation.map(|i| i.into())
    }
}

impl Into<Layer> for (usize, ActivationFunction) {
    fn into(self) -> Layer {
        Layer::new(self.0, Some(self.1))
    }
}

impl Into<Layer> for usize {
    fn into(self) -> Layer {
        Layer::new(self, None)
    }
}
