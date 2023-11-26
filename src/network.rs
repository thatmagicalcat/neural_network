use crate::activations::Activation;
use crate::matrix::Matrix;

pub struct Network<'a> {
    /// Size of each layers
    layers: Vec<usize>,

    /// Vector of matrices representing the
    /// weights of each connection
    weights: Vec<Matrix>,

    /// Vector of matrices representing the
    /// biases of neurons
    biases: Vec<Matrix>,

    /// Vector of matrices representing the
    /// actual value of each neuron
    data: Vec<Matrix>,

    /// Learning rate of this nerual network
    learning_rate: f64,

    /// Activation function for this nerual network
    activation: Activation<'a>,
}

impl<'a> Network<'a> {
    pub fn new(layers: Vec<usize>, learning_rate: f64, activation: Activation<'a>) -> Self {
        let mut weights = Vec::with_capacity(layers.len() - 1);
        let mut biases = Vec::with_capacity(layers.len() - 1);

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }

        Self {
            layers,
            weights,
            biases,
            learning_rate,
            activation,
            data: vec![],
        }
    }

    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.layers[0] {
            panic!("Invalid number of inputs");
        }

        let mut current = Matrix::from(vec![inputs]).transpose();
        self.data = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .mul(&current)
                .add(&self.biases[i])
                .map(self.activation.function);

            self.data.push(current.clone());
        }

        current.transpose()[0].to_owned()
    }

    pub fn back_propogate(&mut self, outputs: Vec<f64>, targets: Vec<f64>) {
        if targets.len() != *self.layers.last().unwrap() {
            panic!("Invalid number of targets");
        }

        let parsed = Matrix::from(vec![outputs]);
        let mut errors = Matrix::from(vec![targets]).sub(&parsed);
        let mut gradients = parsed.map(self.activation.derivative);

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.dot(&errors).map(&|x| x * self.learning_rate);

            self.weights[i] = self.weights[i].add(&gradients.mul(&self.data[i].transpose()));
            self.biases[i] = self.biases[i].add(&gradients);

            errors = self.weights[i].transpose().mul(&errors);
            gradients = self.data[i].map(self.activation.derivative);
        }
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u16) {
        for i in 1..=epochs {
            if epochs < 100 || i % (epochs / 100) == 0 {
                use std::io::{Write, stdout};
                print!("\r[Log] Epoch {i} of {epochs}");
                stdout().flush().unwrap();
            }

            for j in 0..inputs.len() {
                let outputs = self.feed_forward(inputs[j].clone());
                self.back_propogate(outputs, targets[j].clone());
            }
        }

        println!("\r[Log] Done training!                  ");
    }
}
