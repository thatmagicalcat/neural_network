use std::fs;
use std::io::{stdout, Write};

use crate::layers::Layer;
use crate::matrix::Matrix;

use nanoserde::{DeJson, SerJson};

#[derive(SerJson, DeJson)]
pub struct Network {
    /// Size of each layers
    layers: Vec<Layer>,

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
}

impl Network {
    pub fn new(layers: Vec<Layer>, learning_rate: f64) -> Self {
        let layers: Vec<Layer> = layers.into_iter().map(|i| i.into()).collect();
        let mut weights = Vec::with_capacity(layers.len() - 1);
        let mut biases = Vec::with_capacity(layers.len() - 1);

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1].size, layers[i].size));
            biases.push(Matrix::random(layers[i + 1].size, 1));
        }

        Self {
            layers,
            weights,
            biases,
            learning_rate,
            data: vec![],
        }
    }

    pub fn load(file_path: &str) -> std::io::Result<Self> {
        let contents = fs::read_to_string(file_path)?;

        let Ok(data) = Self::deserialize_json(&contents) else {
            panic!("Failed to parse data");
        };

        Ok(data)
    }

    pub fn save(&self, file_path: &str) -> std::io::Result<()> {
        let mut file = fs::File::create(file_path)?;

        file.write_all(self.serialize_json().as_bytes())?;

        Ok(())
    }

    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.layers[0].size {
            panic!("Invalid number of inputs");
        }

        let mut current = Matrix::row(inputs).transpose();
        self.data = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            let activation = self.layers[i + 1]
                .get_activation_function()
                .unwrap_or_else(|| {
                    panic!("No activation function provided");
                });

            current = self.weights[i]
                .mul(&current)
                .add(&self.biases[i])
                .map(activation.function);

            self.data.push(current.clone());
        }

        current.transpose()[0].to_owned()
    }

    pub fn back_propogate(&mut self, outputs: Vec<f64>, targets: Vec<f64>) {
        if targets.len() != self.layers.last().unwrap().size {
            panic!("Invalid number of targets");
        }

        let parsed = Matrix::row(outputs);
        let mut errors = Matrix::row(targets).sub(&parsed);

        let activation = self
            .layers
            .last()
            .unwrap()
            .get_activation_function()
            .unwrap();
        let mut gradients = parsed.map(activation.derivative);

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.dot(&errors).map(|x| x * self.learning_rate);

            self.weights[i] = self.weights[i].add(&gradients.mul(&self.data[i].transpose()));
            self.biases[i] = self.biases[i].add(&gradients);

            errors = self.weights[i].transpose().mul(&errors);

            let activation = self.layers[i + 1]
                .get_activation_function()
                .unwrap_or_else(|| {
                    panic!("No activation function provided");
                });

            gradients = self.data[i].map(activation.derivative);
        }
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u16) {
        for i in 1..=epochs {
            if epochs < 100 || i % (epochs / 100) == 0 {
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
