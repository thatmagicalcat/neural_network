use std::f64::consts::E;

use nanoserde::{SerJson, DeJson};

#[derive(Debug, Clone, Copy, SerJson, DeJson)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    Relu,
    LeakyRelu,
}

impl<'a> From<ActivationFunction> for Activation<'a> {
    fn from(value: ActivationFunction) -> Self {
        match value {
            ActivationFunction::Sigmoid => SIGMOID,
            ActivationFunction::Tanh => TANH,
            ActivationFunction::Relu => RELU,
            ActivationFunction::LeakyRelu => LEAKY_RELU,
        }
    }
}

#[derive(Clone)]
pub struct Activation<'a> {
    pub function: &'a dyn Fn(f64) -> f64,
    pub derivative: &'a dyn Fn(f64) -> f64,
}

const SIGMOID: Activation = Activation {
    function: &|x| 1.0 / (1.0 + E.powf(-x)),
    derivative: &|x| x * (1.0 - x),
};

const TANH: Activation = Activation {
    function: &|x| x.tanh(),
    derivative: &|x| 1.0 - x.tanh().powf(x),
};

const RELU: Activation = Activation {
    function: &|x| 0.0f64.max(x),
    derivative: &|x| if x > 0.0 { 1.0 } else { 0.0 },
};

pub const LEAKY_RELU: Activation = Activation {
    function: &|x| if x > 0.0 { x } else { 0.01 * x },
    derivative: &|x| if x > 0.0 { 1.0 } else { 0.01 },
};

// pub trait ActivationFunc {
//     fn function(x: f64) -> f64;
//     fn derivative(x: f64) -> f64;
// }

// pub struct Sigmoid;
// impl ActivationFunc for Sigmoid {
//     fn function(x: f64) -> f64 {
//         1.0 / (1.0 + E.powf(-x))
//     }

//     fn derivative(x: f64) -> f64 {
//         x * (1.0 - x)
//     }
// }

// pub struct Tanh;
// impl ActivationFunc for Tanh {
//     fn function(x: f64) -> f64 {
//         x.tanh()
//     }

//     fn derivative(x: f64) -> f64 {
//         1.0 - x.tanh().powf(x)
//     }
// }

// pub struct Relu;
// impl ActivationFunc for Relu {
//     fn function(x: f64) -> f64 {
//         0.0f64.max(x)
//     }

//     fn derivative(x: f64) -> f64 {
//         if x > 0.0 {
//             1.0
//         } else {
//             0.0
//         }
//     }
// }
