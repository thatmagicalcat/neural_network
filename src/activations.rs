use std::f64::consts::E;

pub trait ActivationFunc {
    fn function(x: f64) -> f64;
    fn derivative(x: f64) -> f64;
}

pub struct SIGMOID;
impl ActivationFunc for SIGMOID {
    fn function(x: f64) -> f64 {
        1.0 / (1.0 + E.powf(-x))
    }

    fn derivative(x: f64) -> f64 {
        x * (1.0 - x)
    }
}

struct TANH;
impl ActivationFunc for TANH {
    fn function(x: f64) -> f64 {
        x.tanh()
    }

    fn derivative(x: f64) -> f64 {
        1.0 - x.tanh().powf(x)
    }
}

struct RELU;
impl ActivationFunc for RELU {
    fn function(x: f64) -> f64 {
        0.0f64.max(x)
    }

    fn derivative(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}
