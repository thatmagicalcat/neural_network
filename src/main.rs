// A neural network that can predict the output of the xor operator

use neural_network::{activations::ActivationFunction, network::Network};

fn main() {
    // // Data
    // let inputs = vec![vec![0., 0.], vec![1., 0.], vec![0., 1.], vec![1., 1.]];
    // let outputs = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    // let mut network = Network::new(
    //     vec![
    //         2.into(),                                // two input neurons
    //         (3, ActivationFunction::Sigmoid).into(), // 3 neurons in hidden layer
    //         (1, ActivationFunction::Sigmoid).into(), // output layer
    //     ],
    //     0.5,
    // );

    let mut network = Network::load("model.json").unwrap();

    println!("Before training");
    println!("0, 0 = {:?}", network.feed_forward(vec![0., 0.])[0]);
    println!("1, 0 = {:?}", network.feed_forward(vec![1., 0.])[0]);
    println!("0, 1 = {:?}", network.feed_forward(vec![0., 1.])[0]);
    println!("1, 1 = {:?}", network.feed_forward(vec![1., 1.])[0]);

    // train
    // network.train(inputs, outputs, 10000);

    println!("After training");
    println!("0, 0 = {:?}", network.feed_forward(vec![0., 0.])[0]);
    println!("1, 0 = {:?}", network.feed_forward(vec![1., 0.])[0]);
    println!("0, 1 = {:?}", network.feed_forward(vec![0., 1.])[0]);
    println!("1, 1 = {:?}", network.feed_forward(vec![1., 1.])[0]);

    println!("Saving the model");
    network.save("model.json").unwrap();

    // Loading the model
}
