// A neural network that can predict the output of the xor operator

use neural_network::activations;
use neural_network::network::Network;

fn main() {
    // Data
    // let inputs = vec![vec![0., 0.], vec![1., 0.], vec![0., 1.], vec![1., 1.]];
    // let outputs = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    // let mut network = Network::<activations::Sigmoid>::new(
        // vec![2, 3, 1],
        // 0.6,
    // );

    let mut network = Network::<activations::Tanh>::load("model.json").unwrap();

    // println!("Before training");
    // println!("0, 0 = {:?}", network.feed_forward(vec![0., 0.])[0]);
    // println!("1, 0 = {:?}", network.feed_forward(vec![1., 0.])[0]);
    // println!("0, 1 = {:?}", network.feed_forward(vec![0., 1.])[0]);
    // println!("1, 1 = {:?}", network.feed_forward(vec![1., 1.])[0]);

    // train
    // network.train(inputs, outputs, 10000);

    // println!("After training");
    println!("0, 0 = {:?}", network.feed_forward(vec![0., 0.])[0]);
    println!("1, 0 = {:?}", network.feed_forward(vec![1., 0.])[0]);
    println!("0, 1 = {:?}", network.feed_forward(vec![0., 1.])[0]);
    println!("1, 1 = {:?}", network.feed_forward(vec![1., 1.])[0]);

    // println!("Saving the model");
    // network.save("model.json").unwrap();

    // Loading the model
}
