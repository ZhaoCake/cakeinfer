use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand_distr::Normal;

pub struct DenseLayer {
    weights: Array2<f32>,
    bias: Array1<f32>,
}

impl DenseLayer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let weights = Array2::random((input_size, output_size), Normal::new(0., 1.).unwrap());
        let bias = Array1::zeros(output_size);

        DenseLayer { weights, bias }
    }

    fn forward(&self, input: Array1<f32>) -> Array1<f32> {
        input.dot(&self.weights) + &self.bias
    }
}
