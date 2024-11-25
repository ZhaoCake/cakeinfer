use ndarray::{Array, Array4};

pub struct Conv {
    pub kernels: Array4<f32>,
    pub biases: Array<f32, ndarray::Dim<[usize; 1]>>, //
    pub stride: usize,
    pub padding: usize,
}

impl Conv {
    pub fn new(
        num_kernels: usize,
        kernel_size: usize,
        input_depth: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let kernels = Array::zeros((num_kernels, input_depth, kernel_size, kernel_size));
        let biases = Array::zeros(num_kernels);
        Conv {
            kernels,
            biases,
            stride,
            padding,
        }
    }

    pub fn forword(&self, _input: &Array4<f32>) -> Array4<f32> {
        let input = _input;
        let (batch_size, input_depth, input_height, input_width) = input.dim();
        let (_, num_kernels, kernel_size, _) = self.kernels.dim();
        let output_height = (input_height - kernel_size + 2 * self.padding) / self.stride + 1;
        let output_width = (input_width - kernel_size + 2 * self.padding) / self.stride + 1;
        let mut output = Array::zeros((batch_size, num_kernels, output_height, output_width));
        for b in 0..batch_size {
            for k in 0..num_kernels {
                for y in 0..output_height {
                    for x in 0..output_width {
                        let y_start = y * self.stride;
                        let y_end = y_start + kernel_size;
                        let x_start = x * self.stride;
                        let x_end = x_start + kernel_size;
                        let input_slice = input.slice(s![b, .., y_start..y_end, x_start..x_end]);
                        output[[b, k, y, x]] = input_slice
                            .iter()
                            .zip(self.kernels.slice(s![k, .., .., ..]).iter())
                            .map(|(a, b)| a * b)
                            .sum();
                    }
                }
            }
        }
        output
    }
}
