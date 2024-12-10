use ndarray::Array4;

pub struct MaxPool2D {
    kernel_size: usize,
    stride: usize,
}

impl MaxPool2D {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        MaxPool2D {
            kernel_size,
            stride,
        }
    }

    pub fn forward(&self, input: &Array4<f32>) -> Array4<f32> {
        let (batch_size, channels, height, width) = input.dim();
        
        let out_height = (height - self.kernel_size) / self.stride + 1;
        let out_width = (width - self.kernel_size) / self.stride + 1;
        
        let mut output = Array4::zeros((batch_size, channels, out_height, out_width));
        
        for b in 0..batch_size {
            for c in 0..channels {
                for h in 0..out_height {
                    for w in 0..out_width {
                        let h_start = h * self.stride;
                        let w_start = w * self.stride;
                        
                        let mut max_val = f32::MIN;
                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                max_val = max_val.max(
                                    input[[b, c, h_start + kh, w_start + kw]]
                                );
                            }
                        }
                        output[[b, c, h, w]] = max_val;
                    }
                }
            }
        }
        
        output
    }
} 