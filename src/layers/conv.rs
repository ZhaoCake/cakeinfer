use ndarray::Array4;

pub struct Conv2D {
    weights: Array4<f32>,
    bias: Array4<f32>,
    kernel_size: usize,
    stride: usize,
}

impl Conv2D {
    pub fn from_weights(weights: Array4<f32>, bias: Array4<f32>, stride: usize, kernel_size: usize) -> Self {
        Conv2D {
            weights,
            bias,
            kernel_size,
            stride,
        }
    }

    pub fn forward(&self, input: &Array4<f32>) -> Array4<f32> {
        let (batch_size, in_channels, in_height, in_width) = input.dim();
        let (out_channels, _, _, _) = self.weights.dim();
        
        let out_height = (in_height - self.kernel_size) / self.stride + 1;
        let out_width = (in_width - self.kernel_size) / self.stride + 1;
        
        let mut output = Array4::zeros((batch_size, out_channels, out_height, out_width));
        
        // 实现卷积运算
        for b in 0..batch_size {
            for oc in 0..out_channels {
                for h in 0..out_height {
                    for w in 0..out_width {
                        let h_start = h * self.stride;
                        let w_start = w * self.stride;
                        
                        let mut sum = self.bias[[oc, 0, 0, 0]];
                        for ic in 0..in_channels {
                            for kh in 0..self.kernel_size {
                                for kw in 0..self.kernel_size {
                                    sum += input[[b, ic, h_start + kh, w_start + kw]] 
                                        * self.weights[[oc, ic, kh, kw]];
                                }
                            }
                        }
                        // 使用与C代码相同的激活函数
                        output[[b, oc, h, w]] = 1.7159 * (0.66666667 * sum).tanh();
                    }
                }
            }
        }
        
        output
    }
}
