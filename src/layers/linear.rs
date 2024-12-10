use ndarray::Array2;

pub struct Linear {
    weights: Array2<f32>,
    bias: Array2<f32>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Linear {
            weights: Array2::zeros((out_features, in_features)),
            bias: Array2::zeros((out_features, 1)),
        }
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        // input shape: (batch_size, in_features)
        // weights shape: (out_features, in_features)
        // output shape: (batch_size, out_features)
        
        // 确保输入形状正确
        assert_eq!(input.shape()[1], self.weights.shape()[1], 
            "输入特征数量与权重不匹配");
            
        // 矩阵乘法并转置
        let output = input.dot(&self.weights.t());
        
        // 添加偏置
        &output + &self.bias.broadcast((output.shape()[0], output.shape()[1])).unwrap()
    }

    pub fn from_weights(weights: Vec<Vec<f32>>, bias: Vec<f32>) -> Self {
        let out_features = weights.len();
        let in_features = weights[0].len();
        
        // 将权重转置为 (in_features, out_features) 的形状
        let weights_flat: Vec<f32> = weights.into_iter()
            .flatten()
            .collect();
            
        let weights = Array2::from_shape_vec(
            (in_features, out_features),  // 修改这里：转置权重矩阵
            weights_flat
        ).unwrap()
        .reversed_axes();  // 转置矩阵
        
        let bias = Array2::from_shape_vec((out_features, 1), bias).unwrap();
        
        Linear {
            weights,
            bias,
        }
    }
} 