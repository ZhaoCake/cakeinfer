use ndarray::Array2;

pub struct Linear {
    weights: Array2<f32>,
    bias: Array2<f32>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Linear {
            weights: Array2::zeros((out_features, in_features)),
            bias: Array2::zeros((1, out_features)),
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
        let batch_size = output.shape()[0];
        let broadcasted_bias = self.bias.broadcast((batch_size, self.bias.shape()[1]))
            .expect("无法广播偏置向量到正确的形状");
        
        &output + &broadcasted_bias
    }

    pub fn from_weights(weights: Vec<Vec<f32>>, bias: Vec<f32>) -> Self {
        let out_features = weights.len();
        let in_features = weights[0].len();
        
        // 将权重转置为 (in_features, out_features) 的形状
        let weights_flat: Vec<f32> = weights.into_iter()
            .flatten()
            .collect();
            
        let weights = Array2::from_shape_vec(
            (in_features, out_features),
            weights_flat
        ).unwrap()
        .reversed_axes();  // 转置矩阵
        
        // 修改偏置的形状为 (1, out_features)
        let bias = Array2::from_shape_vec((1, out_features), bias).unwrap();
        
        Linear {
            weights,
            bias,
        }
    }
} 