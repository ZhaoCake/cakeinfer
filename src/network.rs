use crate::layers::{
    conv::Conv2D,
    linear::Linear,
};
use ndarray::{Array2, Array4};
use crate::data::load_weights_from_file;

pub struct LeNet5 {
    layer1: Conv2D,    // 6个卷积核 [6][1][5][5]
    layer2: Conv2D,    // 50个卷积核 [50][6][5][5]
    layer3: Linear,    // 全连接层
    layer4: Linear,    // 输出层
}

impl LeNet5 {
    pub fn new() -> Self {
        let weights = load_weights_from_file("resources/lenet_param.txt");
        
        // Layer1: 1->6 卷积层
        let layer1_weights = Array4::from_shape_vec(
            (6, 1, 5, 5),
            weights.layer1.weights.into_iter()
                .flatten()
                .flatten()
                .flatten()
                .collect()
        ).unwrap();
        let layer1_bias = Array4::from_shape_vec((6, 1, 1, 1), weights.layer1.bias).unwrap();

        // Layer2: 6->50 卷积层
        let layer2_weights = Array4::from_shape_vec(
            (50, 6, 5, 5),
            weights.layer2.weights.into_iter()
                .flatten()
                .flatten()
                .flatten()
                .collect()
        ).unwrap();
        let layer2_bias = Array4::from_shape_vec((50, 1, 1, 1), weights.layer2.bias).unwrap();

        // Layer3: 400->120 全连接层
        let layer3 = Linear::from_weights(
            weights.layer3.weights,
            weights.layer3.bias
        );

        // Layer4: 120->10 全连接层
        let layer4 = Linear::from_weights(
            weights.layer4.weights,
            weights.layer4.bias
        );

        LeNet5 {
            layer1: Conv2D::from_weights(
                layer1_weights,  // shape should be [6, 1, 5, 5]
                layer1_bias,    // shape should be [6]
                2,              // stride = 2
                5,              // kernel_size = 5
            ),
            layer2: Conv2D::from_weights(
                layer2_weights,  // 使用之前已经正确转换的 layer2_weights
                layer2_bias,     // 使用之前已经正确转换的 layer2_bias
                2,              // stride = 2
                5,              // kernel_size = 5
            ),
            layer3,
            layer4,
        }
    }

    pub fn forward(&self, input: Array4<f32>) -> Array2<f32> {
        // 检查输入形状
        assert_eq!(input.shape(), &[1, 1, 29, 29], "输入形状必须是 (1,1,29,29)");
        
        // 第一个卷积层 (6个特征图)
        let x = self.layer1.forward(&input);
        assert_eq!(x.shape(), &[1, 6, 13, 13]);
        
        // 第二个卷积层 (50个特征图)
        let x = self.layer2.forward(&x);
        assert_eq!(x.shape(), &[1, 50, 5, 5]);
        
        // 展平操作
        let batch_size = x.shape()[0];
        let x_flat = x.into_shape((batch_size, 50 * 5 * 5))
            .expect("展平操作失败");
        
        // 全连接层 (100个神经元)
        let x = self.layer3.forward(&x_flat);
        assert_eq!(x.shape(), &[1, 100]);
        
        // 输出层 (10个神经元)
        let x = self.layer4.forward(&x);
        assert_eq!(x.shape(), &[1, 10]);
        
        x
    }
}

