#[derive(Debug)]
pub struct LeNetWeights {
    // 卷积层 1: 1->6 channels, 5x5 kernel
    pub conv1_weights: [[[[f32; 5]; 5]; 1]; 6],
    pub conv1_bias: [f32; 6],

    // 卷积层 2: 6->16 channels, 5x5 kernel
    pub conv2_weights: [[[[f32; 5]; 5]; 6]; 16],
    pub conv2_bias: [f32; 16],

    // 全连接层 1: 16*5*5 -> 120
    pub fc1_weights: [[f32; 400]; 120],
    pub fc1_bias: [f32; 120],

    // 全连接层 2: 120 -> 84
    pub fc2_weights: [[f32; 120]; 84],
    pub fc2_bias: [f32; 84],

    // 输出层: 84 -> 10
    pub fc3_weights: [[f32; 84]; 10],
    pub fc3_bias: [f32; 10],
}

impl Default for LeNetWeights {
    fn default() -> Self {
        // 这里提供默认权重结构，实际权重需要后续填充
        Self {
            conv1_weights: [[[[0.0; 5]; 5]; 1]; 6],
            conv1_bias: [0.0; 6],
            conv2_weights: [[[[0.0; 5]; 5]; 6]; 16],
            conv2_bias: [0.0; 16],
            fc1_weights: [[0.0; 400]; 120],
            fc1_bias: [0.0; 120],
            fc2_weights: [[0.0; 120]; 84],
            fc2_bias: [0.0; 84],
            fc3_weights: [[0.0; 84]; 10],
            fc3_bias: [0.0; 10],
        }
    }
} 