use ndarray::{Array2, Array4};
use image::GrayImage;

pub fn predict(img: &GrayImage) -> (usize, Vec<f32>) {
    // 1. 图像预处理
    let processed = preprocess_image(img);
    
    // 2. 创建网络并加载权重
    let model = crate::LeNet5::new();
    
    // 3. 执行推理
    let output = model.forward(processed);
    
    // 4. 获取预测结果
    get_prediction(&output)
}

fn preprocess_image(img: &GrayImage) -> Array4<f32> {
    // 创建一个 29x29 的数组（不是32x32）
    let mut processed = Array4::zeros((1, 1, 29, 29));
    
    // 复制并归一化像素值
    for y in 0..29 {
        for x in 0..29 {
            let pixel = img.get_pixel(x as u32, y as u32);
            processed[[0, 0, y, x]] = pixel[0] as f32 / 255.0;
        }
    }
    
    processed
}

fn get_prediction(output: &Array2<f32>) -> (usize, Vec<f32>) {
    let mut max_idx = 0;
    let mut max_prob = output[[0, 0]];
    
    for (i, &prob) in output.row(0).iter().enumerate() {
        if prob > max_prob {
            max_prob = prob;
            max_idx = i;
        }
    }
    
    (max_idx, output.row(0).to_vec())
} 