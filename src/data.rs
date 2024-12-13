use std::fs::File;
use std::io::{self, BufRead, Read};

pub fn load_weights_from_file(file_path: &str) -> WeightParams {
    let mut file = File::open(file_path).expect("无法打开文件");
    let mut all_weights = String::new();
    file.read_to_string(&mut all_weights).expect("读取文件失败");
    
    let weights: Vec<f32> = all_weights
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    
    let layer1_start = 0;
    let layer1_end = 156;  // 6 * (25 + 1)
    let layer2_start = layer1_end;
    let layer2_end = layer2_start + 7550;  // 50 * (150 + 1)
    let layer3_start = layer2_end;
    let layer3_end = layer3_start + 100 * 1251;  // 100 * (1250 + 1)
    let layer4_start = layer3_end;
    let layer4_end = layer4_start + 10 * 101;  // 10 * (100 + 1)

    let layer1_weights = weights[layer1_start..layer1_end].to_vec();
    let layer2_weights = weights[layer2_start..layer2_end].to_vec();
    let layer3_weights = weights[layer3_start..layer3_end].to_vec();
    let layer4_weights = weights[layer4_start..layer4_end].to_vec();

    WeightParams {
        layer1: convert_to_layer1(&layer1_weights),
        layer2: convert_to_layer2(&layer2_weights),
        layer3: convert_to_layer3(&layer3_weights),
        layer4: convert_to_layer4(&layer4_weights),
    }
}

#[derive(Debug)]
pub struct WeightParams {
    pub layer1: Layer1Weights,
    pub layer2: Layer2Weights,
    pub layer3: Layer3Weights,
    pub layer4: Layer4Weights,
}

#[derive(Debug)]
pub struct Layer1Weights {
    pub weights: Vec<Vec<Vec<Vec<f32>>>>,  // [6][1][5][5]
    pub bias: Vec<f32>,                    // [6]
}

#[derive(Debug)]
pub struct Layer2Weights {
    pub weights: Vec<Vec<Vec<Vec<f32>>>>,  // [50][6][5][5] // 修改：16 -> 50
    pub bias: Vec<f32>,                    // [50]           // 修改：16 -> 50
}

#[derive(Debug)]
pub struct Layer3Weights {
    pub weights: Vec<Vec<f32>>,            // [100][1250]
    pub bias: Vec<f32>,                    // [100]
}

#[derive(Debug)]
pub struct Layer4Weights {
    pub weights: Vec<Vec<f32>>,            // [10][100]
    pub bias: Vec<f32>,                    // [10]
}

fn convert_to_layer1(data: &[f32]) -> Layer1Weights {
    let mut weights = vec![vec![vec![vec![0.0; 5]; 5]; 1]; 6];
    let mut bias = vec![0.0; 6];
    
    // 每个卷积核是 5x5，加上1个偏置，总共 26 个参数
    // 有 6 个输出通道，所以总共 26 * 6 = 156 个参数
    for out_ch in 0..6 {
        // 处理权重
        for i in 0..5 {
            for j in 0..5 {
                weights[out_ch][0][i][j] = data[out_ch * 26 + i * 5 + j];
            }
        }
        // 处理偏置
        bias[out_ch] = data[out_ch * 26 + 25];
    }
    
    Layer1Weights { weights, bias }
}
fn convert_to_layer2(data: &[f32]) -> Layer2Weights {
    let mut weights = vec![vec![vec![vec![0.0; 5]; 5]; 6]; 50]; // 修改：16 -> 50
    let mut bias = vec![0.0; 50];                                // 修改：16 -> 50
    
    // 每个卷积核是 5x5x6，加上1个偏置，总共 (5*5*6 + 1) = 151 个参数
    // 有 50 个输出通道，所以总共 151 * 50 个参数
    for out_ch in 0..50 {                                        // 修改：16 -> 50
        // 处理权重
        for in_ch in 0..6 {
            for i in 0..5 {
                for j in 0..5 {
                    weights[out_ch][in_ch][i][j] = data[out_ch * 151 + in_ch * 25 + i * 5 + j];
                }
            }
        }
        // 处理偏置
        bias[out_ch] = data[out_ch * 151 + 150];
    }
    
    Layer2Weights { weights, bias }
}

fn convert_to_layer3(data: &[f32]) -> Layer3Weights {
    let mut weights = vec![vec![0.0; 1250]; 100];  // [100][1250]
    let mut bias = vec![0.0; 100];                 // [100]
    
    // 输入是 1250 (50*5*5)，输出是 100
    // 每个神经元有 1251 个参数（1250个权重 + 1个偏置）
    for out_ch in 0..100 {
        // 处理权重
        for i in 0..1250 {
            weights[out_ch][i] = data[out_ch * 1251 + i];
        }
        // 处理偏置
        bias[out_ch] = data[out_ch * 1251 + 1250];
    }
    
    Layer3Weights { weights, bias }
}

fn convert_to_layer4(data: &[f32]) -> Layer4Weights {
    let mut weights = vec![vec![0.0; 100]; 10];    // [10][100]
    let mut bias = vec![0.0; 10];                  // [10]
    
    // 输入是 100，输出是 10
    // 每个神经元有 101 个参数（100个权重 + 1个偏置）
    for out_ch in 0..10 {
        // 处理权重
        for i in 0..100 {
            weights[out_ch][i] = data[out_ch * 101 + i];
        }
        // 处理偏置
        bias[out_ch] = data[out_ch * 101 + 100];
    }
    
    Layer4Weights { weights, bias }
}
