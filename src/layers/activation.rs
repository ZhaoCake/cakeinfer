use ndarray::Array2;

pub fn sigmoid(x: f32) -> f32 {
    1.7159 * (0.66666667 * x).tanh()
}

pub fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let mut result = x.clone();
    
    // 对每一行(每个样本)进行 softmax
    for mut row in result.rows_mut() {
        let max_val = row.fold(f32::MIN, |a, &b| a.max(b));
        
        // 减去最大值以提高数值稳定性
        row.mapv_inplace(|x| (x - max_val).exp());
        
        let sum: f32 = row.sum();
        row.mapv_inplace(|x| x / sum);
    }
    
    result
}
