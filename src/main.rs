use image::open;
use cakeinfer::predict;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 加载图像
    let img = open("./Data/0.bmp")?.to_luma8();
    
    // 执行预测
    let (pred_class, probabilities) = predict(&img);
    
    // 打印结果
    println!("Predicted digit: {}", pred_class);
    println!("Probabilities:");
    for (i, prob) in probabilities.iter().enumerate() {
        println!("  {}: {:.4}", i, prob);
    }
    
    Ok(())
}
