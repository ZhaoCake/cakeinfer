pub mod data;
pub mod layers;
pub mod network;
pub mod weights;
pub mod inference;

pub use network::LeNet5;
pub use inference::predict;

#[inline]
pub fn conv(x: f32) -> f32 {
    // 如果C代码使用 _CHAR_WEIGHT_ 定义
    // return x / 10.0;
    
    // 如果C代码使用 _SHORT_WEIGHT_ 定义
    // return x / 100.0;
    
    // 默认情况
    x
}

