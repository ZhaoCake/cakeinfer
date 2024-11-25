pub fn relu(input: f32) -> f32 {
    if input > 0. {
        input
    } else {
        0.
    }
}
