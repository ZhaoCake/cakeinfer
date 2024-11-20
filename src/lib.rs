pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product());
        Self { data, shape }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }
}
