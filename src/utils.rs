pub type Dim = usize;
pub type Stride = Vec<usize>;
pub type Shape = Vec<Dim>;

pub(crate) fn shape_to_stride(shape: &Shape) -> Stride {
    let mut stride = vec![1; shape.len()];   
    for i in (0..shape.len()).rev(){
        if i < shape.len() - 1 {
            stride[i] = stride[i+1] * shape[i+1];
        }
    }
    stride
}
