
macro_rules! tget {
    ($tensor:expr $(, $idx:expr)+ $(,)?) => {{
        $tensor
            .get(Idx::Coord(&[$($idx),*]))
    }};
}

macro_rules! tset {
    ($tensor:expr, v:$value:expr $(, $idx:expr)+ $(,)?) => {{
        $tensor
            .set(Idx::Coord(&[$($idx),*]), $value)
    }};
}


#[cfg(test)]
mod tests {
    use crate::core::{CpuTensor, Shape, value::TensorValue, tensor::{AsView, AsViewMut, TensorAccess, TensorAccessMut}};
    use crate::core::idx::Idx;


    fn make_tensor<T: TensorValue>(buf: Vec<T>, shape: Shape) -> CpuTensor<T> {
        CpuTensor::from_buf(buf, shape).unwrap()
    }

    #[test]
    fn test_tget_macro() {
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);

        assert_eq!(tget!(tensor.view(), 0, 0).unwrap(), 1);
        assert_eq!(tget!(tensor.view(), 0, 2).unwrap(), 3);
        assert_eq!(tget!(tensor.view(), 1, 1).unwrap(), 5);
    }

    // expect failure for too many indices
    #[test]
    #[should_panic]
    fn test_tget_macro_too_many_indices() {
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let _ = tget!(tensor.view(), 0, 0, 0).unwrap();
    }

    // test tset macro
    #[test]
    fn test_tset_macro() {
        let mut tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        assert!(matches!(tset!(tensor.view_mut(), v: 99, 1, 2), Ok(())));
        assert_eq!(tget!(tensor.view(), 1, 2).unwrap(), 99);
    }
}