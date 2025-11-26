pub mod base;
pub mod elementwise;

#[cfg(test)]
mod tests {
    use crate::{backend::cpu::Cpu, core::{primitives::TensorBase, tensor::{AsViewMut, TensorAccessMut}}};

    
    #[test]
    fn test_add() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view += 5;

        let expected = TensorBase::<Cpu, i32>::from_buf(vec![6, 7, 8], vec![3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_add_ref() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let value = 10;
        let mut view = tensor.view_mut();
        view += &value;
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![11, 12, 13], vec![3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_add_negative() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view += -5;
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![5, 15, 25], vec![3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    // same for sub
    #[test]
    fn test_sub() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view -= 5;
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![5, 15, 25], vec![3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_sub_ref() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let value = 10;
        let mut view = tensor.view_mut();
        view -= &value;
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![0, 10, 20], vec![3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_sub_negative() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view -= -5;
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![6, 7, 8], vec![3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    // same for mul

    #[test]
    fn test_mul() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view *= 5;
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![5, 10, 15], vec![3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_mul_ref() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let value = 10;
        let mut view = tensor.view_mut();
        view *= &value;
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_mul_negative() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view *= -5;
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![-5, -10, -15], vec![3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    // Tests with reshaping/slicing

    #[test]
    fn test_add_after_reshape() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let view = tensor.view_mut();
        let mut reshaped = view.view_as(vec![3, 2]).unwrap();
        reshaped += 10;
        
        // Original tensor should be modified
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![11, 12, 13, 14, 15, 16], vec![2, 3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_add_after_slice() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 1).unwrap(); // Second row: [4, 5, 6]
        
        // Verify slice values before mutation
        use crate::core::tensor::TensorAccess;
        assert_eq!(slice.get(vec![0]).unwrap(), 4, "Slice[0] should be 4");
        assert_eq!(slice.get(vec![1]).unwrap(), 5, "Slice[1] should be 5");
        assert_eq!(slice.get(vec![2]).unwrap(), 6, "Slice[2] should be 6");
        
        slice += 100;
        
        // Only the sliced part should be modified
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 104, 105, 106], vec![2, 3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_add_after_slice_and_reshape() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let slice = view.slice_mut(0, 1).unwrap(); // Second row: [4, 5, 6]
        let mut reshaped = slice.view_as(vec![1, 3]).unwrap();
        reshaped += 50;
        
        // Only the sliced part should be modified
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 54, 55, 56], vec![2, 3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_sub_after_reshape() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
        let view = tensor.view_mut();
        let mut reshaped = view.view_as(vec![4]).unwrap();
        reshaped -= 5;
        
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![5, 15, 25, 35], vec![2, 2]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_mul_after_slice() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 0).unwrap(); // First depth slice: [1, 2, 3, 4]
        
        // Verify slice values before mutation
        use crate::core::tensor::TensorAccess;
        assert_eq!(slice.get(vec![0, 0]).unwrap(), 1, "Slice[0,0] should be 1");
        assert_eq!(slice.get(vec![0, 1]).unwrap(), 2, "Slice[0,1] should be 2");
        assert_eq!(slice.get(vec![1, 0]).unwrap(), 3, "Slice[1,0] should be 3");
        assert_eq!(slice.get(vec![1, 1]).unwrap(), 4, "Slice[1,1] should be 4");
        
        slice *= 10;
        
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30, 40, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_add_scalar_reshaped_to_matrix() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![42], vec![1]).unwrap();
        let view = tensor.view_mut();
        let mut reshaped = view.view_as(vec![1, 1]).unwrap();
        reshaped += 8;
        
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![50], vec![1]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_mul_after_column_slice() {
        // Create a matrix and slice a column (non-contiguous)
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let mut col_slice = view.slice_mut(1, 1).unwrap(); // Middle column: [2, 5]
        
        // Verify slice values (non-contiguous access)
        use crate::core::tensor::TensorAccess;
        assert_eq!(col_slice.get(vec![0]).unwrap(), 2, "Column slice[0] should be 2");
        assert_eq!(col_slice.get(vec![1]).unwrap(), 5, "Column slice[1] should be 5");
        
        col_slice *= 3;
        
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![1, 6, 3, 4, 15, 6], vec![2, 3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_sub_ref_after_reshape() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![100, 200, 300, 400], vec![2, 2]).unwrap();
        let value = 50;
        let view = tensor.view_mut();
        let mut reshaped = view.view_as(vec![4]).unwrap();
        reshaped -= &value;
        
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![50, 150, 250, 350], vec![2, 2]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_add_ref_after_slice_chain() {
        // Create a 3D tensor and chain multiple slices
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        let value = 1000;
        let mut view = tensor.view_mut();
        let mut depth_slice = view.slice_mut(0, 1).unwrap(); // Second depth: [6, 7, 8, 9] -> wait, buffer is [5,6,7,8]
        
        // Verify depth slice values
        use crate::core::tensor::TensorAccess;
        assert_eq!(depth_slice.get(vec![0, 0]).unwrap(), 5, "Depth slice[0,0] should be 5");
        assert_eq!(depth_slice.get(vec![0, 1]).unwrap(), 6, "Depth slice[0,1] should be 6");
        assert_eq!(depth_slice.get(vec![1, 0]).unwrap(), 7, "Depth slice[1,0] should be 7");
        assert_eq!(depth_slice.get(vec![1, 1]).unwrap(), 8, "Depth slice[1,1] should be 8");
        
        let mut row_slice = depth_slice.slice_mut(0, 0).unwrap(); // First row of that: [5, 6]
        assert_eq!(row_slice.get(vec![0]).unwrap(), 5, "Row slice[0] should be 5");
        assert_eq!(row_slice.get(vec![1]).unwrap(), 6, "Row slice[1] should be 6");
        
        row_slice += &value;
        
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 1005, 1006, 7, 8], vec![2, 2, 2]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }
}