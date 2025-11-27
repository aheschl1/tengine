pub mod base;
pub mod elementwise;

#[cfg(test)]
mod tests {
    use crate::{backend::cpu::Cpu, core::{meta::MetaTensorView, primitives::TensorBase, tensor::{AsView, AsViewMut, TensorAccess, TensorAccessMut}}};

    
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

    // Tests for non-inplace operations (consume view, return new tensor)

    #[test]
    fn test_add_not_inplace() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view + 5;
        
        // Result should be a new tensor with added values
        assert_eq!(result.raw, vec![6, 7, 8].into_boxed_slice());
        assert_eq!(*result.shape(), vec![3]);
        
        // Original tensor should be unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3].into_boxed_slice());
    }

    #[test]
    fn test_add_ref_not_inplace() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let value = 10;
        let view = tensor.view_mut();
        let result = view + &value;
        
        assert_eq!(result.raw, vec![11, 12, 13].into_boxed_slice());
        assert_eq!(tensor.raw, vec![1, 2, 3].into_boxed_slice());
    }

    #[test]
    fn test_sub_not_inplace() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view - 5;
        
        assert_eq!(result.raw, vec![5, 15, 25].into_boxed_slice());
        assert_eq!(tensor.raw, vec![10, 20, 30].into_boxed_slice());
    }

    #[test]
    fn test_sub_ref_not_inplace() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let value = 10;
        let view = tensor.view_mut();
        let result = view - &value;
        
        assert_eq!(result.raw, vec![0, 10, 20].into_boxed_slice());
        assert_eq!(tensor.raw, vec![10, 20, 30].into_boxed_slice());
    }

    #[test]
    fn test_mul_not_inplace() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view * 5;
        
        assert_eq!(result.raw, vec![5, 10, 15].into_boxed_slice());
        assert_eq!(tensor.raw, vec![1, 2, 3].into_boxed_slice());
    }

    #[test]
    fn test_mul_ref_not_inplace() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let value = 10;
        let view = tensor.view_mut();
        let result = view * &value;
        
        assert_eq!(result.raw, vec![10, 20, 30].into_boxed_slice());
        assert_eq!(tensor.raw, vec![1, 2, 3].into_boxed_slice());
    }

    #[test]
    fn test_add_not_inplace_with_slice() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let slice = view.slice_mut(0, 1).unwrap(); // Second row: [4, 5, 6]
        let result = slice + 100;
        
        // Result should be a new 1D tensor with shape [3]
        assert_eq!(result.raw, vec![104, 105, 106].into_boxed_slice());
        assert_eq!(*result.shape(), vec![3]);
        assert!(result.is_contiguous());
        
        // Original tensor should be unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
    }

    #[test]
    fn test_mul_not_inplace_with_noncontiguous_slice() {
        // Test with non-contiguous column slice
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let col_slice = view.slice_mut(1, 1).unwrap(); // Middle column: [2, 5]
        
        assert!(!col_slice.is_contiguous(), "Column slice should be non-contiguous");
        
        let result = col_slice * 3;
        
        // Result should be a new contiguous tensor
        assert_eq!(result.raw, vec![6, 15].into_boxed_slice());
        assert_eq!(*result.shape(), vec![2]);
        assert!(result.is_contiguous());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
    }

    #[test]
    fn test_sub_not_inplace_with_reshape() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
        let view = tensor.view_mut();
        let reshaped = view.view_as(vec![4]).unwrap();
        let result = reshaped - 5;
        
        // Result should have the reshaped dimensions
        assert_eq!(result.raw, vec![5, 15, 25, 35].into_boxed_slice());
        assert_eq!(*result.shape(), vec![4]);
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![10, 20, 30, 40].into_boxed_slice());
    }

    #[test]
    fn test_add_not_inplace_negative_values() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view + (-5);
        
        assert_eq!(result.raw, vec![5, 15, 25].into_boxed_slice());
        assert_eq!(tensor.raw, vec![10, 20, 30].into_boxed_slice());
    }

    #[test]
    fn test_mul_not_inplace_negative_values() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view * (-5);
        
        assert_eq!(result.raw, vec![-5, -10, -15].into_boxed_slice());
        assert_eq!(tensor.raw, vec![1, 2, 3].into_boxed_slice());
    }

    #[test]
    fn test_add_not_inplace_chained_slices() {
        // Test with chained slices to ensure view_to_owned handles complex cases
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        let mut depth_slice = view.slice_mut(0, 1).unwrap(); // Second depth: [5, 6, 7, 8]
        let row_slice = depth_slice.slice_mut(0, 0).unwrap(); // First row: [5, 6]
        
        let result = row_slice + 1000;
        
        assert_eq!(result.raw, vec![1005, 1006].into_boxed_slice());
        assert_eq!(*result.shape(), vec![2]);
        assert!(result.is_contiguous());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3, 4, 5, 6, 7, 8].into_boxed_slice());
    }

    #[test]
    fn test_mul_not_inplace_matrix() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let view = tensor.view_mut();
        let result = view * 10;
        
        assert_eq!(result.raw, vec![10, 20, 30, 40].into_boxed_slice());
        assert_eq!(*result.shape(), vec![2, 2]);
        assert!(result.is_contiguous());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3, 4].into_boxed_slice());
    }

    #[test]
    fn test_sub_not_inplace_scalar() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![100], vec![]).unwrap();
        let view = tensor.view_mut();
        let result = view - 50;
        
        assert_eq!(result.raw, vec![50].into_boxed_slice());
        assert_eq!(*result.shape(), vec![]);
        assert!(result.is_scalar());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![100].into_boxed_slice());
    }

    // Tests for non-inplace operations on non-mutable views (TensorView)
    
    #[test]
    fn test_add_immutable_view_inline() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4], vec![4]).unwrap();
        // Call operation directly on view() without storing in variable
        let result = tensor.view() + 10;
        
        assert_eq!(result.raw, vec![11, 12, 13, 14].into_boxed_slice());
        assert_eq!(*result.shape(), vec![4]);
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3, 4].into_boxed_slice());
    }

    #[test]
    fn test_add_immutable_view_ref() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![5, 10, 15], vec![3]).unwrap();
        let value = 100;
        let result = tensor.view() + &value;
        
        assert_eq!(result.raw, vec![105, 110, 115].into_boxed_slice());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![5, 10, 15].into_boxed_slice());
    }

    #[test]
    fn test_sub_immutable_view_inline() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![100, 200, 300], vec![3]).unwrap();
        let result = tensor.view() - 50;
        
        assert_eq!(result.raw, vec![50, 150, 250].into_boxed_slice());
        assert_eq!(*result.shape(), vec![3]);
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![100, 200, 300].into_boxed_slice());
    }

    #[test]
    fn test_sub_immutable_view_with_slice() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30, 40, 50, 60], vec![2, 3]).unwrap();
        // Slice to get first row, then subtract
        let result = tensor.view().slice(0, 0).unwrap() - 5;
        
        assert_eq!(result.raw, vec![5, 15, 25].into_boxed_slice());
        assert_eq!(*result.shape(), vec![3]);
        assert!(result.is_contiguous());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![10, 20, 30, 40, 50, 60].into_boxed_slice());
    }

    #[test]
    fn test_mul_immutable_view_inline() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![2, 4, 6, 8], vec![4]).unwrap();
        let result = tensor.view() * 3;
        
        assert_eq!(result.raw, vec![6, 12, 18, 24].into_boxed_slice());
        assert_eq!(*result.shape(), vec![4]);
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![2, 4, 6, 8].into_boxed_slice());
    }

    #[test]
    fn test_mul_immutable_view_matrix() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let result = tensor.view() * 10;
        
        assert_eq!(result.raw, vec![10, 20, 30, 40, 50, 60].into_boxed_slice());
        assert_eq!(*result.shape(), vec![2, 3]);
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
    }

    #[test]
    fn test_add_immutable_view_after_reshape() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4], vec![4]).unwrap();
        // Reshape then add, all inline
        let result = tensor.view().view_as(vec![2, 2]).unwrap() + 100;
        
        assert_eq!(result.raw, vec![101, 102, 103, 104].into_boxed_slice());
        assert_eq!(*result.shape(), vec![2, 2]);
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3, 4].into_boxed_slice());
    }

    #[test]
    fn test_sub_immutable_view_scalar() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![999], vec![]).unwrap();
        let result = tensor.view() - 99;
        
        assert_eq!(result.raw, vec![900].into_boxed_slice());
        assert_eq!(*result.shape(), vec![]);
        assert!(result.is_scalar());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![999].into_boxed_slice());
    }

    #[test]
    fn test_mul_immutable_view_noncontiguous_slice() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        // Get column (non-contiguous) then multiply
        let result = tensor.view().slice(1, 1).unwrap() * 5;
        
        assert_eq!(result.raw, vec![10, 25].into_boxed_slice());
        assert_eq!(*result.shape(), vec![2]);
        assert!(result.is_contiguous());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
    }

    #[test]
    fn test_add_immutable_view_chained_operations() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 4]).unwrap();
        // Chain slice and reshape, then add
        let result = tensor.view()
            .slice(0, 0).unwrap()  // Get first row: [1, 2, 3, 4]
            .view_as(vec![2, 2]).unwrap()  // Reshape to 2x2
            + 1000;
        
        assert_eq!(result.raw, vec![1001, 1002, 1003, 1004].into_boxed_slice());
        assert_eq!(*result.shape(), vec![2, 2]);
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3, 4, 5, 6, 7, 8].into_boxed_slice());
    }

    #[test]
    fn test_sub_immutable_view_negative_values() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![-10, -20, -30], vec![3]).unwrap();
        let result = tensor.view() - 5;
        
        assert_eq!(result.raw, vec![-15, -25, -35].into_boxed_slice());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![-10, -20, -30].into_boxed_slice());
    }

    #[test]
    fn test_mul_immutable_view_with_ref() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![7, 14, 21], vec![3]).unwrap();
        let multiplier = 2;
        let result = tensor.view() * &multiplier;
        
        assert_eq!(result.raw, vec![14, 28, 42].into_boxed_slice());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![7, 14, 21].into_boxed_slice());
    }
}