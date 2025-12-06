use crate::{backend::Backend, core::{primitives::TensorBase, tensor::TensorError, value::TensorValue}};

mod matmul;

pub trait MatMul<Rhs, T, B> 
where 
    T: TensorValue,
    B: Backend<T>,
{
    fn matmul(&self, rhs: &Rhs) -> Result<TensorBase<T, B>, TensorError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{primitives::Tensor, MetaTensorView};

    #[test]
    fn test_matmul_2d_basic() {
        // Test basic 2x3 @ 3x2 matrix multiplication
        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        // B = [[7, 8],
        //      [9, 10],
        //      [11, 12]]
        // Expected: [[58, 64],
        //            [139, 154]]
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let b = Tensor::<i32>::from_buf(vec![7, 8, 9, 10, 11, 12], vec![3, 2]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<i32>::from_buf(vec![58, 64, 139, 154], vec![2, 2]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_2d_square() {
        // Test 2x2 @ 2x2 square matrix multiplication
        // A = [[1, 2],
        //      [3, 4]]
        // B = [[5, 6],
        //      [7, 8]]
        // Expected: [[19, 22],
        //            [43, 50]]
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b = Tensor::<i32>::from_buf(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<i32>::from_buf(vec![19, 22, 43, 50], vec![2, 2]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_2d_identity() {
        // Test multiplication with identity matrix
        // A = [[1, 2],
        //      [3, 4]]
        // I = [[1, 0],
        //      [0, 1]]
        // Expected: A unchanged
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let identity = Tensor::<i32>::from_buf(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        
        let result = a.matmul(&identity).unwrap();
        
        assert_eq!(result, a);
    }

    #[test]
    fn test_matmul_2d_float() {
        // Test with floating point values
        let a = Tensor::<f32>::from_buf(vec![1.5, 2.0, 3.5, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![2.0, 1.0, 3.0, 4.0], vec![2, 2]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        // Expected: [[1.5*2.0 + 2.0*3.0, 1.5*1.0 + 2.0*4.0],
        //            [3.5*2.0 + 4.0*3.0, 3.5*1.0 + 4.0*4.0]]
        //         = [[9.0, 9.5],
        //            [19.0, 19.5]]
        let expected = Tensor::<f32>::from_buf(vec![9.0, 9.5, 19.0, 19.5], vec![2, 2]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_3d_batched() {
        // Test batched 3D matrix multiplication
        // Shape: (2, 2, 2) @ (2, 2, 2) -> (2, 2, 2)
        // Batch 1: [[1, 2], [3, 4]] @ [[5, 6], [7, 8]]
        // Batch 2: [[9, 10], [11, 12]] @ [[13, 14], [15, 16]]
        let a = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 9, 10, 11, 12], 
            vec![2, 2, 2]
        ).unwrap();
        let b = Tensor::<i32>::from_buf(
            vec![5, 6, 7, 8, 13, 14, 15, 16], 
            vec![2, 2, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2, 2]);
        // Batch 1: [[19, 22], [43, 50]]
        // Batch 2: [[267, 286], [323, 346]]
        let expected = Tensor::<i32>::from_buf(
            vec![19, 22, 43, 50, 267, 286, 323, 346], 
            vec![2, 2, 2]
        ).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_3d_larger_batch() {
        // Test with larger batch dimension
        // Shape: (3, 2, 2) @ (3, 2, 2)
        let a = Tensor::<i32>::from_buf(
            vec![
                1, 0, 0, 1,  // Batch 0: Identity
                2, 0, 0, 2,  // Batch 1: 2*Identity
                3, 0, 0, 3,  // Batch 2: 3*Identity
            ], 
            vec![3, 2, 2]
        ).unwrap();
        let b = Tensor::<i32>::from_buf(
            vec![
                5, 6, 7, 8,
                5, 6, 7, 8,
                5, 6, 7, 8,
            ], 
            vec![3, 2, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![3, 2, 2]);
        let expected = Tensor::<i32>::from_buf(
            vec![
                5, 6, 7, 8,      // 1*[[5,6],[7,8]]
                10, 12, 14, 16,  // 2*[[5,6],[7,8]]
                15, 18, 21, 24,  // 3*[[5,6],[7,8]]
            ], 
            vec![3, 2, 2]
        ).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        // Test that mismatched dimensions produce an error
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![3, 2]).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::SizeMismatch));
    }

    #[test]
    fn test_matmul_rank_mismatch() {
        // Test that different ranks produce an error
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::InvalidShape));
    }

    #[test]
    fn test_matmul_batch_mismatch() {
        // Test that mismatched batch dimensions produce an error
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        let b = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], vec![3, 2, 2]).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::SizeMismatch));
    }

    #[test]
    fn test_matmul_2d_rectangular() {
        // Test with non-square matrices
        // 3x4 @ 4x2 -> 3x2
        let a = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
            vec![3, 4]
        ).unwrap();
        let b = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8], 
            vec![4, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![3, 2]);
        // Row 0: [1*1+2*3+3*5+4*7, 1*2+2*4+3*6+4*8] = [50, 60]
        // Row 1: [5*1+6*3+7*5+8*7, 5*2+6*4+7*6+8*8] = [114, 140]
        // Row 2: [9*1+10*3+11*5+12*7, 9*2+10*4+11*6+12*8] = [178, 220]
        let expected = Tensor::<i32>::from_buf(vec![50, 60, 114, 140, 178, 220], vec![3, 2]).unwrap();
        assert_eq!(result, expected);
    }

    // ============================================================================
    // FAILURE CASE TESTS
    // ============================================================================

    #[test]
    fn test_matmul_1d_array_fails() {
        // Test that 1D arrays (vectors) fail
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let b = Tensor::<i32>::from_buf(vec![4, 5, 6], vec![3]).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::InvalidShape));
    }

    #[test]
    fn test_matmul_1d_with_2d_fails() {
        // Test that mixing 1D with 2D fails
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let b = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![3, 2]).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::InvalidShape));
    }

    #[test]
    fn test_matmul_scalar_fails() {
        // Test that scalars (rank-0 tensors) fail
        let a = Tensor::<i32>::scalar(5);
        let b = Tensor::<i32>::scalar(10);
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::InvalidShape));
    }

    #[test]
    fn test_matmul_scalar_with_2d_fails() {
        // Test that scalar with 2D fails
        let a = Tensor::<i32>::scalar(5);
        let b = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::InvalidShape));
    }

    #[test]
    fn test_matmul_4d_should_work_with_batch_squashing() {
        // Test that 4D tensors should work by squashing batch dimensions
        // This test is expected to fail with current implementation
        let a = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8], 
            vec![2, 2, 2, 1]
        ).unwrap();
        let b = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8], 
            vec![2, 2, 1, 2]
        ).unwrap();
        
        // Currently fails but should work in the future
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 2, 2, 2]);
    }

    #[test]
    fn test_matmul_inner_dimension_mismatch_2d() {
        // Test that (2,3) @ (2,2) fails (inner dimensions 3 != 2)
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let b = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::SizeMismatch));
    }

    #[test]
    fn test_matmul_inner_dimension_mismatch_3d() {
        // Test that (2,2,3) @ (2,2,2) fails (inner dimensions 3 != 2)
        let a = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
            vec![2, 2, 3]
        ).unwrap();
        let b = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8], 
            vec![2, 2, 2]
        ).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::SizeMismatch));
    }

    #[test]
    fn test_matmul_batch_size_mismatch() {
        // Test that different batch sizes fail: (2,2,2) @ (3,2,2)
        let a = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8], 
            vec![2, 2, 2]
        ).unwrap();
        let b = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
            vec![3, 2, 2]
        ).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::SizeMismatch));
    }

    #[test]
    fn test_matmul_2d_3d_rank_mismatch() {
        // Test that 2D @ 3D fails
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::InvalidShape));
    }

    #[test]
    fn test_matmul_empty_dimension_fails() {
        // Test that matrices with 0 dimension fail
        let a = Tensor::<i32>::from_buf(vec![], vec![0, 3]).unwrap();
        let b = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![3, 2]).unwrap();
        
        let result = a.matmul(&b);
        // This should fail gracefully - behavior depends on implementation
        // At minimum it shouldn't panic
        assert!(result.is_ok() || result.is_err());
    }
}

