use crate::{backend::Backend, core::{meta::is_contiguous_relaxed, primitives::TensorBase, tensor::{AsTensor, AsView, TensorError}, value::TensorValue, MetaTensor, MetaTensorView, Shape, Strides}, ops::linalg::MatMul};


impl<L, R, T, B> MatMul<R, T, B> for L
where
    T: TensorValue,
    B: Backend<T>,
    L: AsView<T, B>,
    R: AsView<T, B>,
{
    fn matmul(&self, rhs: &R) -> Result<TensorBase<T, B>, TensorError> {
        let view = self.view();
        let rhs_view = rhs.view();

        let mut _copy_left = None;  // just to keep ownership
        let mut _copy_right = None;

        let lhs = if view.is_contiguous() {
            (view.meta, view.buf)
        } else{
            _copy_left = Some(view.contiguous());
            (_copy_left.as_ref().unwrap().meta.clone(), &_copy_left.as_ref().unwrap().buf)
        };

        let rhs = if rhs_view.is_contiguous() {
            (rhs_view.meta, rhs_view.buf)
        } else{
            _copy_right = Some(rhs_view.contiguous());
            (_copy_right.as_ref().unwrap().meta.clone(), &_copy_right.as_ref().unwrap().buf)
        };
        
        let ((lshape, lstrides), (rshape, rstrides), _result_params) = get_matmul_params(&lhs.0, &rhs.0)?;

        // ensure contiguous
        if !is_contiguous_relaxed(&lshape, &lstrides) {
            return Err(TensorError::ContiguityError)
        } else if !is_contiguous_relaxed(&rshape, &rstrides) {
            return Err(TensorError::ContiguityError)
        }
        
        // let res = TensorBase::<T, B>::zeros(shape);

        panic!("matmul not yet implemented")
    }
}

/// given two operands, computes new strides and shape (batched)
/// and returns the resultant buffer shape
fn get_matmul_params(
    lhs_meta: &MetaTensor,
    rhs_meta: &MetaTensor,
) -> Result<((Shape, Strides), (Shape, Strides), (Shape, Strides)), TensorError> {
    // check dimensions TODO INCREASE VALIDITY
    if lhs_meta.rank() < 2 || rhs_meta.rank() < 2 {
        return Err(TensorError::InvalidShape);
    }

    let squashed_left_shape = lhs_meta.shape.squash_leading_dims(lhs_meta.rank() - 2);
    let squashed_right_shape = rhs_meta.shape.squash_leading_dims(rhs_meta.rank() - 2);

    let squashed_left_stride = lhs_meta.strides.squash_leading_dims(lhs_meta.rank() - 2);
    let squashed_right_stride = rhs_meta.strides.squash_leading_dims(rhs_meta.rank() - 2);

    // let result_shape = vec![squash];
    let result_shape = Shape(vec![]);  // TODO: implement result shape computation

    Ok(((squashed_left_shape, squashed_left_stride), (squashed_right_shape, squashed_right_stride), (result_shape, Strides(vec![]))))
}
