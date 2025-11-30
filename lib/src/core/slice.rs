use std::ops::{Range, RangeBounds, RangeFrom, RangeTo, RangeInclusive, RangeFull};
use super::{Shape, Stride};
use super::tensor::TensorError;


#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct Slice {
    pub start: Option<usize>,
    pub end: Option<usize>,
    pub step: Option<isize>,
}

impl Slice {
    pub fn new(start: Option<usize>, end: Option<usize>, step: Option<isize>) -> Self {
        Slice { start, end, step }
    }

    /// Creates a full slice (equivalent to `..`).
    pub fn full() -> Self {
        Slice { start: None, end: None, step: None }
    }

    pub fn step(mut self, step: isize) -> Self {
        self.step = Some(step);
        self
    }

    pub fn start(mut self, start: usize) -> Self {
        self.start = Some(start);
        self
    }

    pub fn end(mut self, end: usize) -> Self {
        self.end = Some(end);
        self
    }
}

impl From<Range<usize>> for Slice {
    fn from(range: Range<usize>) -> Self {
        // Automatically infer negative step if start > end
        if range.start > range.end {
            Slice {
                start: Some(range.start),
                end: Some(range.end),
                step: Some(-1),
            }
        } else {
            Slice {
                start: Some(range.start),
                end: Some(range.end),
                step: None,
            }
        }
    }
}

impl From<RangeInclusive<usize>> for Slice {
    fn from(range: RangeInclusive<usize>) -> Self {
        let start = *range.start();
        let end = *range.end();
        
        // Automatically infer negative step if start > end
        if start > end {
            Slice {
                start: Some(start),
                end: if end == 0 { None } else { Some(end - 1) }, // For negative step, end is exclusive, so end-1 to include 'end'
                step: Some(-1),
            }
        } else {
            Slice {
                start: Some(start),
                end: Some(end + 1), // Convert inclusive to exclusive
                step: None,
            }
        }
    }
}

impl From<RangeFrom<usize>> for Slice {
    fn from(range: RangeFrom<usize>) -> Self {
        Slice {
            start: Some(range.start),
            end: None,
            step: None,
        }
    }
}

impl From<RangeTo<usize>> for Slice {
    fn from(range: RangeTo<usize>) -> Self {
        Slice {
            start: None,
            end: Some(range.end),
            step: None,
        }
    }
}

impl From<RangeFull> for Slice {
    fn from(_: RangeFull) -> Self {
        Slice::full()
    }
}

impl From<usize> for Slice {
    fn from(idx: usize) -> Self {
        Slice {
            start: Some(idx),
            end: Some(idx),
            step: None,
        }
    }
}

impl RangeBounds<usize> for Slice {
    fn start_bound(&self) -> std::ops::Bound<&usize> {
        match &self.start {
            Some(s) => std::ops::Bound::Included(s),
            None => std::ops::Bound::Unbounded,
        }
    }

    fn end_bound(&self) -> std::ops::Bound<&usize> {
        match &self.end {
            Some(e) => std::ops::Bound::Excluded(e),
            None => std::ops::Bound::Unbounded,
        }
    }
}

/// Computes the new shape, stride, and offset for a view obtained by slicing
/// along a specific dimension.
///
/// # Behavior
/// - For single-index slices (start == end), removes that dimension
/// - For range slices, keeps the dimension but adjusts its size
/// - Supports negative steps for reverse iteration
///
/// # Errors
/// - `InvalidDim` if `dim` is out of bounds
/// - `IdxOutOfBounds` if indices are out of range
/// - `InvalidShape` if step is 0
pub(crate) fn compute_sliced_parameters(
    shape: &Shape,
    stride: &Stride,
    offset: usize,
    dim: usize,
    slice: impl Into<Slice>
) -> Result<(Shape, Stride, usize), TensorError>
{
    let slice: Slice = slice.into();

    if dim >= shape.len() {
        return Err(TensorError::InvalidDim);
    }
    
    let step: isize = slice.step.unwrap_or(1);
    if step == 0 {
        return Err(TensorError::InvalidShape);
    }

    let start: isize = match slice.start {
        Some(s) => s as isize,
        None if step > 0 => 0,
        None if step < 0 => (shape[dim] as isize) - 1,
        _ => unreachable!(),
    };

    let end: isize = match slice.end {
        Some(e) => e as isize,
        None if step > 0 => shape[dim] as isize,
        None if step < 0 => -1,
        _ => unreachable!(),
    };

    // Range validation
    if step > 0 && (start < 0 || start >= shape[dim] as isize || end < 0 || end > shape[dim] as isize) {
        return Err(TensorError::IdxOutOfBounds);
    }

    if step < 0 && (start < 0 || start >= shape[dim] as isize || end < -1 || end >= shape[dim] as isize) {
        return Err(TensorError::IdxOutOfBounds);
    }

    // Calculate the length of the resulting slice
    let len: usize = {
        if (step > 0 && start >= end) || (step < 0 && start <= end) {
            0
        } else {
            let dist = (start - end).abs() - 1;
            (dist as usize / step.unsigned_abs()) + 1
        }
    };

    if len == 0 {
        // Empty slice - collapse dimension
        let mut new_shape = shape.clone();
        let mut new_stride = stride.clone();
        new_shape.remove(dim);
        new_stride.remove(dim);

        let clamped_start = start.clamp(0, (shape[dim] - 1) as isize);
        let new_offset = offset + (clamped_start * stride[dim]).max(0) as usize;
        return Ok((new_shape, new_stride, new_offset));
    }

    let clamped_start = start.clamp(0, (shape[dim] - 1) as isize) as usize;
    let new_offset = offset + (clamped_start as isize * stride[dim]) as usize;

    let mut new_shape = shape.clone();
    let mut new_stride = stride.clone();

    new_shape[dim] = len;
    // For negative steps, stride becomes negative (reverse iteration)
    new_stride[dim] = stride[dim] * step;

    Ok((new_shape, new_stride, new_offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_from_range() {
        let slice: Slice = (2..5).into();
        assert_eq!(slice.start, Some(2));
        assert_eq!(slice.end, Some(5));
        assert_eq!(slice.step, None);
    }

    #[test]
    fn test_slice_from_usize() {
        let slice: Slice = 3.into();
        assert_eq!(slice.start, Some(3));
        assert_eq!(slice.end, Some(3));
        assert_eq!(slice.step, None);
    }

    #[test]
    fn test_slice_builder() {
        let slice = Slice::full().step(-2).start(10).end(2);
        assert_eq!(slice.start, Some(10));
        assert_eq!(slice.end, Some(2));
        assert_eq!(slice.step, Some(-2));
    }

    #[test]
    fn test_slice_range_bounds() {
        let slice = Slice::new(Some(2), Some(8), None);
        assert_eq!(slice.start_bound(), std::ops::Bound::Included(&2));
        assert_eq!(slice.end_bound(), std::ops::Bound::Excluded(&8));
    }
}
