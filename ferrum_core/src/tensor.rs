//! The fundamental data structure: a flat `Vec<f32>` with a shape.
#![warn(missing_docs)]
use crate::error::{InferError, Result};

#[derive(Clone, Debug, PartialEq)]
/// A dense row-major tensor: a flat `Vec<f32>` plus its shape.
pub struct Tensor {
    /// Dimension sizes; their product equals `data.len()`.
    pub shape: Vec<usize>,
    /// Elements in row-major order.
    pub data: Vec<f32>,
}

impl Tensor {
    /// Build a tensor, failing with `ShapeMismatch` if the shape's product differs from `data.len()`.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(InferError::ShapeMismatch {
                expected,
                got: data.len(),
            });
        }
        Ok(Self { shape, data })
    }

    /// A tensor of the given shape filled with `0.0`.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; n],
        }
    }

    /// A `rows x cols` matrix; fails if `data.len() != rows * cols`.
    pub fn matrix(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self> {
        Self::new(vec![rows, cols], data)
    }

    /// A rank-1 tensor holding `data`.
    pub fn vector(data: Vec<f32>) -> Self {
        let n = data.len();
        Self {
            shape: vec![n],
            data,
        }
    }

    /// A `1 x n` matrix from `data`.
    pub fn row(data: Vec<f32>) -> Result<Self> {
        let n = data.len();
        Self::matrix(1, n, data)
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }
    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// `(rows, cols)` of a rank-2 tensor; fails with `NotAMatrix` otherwise.
    pub fn matrix_dims(&self) -> Result<(usize, usize)> {
        match self.shape.as_slice() {
            [r, c] => Ok((*r, *c)),
            _ => Err(InferError::NotAMatrix(self.shape.clone())),
        }
    }

    /// Element at row `r`, column `c` of a matrix (panics when out of range or not rank 2).
    pub fn at(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.shape[1] + c]
    }

    /// A copy of the data under a new shape; fails if the element count differs.
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Tensor> {
        Tensor::new(shape, self.data.clone())
    }

    /// A new tensor of the same shape with `f` applied to every element.
    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().copied().map(f).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_validates_element_count() {
        assert!(Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).is_ok());
        let err = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0]).unwrap_err();
        assert!(matches!(
            err,
            InferError::ShapeMismatch {
                expected: 4,
                got: 3
            }
        ));
    }

    #[test]
    fn zeros_has_correct_shape_and_data() {
        let t = Tensor::zeros(vec![3, 2]);
        assert_eq!(t.shape, vec![3, 2]);
        assert!(t.data.iter().all(|&x| x == 0.0));
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn at_indexes_row_major() {
        // [[0,1,2],[3,4,5]]
        let m = Tensor::matrix(2, 3, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(m.at(0, 2), 2.0);
        assert_eq!(m.at(1, 0), 3.0);
        assert_eq!(m.at(1, 2), 5.0);
    }

    #[test]
    fn matrix_dims_rejects_non_matrix() {
        let v = Tensor::vector(vec![1.0, 2.0, 3.0]);
        assert!(matches!(v.matrix_dims(), Err(InferError::NotAMatrix(_))));
    }

    #[test]
    fn map_transforms_and_preserves_shape() {
        let t = Tensor::vector(vec![1.0, -2.0, 3.0]);
        let out = t.map(|x| x * 2.0);
        assert_eq!(out.data, vec![2.0, -4.0, 6.0]);
        assert_eq!(out.shape, t.shape);
    }

    #[test]
    fn reshape_checks_element_count() {
        let t = Tensor::matrix(2, 3, vec![0.0; 6]).unwrap();
        assert!(t.reshape(vec![3, 2]).is_ok());
        assert!(t.reshape(vec![2, 2]).is_err());
    }

    #[test]
    fn row_constructor() {
        let r = Tensor::row(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(r.shape, vec![1, 3]);
    }
}
