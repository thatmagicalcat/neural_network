use std::ops::{Index, IndexMut};

use rand::{thread_rng, Rng};

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    /// Create a row * col matrix with
    /// all its values set to 0
    pub fn zero(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    /// Create a row * col matrix with
    /// random values ranging from 0.0 to 1.0
    pub fn random(rows: usize, cols: usize) -> Self {
        let mut matrix = Self::zero(rows, cols);

        for row in 0..rows {
            for col in 0..cols {
                matrix[row][col] = thread_rng().gen_range(0.0..1.0);
            }
        }

        matrix
    }

    /// Adds two matrices
    /// # Panic
    /// If the two matrices are not of the same dimensions
    pub fn add(&self, rhs: &Self) -> Self {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            panic!("Attempted to add a matrix of incorrect dimensions");
        }

        let mut result = Matrix::zero(self.rows, self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                result[row][col] = self[row][col] + rhs[row][col];
            }
        }

        result
    }

    /// Subtracts two matrices
    /// # Panic
    /// If the two matrices are not of the same dimensions
    pub fn sub(&self, rhs: &Self) -> Self {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            panic!("Attempted to subtract a matrix of incorrect dimensions");
        }

        let mut result = Matrix::zero(self.rows, self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                result[row][col] = self[row][col] - rhs[row][col];
            }
        }

        result
    }

    /// Calculates the cross product
    /// of the two matrices
    /// # Panic
    /// If the two matrices are not in
    /// appropriate dimensions
    pub fn mul(&self, rhs: &Self) -> Self {
        if self.cols != rhs.rows {
            panic!("Attempted to multiply by a matrix of incorrect dimensions");
        }

        let mut result = Self::zero(self.rows, rhs.cols);

        for row in 0..self.rows {
            for col in 0..rhs.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self[row][k] * rhs[k][col];
                }

                result[row][col] = sum;
            }
        }

        result
    }

    /// Calculates the dot product
    /// of the two matrices
    /// # Panic
    /// If the two matrices are not of the same dimensions
    pub fn dot(&self, rhs: &Self) -> Self {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            panic!("Attempted to dot multiply a matrix of incorrect dimensions");
        }

        let mut result = Matrix::zero(self.rows, self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                result[row][col] = self[row][col] * rhs[row][col];
            }
        }

        result
    }

    /// Takes a closure and creates an iterator which calls that closure on each
    /// element.
    pub fn map(&self, func: impl Fn(f64) -> f64) -> Self {
        Matrix::from(
            self.data
                .clone()
                .into_iter()
                .map(|i| i.into_iter().map(|i| func(i)).collect())
                .collect::<Vec<Vec<_>>>(),
        )
    }

    pub fn transpose(&self) -> Self {
        let mut result = Matrix::zero(self.cols, self.rows);

        for row in 0..self.rows {
            for col in 0..self.cols {
                result[col][row] = self[row][col];
            }
        }

        result
    }
}

impl From<Vec<Vec<f64>>> for Matrix {
    fn from(value: Vec<Vec<f64>>) -> Self {
        Self {
            rows: value.len(),
            cols: value.first().and_then(|i| Some(i.len())).unwrap_or(0),
            data: value,
        }
    }
}

impl Index<usize> for Matrix {
    type Output = [f64];
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}
