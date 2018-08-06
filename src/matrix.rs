use std::fmt;
use std::str;
use utils::read_csv_job;

#[derive(Clone, Debug)]
pub struct DMatrix<S>
{
    r: usize,
    c: usize,
    data: Vec<S>,
}

impl<S: Clone + Default> DMatrix<S>
{
    pub fn new() -> DMatrix<S> {
        DMatrix{r: 0, c: 0, data: Vec::new()}
    }

    pub fn new_zeros(r: usize, c: usize) -> DMatrix<S> {
        DMatrix{r: r, c: c, data: vec![S::default(); r * c]}
    }
    
    pub fn from_csv(fname: &str,
                    csv_skip_header: u32,
                    sep: char,
                    usecols_opt: Option<&[usize]>,
                   ) -> Result<DMatrix<S>, String>
        where S: str::FromStr
    {
        let mut mat: DMatrix<S> = DMatrix::new();
        read_csv_job::<S, _>(fname, csv_skip_header, sep, usecols_opt, false, |vec| {
            if mat.is_empty() {
                mat.resize_and_zero(0, vec.len());
            }
            mat.append_row(&vec);
            })?;
        Ok(mat)
    }

    pub fn from_row_slice(r: usize, c: usize, data: &[S]) -> Result<DMatrix<S>, String> {
        if data.len() != r * c {
            return Err("data length does not match given matrix size".to_string())
        }
        let mut result = DMatrix::new_zeros(r, c);
        for i in 0..r {
            result.set_row(i, &data[c*i..c*(i+1)])?
        }
        return Ok(result);
    }

    pub fn resize_and_zero(&mut self, r: usize, c: usize) {
        for v in &mut self.data {
            *v = S::default();
        }
        if r != self.r || c != self.c {
            self.data.resize((r * c) as usize, S::default());
            self.r = r;
            self.c = c;
        }
    }

    pub fn append_row(&mut self, row: &[S]) -> Option<usize> {
        if row.len() != self.c {
            return None;
        }
        for v in row {
            self.data.push(v.clone());
        }
        self.r += 1;
        return Some(self.r - 1)
    }

    pub fn get_val(&self, row: usize, col: usize) -> S {
        self.data[row * self.c + col].clone()
    }

    pub fn set_val(& mut self, val: S, row: usize, col: usize) {
        self.data[row*self.c + col] = val.clone();
    }

    pub fn get_val_mut(&mut self, row: usize, col: usize) -> &mut S {
        &mut self.data[row * self.c + col]
    }

    pub fn get_row(&self, row: usize) -> &[S] {
        return &self.data[(row * self.c)..((row + 1) * self.c)];
    }

    pub fn get_row_mut(&mut self, row: usize) -> &mut [S] {
        return &mut self.data[(row * self.c)..((row + 1) * self.c)];
    }

    pub fn set_row(&mut self, row: usize, new_values: &[S]) -> Result<(), String> {
        if new_values.len() != self.cols() {
            return Err(format!("new_values has incorrect length {}, should be {}", new_values.len(), self.cols()));
        }
        for (m, v) in self.get_row_mut(row).iter_mut().zip(new_values.iter()) {
            *m = v.clone();
        }
        Ok(())
    }

    pub fn data(&self) -> &[S] {
        return &self.data;
    }

    pub fn is_empty(&self) -> bool {
        return self.c == 0 || self.r == 0;
    }

    pub fn rows(&self) -> usize {
        return self.r;
    }

    pub fn cols(&self) -> usize {
        return self.c;
    }

    pub fn check_shape(&self) -> Result<(), String> {
        if self.cols() == 0 {
            Err("cols = 0".to_string())
        } else if self.rows() == 0 {
            Err("rows = 0".to_string())
        } else {
            Ok(())
        }
    }

    pub fn filter<F>(&self, f: F) -> DMatrix<S> where F: Fn(usize) -> bool {
        let mut result: DMatrix<S> = DMatrix::new_zeros(0, self.cols());
        for i in 0..self.rows() {
            if f(i) {
                result.append_row(self.get_row(i));
            }
        }
        return result;
    }

    pub fn transpose_copy(&self) -> DMatrix<S> {
        let mut result = DMatrix::new_zeros(self.cols(), self.rows());
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                result.set_val(self.get_val(i, j), j, i);
            }
        }
        return result;
    }
}

impl<S: fmt::Display> fmt::Display for DMatrix<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        const MAX_LEN: usize = 10;
        if self.r > MAX_LEN || self.c > MAX_LEN {
            write!(f, "[{} x {}]", self.r, self.c)?;
        }
        else if self.r > 0 || self.c > 0
        {
            let mut i = 0;
            for v in &self.data {
                if i == 0 {
                    write!(f, "[")?;
                }
                write!(f, "{}", v)?;
                i += 1;
                if i == self.c {
                    write!(f, "]\n")?;
                    i = 0;
                }
                else {
                    write!(f, ", ")?;
                }
            }
        }
        Ok(())
    }
}

#[test]
fn from_csv() {
    let mat: DMatrix<f32> = DMatrix::from_csv("data/test.csv", 1, ',', None).unwrap();
    assert_eq!(mat.rows(), 2);
    assert_eq!(mat.cols(), 3);
    assert_eq!(mat.get_row(0), [1.0, 2.0, 3.0]);
    assert_eq!(mat.get_row(1), [4.5, 1.4, 999.9]);
}

#[test]
fn from_csv_usecols() {
    let mat: DMatrix<f32> = DMatrix::from_csv("data/test.csv", 1, ',', Some(&[1, 2])).unwrap();
    assert_eq!(mat.rows(), 2);
    assert_eq!(mat.cols(), 2);
    assert_eq!(mat.get_row(0), [2.0, 3.0]);
    assert_eq!(mat.get_row(1), [1.4, 999.9]);
}

#[test]
fn test_filter() {
    let matrix_constructor = || DMatrix::from_row_slice(3, 3, &[
            1., 2., 3.,
            4., 5., 6.,
            7., 8., 9.
        ] as &[f64]).unwrap();

    let m1 = matrix_constructor();
    let m1_filtered = m1.filter(|i | true);
    assert_eq!(m1_filtered.cols(), 3);
    assert_eq!(m1_filtered.rows(), 3);

    let m2 = matrix_constructor();
    let m2_filtered = m2.filter(|i | false);
    assert_eq!(m2_filtered.cols(), 3);
    assert_eq!(m2_filtered.rows(), 0);

    let m3 = matrix_constructor();
    let m3_filtered = m3.filter(|i | m3.get_row(i)[0] > 2.);
    assert_eq!(m3_filtered.cols(), 3);
    assert_eq!(m3_filtered.rows(), 2);
    assert_eq!(m3_filtered.get_row(0), &[4., 5., 6.] as &[f64]);
    assert_eq!(m3_filtered.get_row(1), &[7., 8., 9.] as &[f64]);
}

#[test]
fn test_transpose() {
    let m1 = DMatrix::from_row_slice(2, 3, &[
            1., 2., 3.,
            4., 5., 6.
        ] as &[f64]).unwrap();
    let m2 = m1.transpose_copy();

    assert_eq!(m1.cols(), 3);
    assert_eq!(m1.rows(), 2);
    assert_eq!(m1.get_row(0), &[1., 2., 3.] as &[f64]);
    assert_eq!(m1.get_row(1), &[4., 5., 6.] as &[f64]);

    assert_eq!(m2.cols(), 2);
    assert_eq!(m2.rows(), 3);
    assert_eq!(m2.get_row(0), &[1., 4.] as &[f64]);
    assert_eq!(m2.get_row(1), &[2., 5.] as &[f64]);
    assert_eq!(m2.get_row(2), &[3., 6.] as &[f64]);
}