use std::fmt;
use std::str;
use utils::read_csv_job;

#[derive(Clone)]
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
