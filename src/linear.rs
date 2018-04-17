use matrix::DMatrix;
use utils::{vec_dot, vec_norm};

#[derive(Debug, Clone)]
pub enum Stepping {
    Constant(f64),
    Decay(f64),
}

pub struct LinearRegressionOptions {
    /// Критерий сходимости по изменению вектора коэффециентов
    x_eps: f64,
    eps: f64,
    max_iter: u32,
    stepping: Stepping,
}

impl LinearRegressionOptions {
    pub fn new() -> LinearRegressionOptions {
        LinearRegressionOptions{ x_eps: 1.0e-7,  eps: 1.0e-7, max_iter: 500, stepping: Stepping::Constant(0.01) }
    }

    pub fn x_eps(mut self, x_eps: f64) -> LinearRegressionOptions {
        self.x_eps = x_eps;
        self
    }

    pub fn eps(mut self, eps: f64) -> LinearRegressionOptions {
        self.eps = eps;
        self
    }

    pub fn max_iter(mut self, max_iter: u32) -> LinearRegressionOptions {
        self.max_iter = max_iter;
        self
    }

    pub fn stepping(mut self, stepping: Stepping) -> LinearRegressionOptions {
        self.stepping = stepping;
        self
    }
}

pub struct LinearRegression {
    options: LinearRegressionOptions,
    coefs: Vec<f64>,
}

impl LinearRegression {
    pub fn new(options: LinearRegressionOptions) -> LinearRegression {
        LinearRegression{ options: options, coefs: Vec::new() }
    }

    pub fn fit(&mut self, train: &DMatrix<f64>, y: &[f64]) -> Result<(), String> {
        let n_samples = train.rows();
        let n_features = train.cols();
        if n_samples < 2 { return Err("Train array should contain number of rows > 1".to_string()); }
        if n_features == 0 { return Err("Train array should contain number of columns > 0".to_string()); }
        if y.len() != n_samples { return Err("y.len() != train.rows()".to_string()); }

        self.coefs = vec![0.0; n_features + 1];
        let mut grad = vec![0.0; n_features + 1];
        let mut discrepancy = vec![0.0; n_samples];
        let n_samples_inv = 1.0 / n_samples as f64;
        #[allow(non_snake_case)]
        let mut J_prev = 0.0;
        #[allow(non_snake_case)]
        let mut J;
        for iter in 0..self.options.max_iter {
            for i in 0..n_samples {
                discrepancy[i] = -y[i] + self.coefs[0] + self.coefs.iter()
                                                               .skip(1)
                                                               .zip(train.get_row(i))
                                                               .fold(0.0, |sum, (&c, &f)| sum + c * f);
            }
            grad[0] = discrepancy.iter().sum();
            for i in 0..n_samples {
                train.get_row(i).iter()
                     .enumerate()
                     .for_each(|(j, &x)| grad[j + 1] += discrepancy[i] * x);
            }
            let step_scale = match self.options.stepping {
                Stepping::Constant(a) => a,
                Stepping::Decay(t) => t / (iter + 1) as f64,
            };
            let coef = n_samples_inv * step_scale;
            grad.iter_mut().for_each(|g| *g *= coef);
            if vec_norm(&grad) < self.options.x_eps { break; }

            J = 0.5 * n_samples_inv * vec_dot(&discrepancy, &discrepancy);
            if (J - J_prev).abs() < self.options.eps { break; }
            J_prev = J;

            self.coefs.iter_mut().zip(grad.iter()).for_each(|(c, g)| *c -= *g);
        }
        Ok(())
    }

    pub fn predict(&self, test: &DMatrix<f64>) -> Result<Vec<f64>, String> {
        let n_samples = test.rows();
        let n_features = test.cols();
        if self.coefs.len() == 0 { return Err("Model isn't trained".to_string()); }
        if n_features + 1 != self.coefs.len() { return Err("test.len() + 1 != coefs.len()".to_string()); }
        let mut predicts = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            predicts.push(self.coefs[0] + self.coefs.iter()
                                                    .skip(1)
                                                    .zip(test.get_row(i))
                                                    .fold(0.0, |sum, (&c, &f)| sum + c * f)
                         );
        }
        Ok(predicts)
    }

    pub fn bias(&self) -> Result<f64, String> {
        if self.coefs.len() == 0 { return Err("Model isn't trained".to_string()); }
        Ok(self.coefs[0])
    }

    pub fn coefficients(&self) -> Result<&[f64], String> {
        if self.coefs.len() == 0 { return Err("Model isn't trained".to_string()); }
        Ok(&self.coefs[1..])
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use utils::{rmse_error, mae_error};

    #[test]
    #[should_panic(expected = "Train array should contain number of rows > 1")]
    fn empty_train() {
        LinearRegression::new( LinearRegressionOptions::new() )
                         .fit(&DMatrix::new_zeros(0, 2), &[]).unwrap();
    }

    #[test]
    #[should_panic(expected = "Train array should contain number of columns > 0")]
    fn zero_dim() {
        LinearRegression::new( LinearRegressionOptions::new() )
                         .fit(&DMatrix::new_zeros(2, 0), &[0.0, 0.0]).unwrap();
    }

    #[test]
    #[should_panic(expected = "Train array should contain number of rows > 1")]
    fn one_point() {
        LinearRegression::new( LinearRegressionOptions::new() )
                         .fit(&DMatrix::new_zeros(1, 2), &[0.0, 0.0]).unwrap();
    }

    #[test]
    fn one_dim() {
        let mut train_x: DMatrix<f64> = DMatrix::new_zeros(0, 1);
        train_x.append_row(&[1.0]);
        train_x.append_row(&[4.0]);
        let train_y = vec![1.0, 3.0];
        let mut test_x: DMatrix<f64> = DMatrix::new_zeros(0, 1);
        test_x.append_row(&[1.0]);
        test_x.append_row(&[4.0]);
        test_x.append_row(&[0.0]);
        test_x.append_row(&[7.0]);
        let test_y = vec![1.0, 3.0, 1.0 - 2.0 / 3.0, 5.0];
        let mut lr = LinearRegression::new( LinearRegressionOptions::new()
                                                .stepping(Stepping::Decay(2.0))
                                                .max_iter(1000)
                                          );
        lr.fit(&train_x, &train_y).unwrap();
        println!("lr.coefs = {:?}", lr.coefs);
        let y = lr.predict(&test_x).unwrap();

        println!("y = {:?}", y);
        // let rmse = rmse_error(&test_y, &y);
        // let mae = mae_error(&test_y, &y);
        // println!("rmse = {}, mae = {}", rmse, mae);
        for (y_pred, y_true) in test_y.iter().zip(y) {
            assert!((y_pred - y_true).abs() < 1.0e-2);
        }
    }
}
