use matrix::DMatrix;

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
}

pub struct LinearRegression {
    options: LinearRegressionOptions,
    coefs: Vec<f64>,
    bias: f64,
}

impl LinearRegression {
    pub fn new(options: LinearRegressionOptions) -> LinearRegression {
        LinearRegression{ options: options, coefs: Vec::new(), bias: 0.0 }
    }

    fn fit(&mut self, train: &DMatrix<f64>, y: &[f64]) -> Result<(), String> {
        let n_samples = train.rows();
        let n_features = train.cols();
        if n_samples < 2 { return Err("Train array should contain number of rows > 1".to_string()); }
        if n_features == 0 { return Err("Train array should contain number of columns > 0".to_string()); }

        self.coefs = vec![0.0; n_features + 1];
        self.bias = 0.0;
        let mut grad = vec![0.0; n_features + 1];
        let mut discrepancy = vec![0.0; n_samples];
        for iter in 0..self.options.max_iter {
            for i in 0..n_samples {
                discrepancy[i] = -y[i] + self.bias + self.coefs.iter()
                                                               .zip(train.get_row(i))
                                                               .fold(0.0, |sum, (&c, &f)| sum + c * f);
            }
            grad[0] = discrepancy.iter().sum();
            for i in 0..n_samples {
                train.get_row(i).iter()
                     .enumerate()
                     .zip(discrepancy.iter())
                     .for_each(|((i, &x), &d)| grad[i+1] += d * x);
            }
            let step_scale = match self.options.stepping {
                Stepping::Constant(a) => a,
                Stepping::Decay(t) => t / (iter + 1) as f64,
            };
            let coef = 2.0 / n_samples as f64 * step_scale;
            grad.iter_mut().for_each(|g| *g *= coef);
            let J = 1.0 / n_samples as f64 * vec_dot(&discrepancy, &discrepancy);
            if (vec_norm(&grad) < self.options.x_eps) { break; }
        }
        Ok(())
    }
}
