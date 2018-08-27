use std;
use std::f64::consts;
use matrix::DMatrix;

const MIN_PROBABILITY: f64 = 1e-9;
const MIN_STD: f64 = 1e-5;

pub struct NaiveBayes<T: Distribution>  {
    class_prob: Vec<f64>,
    n_features: usize,
    n_labels: usize,
    distributions: T,
}

impl<T: Distribution> NaiveBayes<T> {
    pub fn new() -> Self {
        return NaiveBayes { class_prob: Vec::new(),
                            n_features: 0,
                            n_labels: 0,
                            distributions: T::new(),
                          }
    }

    // labels should be 0..n
    pub fn fit(&mut self, train: &DMatrix<f64>, labels: &[u32]) -> Result<(), String> {
        let n_samples = train.rows();
        let n_features = train.cols();
        if n_samples < 2 { return Err("Train array should contain number of rows > 1".to_string()); }
        if n_features == 0 { return Err("Train array should contain number of columns > 0".to_string()); }
        if labels.len() != n_samples { return Err("labels.len() != train.rows()".to_string()); }
        let n_labels = (*labels.iter().max().unwrap() + 1) as usize;

        self.distributions.fit(train, labels, n_labels)?;
        // calculate class_prob
        self.class_prob = vec![0.0; n_labels];
        labels.iter().for_each(|l| self.class_prob[*l as usize] += 1.0);
        for i in 0..n_labels {
            if self.class_prob[i] > 0.0 {
                self.class_prob[i] /= n_samples as f64;
            } else {
                self.class_prob[i] = MIN_PROBABILITY;
            }
        }

        self.n_labels = n_labels;
        self.n_features = n_features;
        return Ok(())
    }

    pub fn predict_proba(&self, test: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
        if self.n_labels == 0 || self.n_features == 0 { return Err("Model not trained".to_string()); }
        if self.n_features != test.cols() { return Err("Different number of features".to_string()); }
        let n_samples = test.rows();
        if n_samples == 0 { return Ok(DMatrix::new()); }
        let mut proba = DMatrix::new_zeros(n_samples, self.n_labels);
        for i in 0..n_samples {
            let log_proba = self.distributions.predict_log_proba(test.get_row(i))?;
            let proba_row = proba.get_row_mut(i);
            debug_assert!(log_proba.len() == proba_row.len());
            // norm coef. for class probablilities
            let mut logsumexp = 0.0;
            for j in 0..self.n_labels {
                logsumexp += (log_proba[j] + self.class_prob[j].ln()).exp();
            }
            logsumexp = logsumexp.ln();
            for j in 0..self.n_labels {
                proba_row[j] = (log_proba[j] + self.class_prob[j].ln() - logsumexp).exp();
            }
        }
        Ok(proba)
    }

    pub fn predict(&self, test: &DMatrix<f64>) -> Result<Vec<u32>, String> {
        // TODO: We do normalization in predict_proba, but this is useless for class prediction.
        let proba = self.predict_proba(test)?;
        let n_samples = test.rows();
        let mut labels = vec![0; n_samples];
        for i in 0..n_samples {
            let proba_row = proba.get_row(i);
            let mut max_idx = 0;
            let mut max_proba = proba_row[max_idx];
            for j in 1..proba_row.len() {
                if proba_row[j] > max_proba {
                    max_proba = proba_row[j];
                    max_idx = j;
                }
            }
            labels[i] = max_idx as u32;
        }
        Ok(labels)
    }
}

pub trait Distribution {
    fn new() -> Self;
    fn fit(&mut self, train: &DMatrix<f64>, labels: &[u32], n_labels: usize) -> Result<(), String>;
    fn predict_log_proba(&self, test: &[f64]) -> Result<Vec<f64>, String>;
}

pub struct Gaussian {
    // n_labels x n_features
    theta: DMatrix<f64>,
    sigma: DMatrix<f64>,
}

impl Gaussian {
    fn log_normal(x: f64, mu: f64, var: f64) -> f64 {
        -0.5 * (2.0 * consts::PI * var).ln() - (x - mu).powi(2) / (2.0 * var)
    }
}

impl Distribution for Gaussian {
    fn new() -> Gaussian {
        Gaussian{theta: DMatrix::new(), sigma: DMatrix::new()}
    }

    fn fit(&mut self, train: &DMatrix<f64>, labels: &[u32], n_labels: usize) -> Result<(), String> {
        let n_features = train.cols();
        let n_samples = train.rows();
        self.theta = DMatrix::new_zeros(n_labels, n_features);
        self.sigma = DMatrix::new_zeros(n_labels, n_features);

        // calc theta
        let mut class_nsamples = vec![0; n_labels];
        for i in 0..n_samples {
            let train_row = train.get_row(i);
            let label = labels[i] as usize;
            let theta_row = self.theta.get_row_mut(label);
            class_nsamples[label] += 1;
            for j in 0..n_features {
                theta_row[j] += train_row[j];
            }
        }
        for l in 0..n_labels {
            if class_nsamples[l] == 0 {
                continue
            }
            let theta_row = self.theta.get_row_mut(l);
            for j in 0..n_features {
                theta_row[j] /= class_nsamples[l] as f64;
            }
        }

        // calc sigma
        for i in 0..n_samples {
            let train_row = train.get_row(i);
            let label = labels[i] as usize;
            let sigma_row = self.sigma.get_row_mut(label);
            let theta_row = self.theta.get_row(label);
            for j in 0..n_features {
                sigma_row[j] += (train_row[j] - theta_row[j]).powi(2);
            }
        }
        for l in 0..n_labels {
            let sigma_row = self.sigma.get_row_mut(l);
            for j in 0..n_features {
                if sigma_row[j] < MIN_STD {
                    sigma_row[j] = MIN_STD;
                } else {
                    sigma_row[j] /= class_nsamples[l] as f64;
                }
            }
        }
        Ok(())
    }

    fn predict_log_proba(&self, sample: &[f64]) -> Result<Vec<f64>, String> {
        let n_features = self.theta.cols();
        debug_assert!(n_features == sample.len());
        let n_labels = self.theta.rows();
        let mut proba = vec![0.0; n_labels];
        for l in 0..n_labels {
            for j in 0..n_features {
                let tmp = Gaussian::log_normal(sample[j], self.theta.get_val(l, j), self.sigma.get_val(l, j));
                if tmp.is_nan() {
                    panic!(format!("sample = {}, theta = {}, sigma = {}", sample[j], self.theta.get_val(l, j), self.sigma.get_val(l, j)));
                }
                proba[l] += Gaussian::log_normal(sample[j], self.theta.get_val(l, j), self.sigma.get_val(l, j));
            }
        }
        Ok(proba)
    }
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[should_panic(expected = "Train array should contain number of rows > 1")]
    fn empty_train() {
        NaiveBayes::<Gaussian>::new().fit(&DMatrix::new_zeros(0, 2), &[]).unwrap();
    }

    #[test]
    fn simple_gaussian() {
        // Solution is compared with scikit-learn:
        //   from sklearn import naive_bayes
        //   import numpy as np
        //   train = np.array([1.0, 1.0,
        //                     2.0, 1.0,
        //                     0.5, 3.0,
        //                     0.1, 0.2,
        //                     0.4, 3.4,
        //                     5.6, 3.2]).reshape((6,2))
        //   y = np.array([1, 2, 2, 0, 1, 0])
        //   b.fit(train, y)
        //   print(b.class_prior_)
        //   print(b.predict_proba(train))
        
        let train = DMatrix::from_row_slice(6, 2, &[1.0, 1.0,
                                                    2.0, 1.0,
                                                    0.5, 3.0,
                                                    0.1, 0.2,
                                                    0.4, 3.4,
                                                    5.6, 3.2]).unwrap();
        let labels = vec![1, 2, 2, 0, 1, 0];
        let mut bayes = NaiveBayes::<Gaussian>::new();
        if let Err(e) = bayes.fit(&train, &labels) {
            assert!(false, e);
        }
        println!("class_prob: {:?}", bayes.class_prob);
        assert_eq!(bayes.class_prob, [1.0/3.0, 1.0/3.0, 1.0/3.0]);

        let correct_proba = DMatrix::from_row_slice(6, 3, &[8.84496280e-02, 5.21297222e-01, 3.90253150e-01,
                                                            2.96985781e-01, 2.01930940e-04, 7.02812288e-01,
                                                            4.84198770e-02, 7.46102317e-01, 2.05477806e-01,
                                                            3.37349561e-01, 3.54585005e-01, 3.08065434e-01,
                                                            6.25594481e-02, 7.45396483e-01, 1.92044069e-01,
                                                            9.99999639e-01, 2.58560977e-57, 3.60688128e-07]).unwrap();
        let proba = bayes.predict_proba(&train).unwrap();
        println!("proba: {}", proba);
        if !proba.equal(&correct_proba, 1e-7) {
            panic!("Wrong predictions!".to_string());
        }
    }
}