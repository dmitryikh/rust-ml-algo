use std;
use matrix::DMatrix;

pub struct NaiveBayes {
    class_num: usize,
    feature_num: usize,

    distr_matrix: Vec<Vec<Box<Distr>>>,
    class_prob: Vec<f64>,
}

impl NaiveBayes {
    pub fn new(class_num: usize, feature_num: usize, distr_matrix: Vec<Vec<Box<Distr>>>) -> Result<NaiveBayes, String> {
        if distr_matrix.len() != class_num {
            return Err("length of top level vector must be equal to class_num".to_string())
        };
        for v in distr_matrix.iter() {
            if v.len() != feature_num {
                return Err("length of each low level vector must be equal to feature_num".to_string())
            }
        }
        return Ok(NaiveBayes{class_num, feature_num, distr_matrix, class_prob: Vec::new()});
    }

    pub fn new_uniform<F: Fn() -> Box<Distr>>(class_num: usize, feature_num: usize, distr_gen: F) -> NaiveBayes {
        let mut distr_matrix: Vec<Vec<Box<Distr>>> = Vec::with_capacity(class_num);
        for _ in 0..class_num {
            let mut distr_vec = Vec::with_capacity(feature_num);
            for _ in 0..feature_num {
                distr_vec.push(distr_gen())
            }
            distr_matrix.push(distr_vec)
        };
        return NaiveBayes{
            class_num, feature_num, distr_matrix, class_prob: Vec::new()
        };
    }

    pub fn predict(&self, sample: &Vec<f64>) -> Result<Vec<f64>, String> {
        if sample.len() != self.feature_num {
            return Err("invalid sample length".to_string());
        }
        let mut result = Vec::with_capacity(self.class_num);
        for class_id in 0..self.class_num {
            result.push(self.predict_one(class_id, sample));
        }
        return Ok(result)
    }

    // train is assumed to have shape [feature_num, example_num]: same feature for all examples
    // is contained in the same row
    pub fn fit(& mut self, train: &Vec<&DMatrix<f64>>) -> Result<(), String> {
        if let Err(s) = self.check_train(train) {
            return Err(s);
        }
        let train_lengths: Vec<f64> = train.iter().map(|item| item.cols() as f64).collect();
        let total_trains = train_lengths.iter().fold(0f64, |sum, curr| sum + curr);
        self.class_prob = train_lengths.iter().map(|item| item / total_trains).collect();

        for class_id in 0..self.class_num {
            let class_train = train[class_id];
            for feature_id in 0..self.feature_num {
                if let Err(s) = self.fit_one(class_id, feature_id, class_train) {
                    return Err(s);
                }
            }
        }

        return Ok(());
    }

    fn predict_one(&self, class_id: usize, sample: &Vec<f64>) -> f64 {
        (0..self.feature_num)
            .fold(
                self.class_prob[class_id],
                |res, feature_id| res * self.distr_matrix[class_id][feature_id].get_prob(sample[feature_id])
            )
    }

    fn fit_one(& mut self, class_id: usize, feature_id: usize, train: &DMatrix<f64>) -> Result<(), String> {
        self.distr_matrix[class_id][feature_id].fit(train.get_row(feature_id))
    }

    fn check_train(&self, train: &Vec<&DMatrix<f64>>) -> Result<(), String> {
        if train.is_empty() {
            return Err("empty train obtained".to_string());
        }
        for i in 0..train.len() {
            if train[i].rows() != self.feature_num {
                return Err(format!("invalid feature num in sample {}", i));
            }
        }
        return Ok(());
    }
}

pub trait Distr {
    fn get_prob(&self, x: f64) -> f64;
    fn fit(& mut self, train: &[f64]) -> Result<(), String>;
}

#[derive(Debug)]
pub struct GaussianDistr {
    expectation: f64,
    dispersion: f64,
}

impl GaussianDistr {
    fn new() -> GaussianDistr {
        return GaussianDistr{expectation: 0., dispersion: 1.};
    }
}

impl Distr for GaussianDistr {
    fn get_prob(&self, x: f64) -> f64 {
        let dx = x - self.expectation;
        return 1. / (2. * std::f64::consts::PI * self.dispersion).sqrt() * (-dx * dx / (2. * self.dispersion)).exp();
    }

    fn fit(& mut self, train: &[f64]) -> Result<(), String> {
        self.expectation = train.iter().fold(0f64, |sum, curr| sum + curr) / train.len() as f64;
        self.dispersion = train.iter()
            .map(|x| (x - self.expectation) * (x - self.expectation))
            .fold(0f64, |sum, curr| sum + curr) / train.len() as f64;
        return Ok(());
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_bayes_fit_gaussian() {
        let train: Vec<f64> = vec!{1., 1., 1.};
        let mut g = GaussianDistr::new() as GaussianDistr;
        let train_res = g.fit(&train);
        assert_eq!(train_res, Ok(()));
        assert_eq!(1f64, g.expectation);
        assert_eq!(0f64, g.dispersion);
    }

    #[test]
    fn test_bayes_predict_gaussian() {
        let train: Vec<f64> = vec!{1., 2., 2., 1.};
        let mut g = GaussianDistr::new() as GaussianDistr;
        let train_res = g.fit(&train);
        assert_eq!(train_res, Ok(()));
        assert_eq!(1.5f64, g.expectation);
        assert_eq!(0.25f64, g.dispersion);

        let prediction = g.get_prob(1.5);
        let expected = 0.7978845608028654;
        assert!(expected - 1e-9 < prediction);
        assert!(prediction < expected + 1e-9);
    }

    #[test]
    fn test_bayes_new() {
        if let Err(e) = NaiveBayes::new(
            2, 2, vec!(
                vec!(Box::new(GaussianDistr::new()), Box::new(GaussianDistr::new())),
                vec!(Box::new(GaussianDistr::new()), Box::new(GaussianDistr::new()))
            )
        ) {
            assert!(false, e);
        }

        if let Err(e) = NaiveBayes::new(2, 2, vec!()) {
            assert_eq!(e, "length of top level vector must be equal to class_num".to_string())
        } else { assert!(false, "had to crash") }

        if let Err(e) = NaiveBayes::new(2, 2, vec!(vec!(), vec!())) {
            assert_eq!(e, "length of each low level vector must be equal to feature_num".to_string())
        } else { assert!(false, "had to crash") }
    }

    #[test]
    fn test_bayes_train_empty() {
        if let Err(e) = NaiveBayes::new_uniform(2, 2, || Box::new(GaussianDistr::new()))
            .fit(&vec!()){
            assert_eq!("empty train obtained", e)
        } else {
            assert!(false, "had to fail")
        }
    }

    #[test]
    fn test_bayes_train_bad_shaped() {
        let sample = DMatrix::from_row_slice(1, 1, &[1.]).unwrap();
        let train: &Vec<&DMatrix<f64>> = &vec!(&sample);
        if let Err(e) = NaiveBayes::new_uniform(2, 2, || Box::new(GaussianDistr::new()))
            .fit(train){
            assert_eq!("invalid feature num in sample 0", e)
        } else {
            assert!(false, "had to fail")
        }
    }

    #[test]
    fn test_bayes_predict_nan() {
        let train_class_1: DMatrix<f64> = DMatrix::from_row_slice(2, 1, &[
            0.,
            0.,
        ] as &[f64]).unwrap();
        let train_class_2: DMatrix<f64> = DMatrix::from_row_slice(2, 1, &[
            5.,
            5.,
        ] as &[f64]).unwrap();
        let total_train = vec!(&train_class_1, &train_class_2);
        let mut bayes = NaiveBayes::new_uniform(2, 2, || Box::new(GaussianDistr::new()));
        if let Err(e) = bayes.fit(&total_train) {
            assert!(false, e);
        }

        match bayes.predict(&vec!(0.5, 0.5)) {
            Ok(v) => assert!(v[0].is_nan() && v[1].is_nan()),
            Err(e) => assert!(false, e)
        };

        match bayes.predict(&vec!(5.5, 5.5)) {
            Ok(v) => assert!(v[1].is_nan() && v[0].is_nan()),
            Err(e) => assert!(false, e)
        };
    }

    #[test]
    fn test_bayes_predict_ok() {
        let train_class_1: DMatrix<f64> = DMatrix::from_row_slice(2, 4, &[
            0., 1., 0., 1.,
            0., 0., 1., 1.,
        ] as &[f64]).unwrap();
        let train_class_2: DMatrix<f64> = DMatrix::from_row_slice(2, 4, &[
            5., 6., 5., 6.,
            5., 5., 6., 6.,
        ] as &[f64]).unwrap();
        let total_train = vec!(&train_class_1, &train_class_2);
        let mut bayes = NaiveBayes::new_uniform(2, 2, || Box::new(GaussianDistr::new()));
        if let Err(e) = bayes.fit(&total_train) {
            assert!(false, e);
        }

        assert_eq!(bayes.class_prob, vec!(0.5, 0.5));

        match bayes.predict(&vec!(1.)) {
            Ok(v) => assert!(false, "had to fail"),
            Err(e) => assert_eq!("invalid sample length", e)
        }

        match bayes.predict(&vec!(0.5, 0.5)) {
            Ok(v) => assert!(v[0] > v[1]),
            Err(e) => assert!(false, e)
        };

        match bayes.predict(&vec!(5.5, 5.5)) {
            Ok(v) => assert!(v[1] > v[0]),
            Err(e) => assert!(false, e)
        };
    }
}

