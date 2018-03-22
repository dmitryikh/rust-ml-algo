use std::f64;
use rand;

use matrix::DMatrix;
use kmeans::KMeans;


#[derive(Debug, Clone)]
pub struct SphericalGaussian {
    var: f64,
    mu: Vec<f64>,
}

impl SphericalGaussian {
    fn new(var: f64, mu: &[f64]) -> SphericalGaussian {
        SphericalGaussian{ var: var, mu: mu.to_vec() }
    }

    fn eval(&self, x: &[f64]) -> f64 {
        let k = 1.0 / ((2.0 * f64::consts::PI * self.var).powi(self.mu.len() as i32)).sqrt();
        let mut norm = 0.0;
        for (xi, mui) in x.iter().zip(self.mu.iter()) {
            norm += (xi - mui).powi(2); 
        }
        k * (-0.5 * norm / self.var).exp()
    }
}

// impl Default for SphericalGaussian {
//     fn default() -> SphericalGaussian {
//         SphericalGaussian{ var: 1.0, mu: vec![0.0; 1] }
//     }
// }

pub enum Initialization {
    Random,
    KMeans,
    Components(Vec<SphericalGaussian>),
}

pub struct GaussianMixtureOptions {
    n_components: usize,
    init_scenario: Initialization,
    n_init: usize,
    max_iter: usize,
    tolerance: f64,
}

impl GaussianMixtureOptions {
    pub fn new() -> GaussianMixtureOptions {
        GaussianMixtureOptions {
            n_components: 1,
            init_scenario: Initialization::Random,
            n_init: 1,
            max_iter: 100,
            tolerance: 1e-12,
        }
    }

    pub fn n_components(mut self, n_components: usize) -> GaussianMixtureOptions {
        self.n_components = n_components;
        self
    }

    pub fn init_scenario(mut self, init_scenario: Initialization) -> GaussianMixtureOptions {
        self.init_scenario = init_scenario;
        self
    }

    pub fn n_init(mut self, n_init: usize) -> GaussianMixtureOptions {
        self.n_init = n_init;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> GaussianMixtureOptions {
        self.max_iter = max_iter;
        self
    }

    pub fn tolerance(mut self, tolerance: f64) -> GaussianMixtureOptions {
        self.tolerance = tolerance;
        self
    }
}

struct GaussianMixture {
    components: Vec<SphericalGaussian>,
    weights: Vec<f64>,
    log_likelihood: f64,
    options: GaussianMixtureOptions,
}

impl GaussianMixture {
    pub fn new(options: GaussianMixtureOptions) -> GaussianMixture {
        GaussianMixture {
            components: Vec::new(),
            weights: Vec::new(),
            log_likelihood: 0.0,
            options: options,
        }
    }

    pub fn fit(&mut self, train: &DMatrix<f64>) -> Result<(), String> {
        if let Initialization::Components(ref c) = self.options.init_scenario {
            self.options.n_components = c.len();
            self.options.n_init = 1;
        }
        if self.options.n_init == 0 { return Err("n_init_opt can't be zero!".to_string()); }

        // let (mut centroids, mut weights, mut log_likelihood) = self._fit_once(train)?;
        let (mut components, mut weights, mut log_likelihood) = self._fit_once(train)?;
        for _ in 1..self.options.n_init {
            let (components_tmp, weigths_tmp, log_likelihood_tmp) = self._fit_once(train)?;
            if log_likelihood_tmp > self.log_likelihood {
                components = components_tmp;
                log_likelihood = log_likelihood_tmp;
                weights = weigths_tmp;
            }
        }
        self.components = components;
        self.weights = weights;
        self.log_likelihood = log_likelihood;
        Ok(())
    }

    fn _fit_once(&self, train: &DMatrix<f64>) -> Result<(Vec<SphericalGaussian>, Vec<f64>, f64), String> {

        let mut  components = match self.options.init_scenario {
            Initialization::Random => { GaussianMixture::take_random_components(train, self.options.n_components)? },
            Initialization::KMeans => {
                let (centroids, _, _) = KMeans::fit(&train, Some(self.options.n_components), None, None)?;
                let mut components = Vec::with_capacity(self.options.n_components);
                for i in 0..self.options.n_components {
                    components.push(SphericalGaussian::new(1.0, centroids.get_row(i)));
                }
                components
            },
            Initialization::Components(ref c) => { c[..].to_vec() },
        };

        let n_samples = train.rows();
        let n_dims = train.cols();
        let n_components = self.options.n_components;

        let mut w = vec![1.0 / n_components as f64; n_components];
        let mut g_mat: DMatrix<f64> = DMatrix::new_zeros(n_components, n_samples);
        let mut log_likelihood_prev = 0.0;

        for _ in 0..self.options.max_iter {
            let mut log_likelihood = 0.0;
            // 1. E-шаг
            for i in 0..n_samples {
                let mut denom = 0.0;
                for j in 0..n_components {
                    let wp = w[j] * components[j].eval(train.get_row(i));
                    denom += wp;
                    *g_mat.get_val_mut(j, i) = wp;
                }
                if denom != 0.0 {
                    // Случай, когда точка сильно далеко от всех центров кластеров, пробуем ее
                    // пропустить
                    for j in 0..n_components {
                        *g_mat.get_val_mut(j, i) /= denom;
                    }
                    log_likelihood += denom.ln();
                }
            }

            log_likelihood /= n_samples as f64;
            if (log_likelihood - log_likelihood_prev).abs() < self.options.tolerance { break; }
            log_likelihood_prev = log_likelihood;

            // 2. M-шаг
            // Обновляем w
            for j in 0..n_components {
                w[j] = 0.0;
                for i in 0..n_samples {
                    w[j] += g_mat.get_val(j, i);
                }
                w[j] /= n_samples as f64;
            }

            // Обновляем mu
            for j in 0..n_components {
                let ref mut mu = components[j].mu;
                for v in mu.iter_mut() {
                    *v = 0.0;
                }
                for i in 0..n_samples {
                    let g_ji = g_mat.get_val(j, i);
                    for (v, p) in mu.iter_mut().zip(train.get_row(i).iter()) {
                        *v += g_ji * *p;
                    }
                }
                let coef = 1.0 / (n_samples as f64 * w[j]);
                for v in mu.iter_mut() {
                    *v *= coef;
                }
            }

            // Обновляем var
            for centroid in components.iter_mut() {
                centroid.var = 0.0;
            }
            for j in 0..n_components {
                let g_j = g_mat.get_row(j);
                let mut var_diag = vec![0.0; n_dims];
                for i in 0..n_samples {
                    for (k, (v, p)) in components[j].mu.iter().zip(train.get_row(i).iter()).enumerate() {
                        var_diag[k] += (p - v).powi(2) * g_j[i];
                    }
                }
                let coef = 1.0 / (n_samples as f64 * w[j]);
                let mut sum = 0.0;
                for v in var_diag.iter_mut() {
                    sum += *v;
                }
                // Т.к. у нас гауссиан сферический (в вакууме), то все элементы на диагонали
                // матрицы ковариации д.б. одинаковые.
                components[j].var = sum * coef / (n_dims as f64);
            }
        }

        Ok((components, w, log_likelihood_prev))
    }

    pub fn predict(&self, test: &DMatrix<f64>) -> Result<(Vec<u32>, Vec<f64>), String> {
        if self.components.is_empty() || self.weights.is_empty() { return Err("components or weights is empty".to_string()); }

        let mut labels = Vec::with_capacity(test.rows());
        let mut probs = Vec::with_capacity(test.rows());

        for i in 0..test.rows() {
            let point = test.get_row(i);
            let mut max_id = 0;
            let mut max_score = self.components[max_id].eval(point) * self.weights[max_id];
            let mut sum_score = 0.0;

            for id in 1..self.components.len() {
                let score = self.components[id].eval(point) * self.weights[id];
                if score > max_score {
                    max_score = score;
                    max_id = id;
                }
                sum_score += score;
            }
            labels.push(max_id as u32);
            probs.push(max_score / sum_score);
        }
        Ok((labels, probs))
    }

    fn take_random_components(train: &DMatrix<f64>, n_components: usize)
        -> Result<Vec<SphericalGaussian>, String> {
        let dim = train.cols(); 
        if dim == 0 || n_components == 0 || train.rows() == 0 {
            return Err("incorrect dimensions".to_string());
        }
        let mut components: Vec<SphericalGaussian> = Vec::with_capacity(n_components);

        let mut rng = rand::thread_rng();
        for row_i in rand::seq::sample_iter(&mut rng, 0..train.rows(), n_components).unwrap().iter() {
            components.push(SphericalGaussian::new(1.0, train.get_row(*row_i)));
        }

        Ok(components)
    }
}

#[test]
fn dim1() {
    let var = 5.5;
    let mu = 2.3;
    let x = 3.0;
    let sg = SphericalGaussian::new(var, &[mu]);
    let uni = 1.0 / (2.0 * f64::consts::PI * var).sqrt() * (-0.5 * (x - mu).powi(2) / var).exp();
    assert_eq!(uni, sg.eval(&[x]));
}

#[test]
fn em_blobs() {
    let train: DMatrix<f64> = DMatrix::from_csv("data/blobs.csv", 1, ',', Some(&[0, 1])).unwrap();
    let mut gm = GaussianMixture::new( GaussianMixtureOptions::new()
                                           .n_components(3)
                                           .n_init(10)
                                     );
    gm.fit(&train).unwrap();
    println!("components = \n{:?}", gm.components);
    println!("weights = \n{:?}", gm.weights);
    println!("log_likelihood = \n{:?}", gm.log_likelihood);
    assert!((gm.log_likelihood - (-3.914224)).abs() < 1.0e-5);
}

#[test]
fn em_w_kmeans_blobs() {
    let train: DMatrix<f64> = DMatrix::from_csv("data/blobs.csv", 1, ',', Some(&[0, 1])).unwrap();
    let mut gm = GaussianMixture::new( GaussianMixtureOptions::new()
                                           .n_components(3)
                                           .init_scenario(Initialization::KMeans)
                                     );
    gm.fit(&train).unwrap();
    println!("components = \n{:?}", gm.components);
    println!("weights = \n{:?}", gm.weights);
    println!("log_likelihood = \n{:?}", gm.log_likelihood);
    assert!((gm.log_likelihood - (-3.914224)).abs() < 1.0e-5);

    let labels: DMatrix<u32> = DMatrix::from_csv("data/blobs.csv", 1, ',', Some(&[2])).unwrap();
    let (p_labels, p_prob) = gm.predict(&train).unwrap();
}
