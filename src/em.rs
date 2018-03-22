use std::f32;
use rand;

use matrix::DMatrix;
use kmeans::KMeans;


#[derive(Debug)]
struct SphericalGaussian {
    var: f32,
    mu: Vec<f32>,
}

impl SphericalGaussian {
    fn new(var: f32, mu: &[f32]) -> SphericalGaussian {
        SphericalGaussian{ var: var, mu: mu.to_vec() }
    }

    fn eval(&self, x: &[f32]) -> f32 {
        let k = 1.0 / ((2.0 * f32::consts::PI * self.var).powi(self.mu.len() as i32)).sqrt();
        let mut norm = 0.0;
        for (xi, mui) in x.iter().zip(self.mu.iter()) {
            norm += (xi - mui).powi(2); 
        }
        k * (-0.5 * norm / self.var).exp()
    }
}

impl Default for SphericalGaussian {
    fn default() -> SphericalGaussian {
        SphericalGaussian{ var: 1.0, mu: vec![0.0; 1] }
    }
}

struct EMCLustering {
}

impl EMCLustering {
    pub fn fit( train: &DMatrix<f32>,
            n_clusters_opt: Option<usize>,
            max_iter_opt: Option<usize>,
            n_init_opt: Option<usize>,
          ) -> Result<(Vec<SphericalGaussian>,Vec<f32>,  f32), String> {

        let n_init = match n_init_opt {
            Some(i) if i == 0 => {return Err("n_init_opt can't be zero!".to_string());},
            Some(i) => i,
            None => 10,
        };

        let (mut centroids, mut weights, mut log_likelihood) = EMCLustering::fit_one( train,
                                                                                      None,
                                                                                      n_clusters_opt,
                                                                                      max_iter_opt,
                                                                                    )?;
        for _ in 1..n_init {
            let (centroids_tmp, weigths_tmp, log_likelihood_tmp) = EMCLustering::fit_one( train,
                                                                                          None,
                                                                                          n_clusters_opt,
                                                                                          max_iter_opt,
                                                                                        )?;
            if log_likelihood_tmp > log_likelihood {
                centroids = centroids_tmp;
                log_likelihood = log_likelihood_tmp;
                weights = weigths_tmp;
            }
        }
        Ok((centroids, weights, log_likelihood))
    }

    pub fn fit_w_kmeans( train: &DMatrix<f32>,
                         n_clusters: usize,
                         max_iter_opt: Option<usize>,
                       ) -> Result<(Vec<SphericalGaussian>,Vec<f32>,  f32), String> {
        let (centroids, _, _) = KMeans::fit(&train, Some(n_clusters), None, None)?;
        let mut gaussian_centroids: Vec<SphericalGaussian> = Vec::with_capacity(n_clusters);
        for i in 0..n_clusters {
            gaussian_centroids.push(SphericalGaussian::new(1.0, centroids.get_row(i)));
        }
        EMCLustering::fit_one(&train, Some(gaussian_centroids), None, max_iter_opt)
    }

    pub fn predict_one(point: &[f32], centroids: &[SphericalGaussian], weights: &[f32]) -> Result<(u32, f32), String> {
        let n_dims = point.len();
        if centroids.is_empty() || weights.is_empty() { return Err("centroids or weights is empty".to_string()); }
        let mut max_id = 0;
        let mut max_score = centroids[max_id].eval(point) * weights[max_id];
        let mut sum_score = 0.0;

        for id in 1..centroids.len() {
            let score = centroids[id].eval(point) * weights[id];
            if score > max_score {
                max_score = score;
                max_id = id;
            }
            sum_score += score;
        }
        if sum_score == 0.0 { return Err("The point is far away..".to_string()); }
        Ok((max_id as u32, max_score / sum_score))
    }

    pub fn predict(test: &DMatrix<f32>, centroids: &[SphericalGaussian], weights: &[f32])
        -> Result<(Vec<u32>, Vec<f32>), String> {
        let n_dims = test.cols();
        if centroids.is_empty() || weights.is_empty() { return Err("centroids or weights is empty".to_string()); }

        let mut labels = Vec::with_capacity(test.rows());
        let mut probs = Vec::with_capacity(test.rows());

        for i in 0..test.rows() {
            let point = test.get_row(i);
            let mut max_id = 0;
            let mut max_score = centroids[max_id].eval(point) * weights[max_id];
            let mut sum_score = 0.0;

            for id in 1..centroids.len() {
                let score = centroids[id].eval(point) * weights[id];
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

    pub fn fit_one( train: &DMatrix<f32>,
                    centroids_opt: Option<Vec<SphericalGaussian>>,
                    n_clusters_opt: Option<usize>,
                    max_iter_opt: Option<usize>,
                  ) -> Result<(Vec<SphericalGaussian>, Vec<f32>, f32), String> {
        let mut centroids;
        let n_clusters;
        match centroids_opt {
            Some(c) => {
                centroids = c;
                n_clusters = centroids.len();
            },
            None => {
                n_clusters = match n_clusters_opt {
                    Some(n) => n,
                    None => {
                        return Err("both n_clusters_opt and centroids_opt are not porvided".to_string());
                    }
                };
                centroids = EMCLustering::take_random_centroids(train, n_clusters)?;
            }
        };
        let max_iter = match max_iter_opt {
            Some(i) => i,
            None => 100
        };
        if n_clusters == 0 {
            return Err("n_clusters are zero".to_string());
        }

        let n_samples = train.rows();
        let n_dims = train.cols();
        let mut w = vec![1.0 / n_clusters as f32; n_clusters];
        let mut g_mat: DMatrix<f32> = DMatrix::new_zeros(n_clusters, n_samples);
        let mut log_likelihood_prev = 0.0;

        for _ in 0..max_iter {
            let mut log_likelihood = 0.0;
            // 1. E-шаг
            for i in 0..n_samples {
                let mut denom = 0.0;
                for j in 0..n_clusters {
                    let wp = w[j] * centroids[j].eval(train.get_row(i));
                    denom += wp;
                    *g_mat.get_val_mut(j, i) = wp;
                }
                if denom != 0.0 {
                    // Случай, когда точка сильно далеко от всех центров кластеров, пробуем ее
                    // пропустить
                    for j in 0..n_clusters {
                        *g_mat.get_val_mut(j, i) /= denom;
                    }
                    log_likelihood += denom.ln();
                }
            }

            log_likelihood /= n_samples as f32;
            if (log_likelihood - log_likelihood_prev).abs() < 1e-12 { break; }
            log_likelihood_prev = log_likelihood;

            // 2. M-шаг
            // Обновляем w
            for j in 0..n_clusters {
                w[j] = 0.0;
                for i in 0..n_samples {
                    w[j] += g_mat.get_val(j, i);
                }
                w[j] /= n_samples as f32;
            }

            // Обновляем mu
            for j in 0..n_clusters {
                let ref mut mu = centroids[j].mu;
                for v in mu.iter_mut() {
                    *v = 0.0;
                }
                for i in 0..n_samples {
                    let g_ji = g_mat.get_val(j, i);
                    for (v, p) in mu.iter_mut().zip(train.get_row(i).iter()) {
                        *v += g_ji * *p;
                    }
                }
                let coef = 1.0 / (n_samples as f32 * w[j]);
                for v in mu.iter_mut() {
                    *v *= coef;
                }
            }

            // Обновляем var
            for centroid in centroids.iter_mut() {
                centroid.var = 0.0;
            }
            for j in 0..n_clusters {
                let g_j = g_mat.get_row(j);
                let mut var_diag = vec![0.0; n_dims];
                for i in 0..n_samples {
                    for (k, (v, p)) in centroids[j].mu.iter().zip(train.get_row(i).iter()).enumerate() {
                        var_diag[k] += (p - v).powi(2) * g_j[i];
                    }
                }
                let coef = 1.0 / (n_samples as f32 * w[j]);
                let mut sum = 0.0;
                for v in var_diag.iter_mut() {
                    sum += *v;
                }
                // Т.к. у нас гауссиан сферический (в вакууме), то все элементы на диагонали
                // матрицы ковариации д.б. одинаковые.
                centroids[j].var = sum * coef / (n_dims as f32);
            }
        }

        Ok((centroids, w, log_likelihood_prev))
    }

    pub fn take_random_centroids(train: &DMatrix<f32>, n_clusters: usize)
        -> Result<Vec<SphericalGaussian>, String> {
        let dim = train.cols(); 
        if dim == 0 || n_clusters == 0 || train.rows() == 0 {
            return Err("incorrect dimensions".to_string());
        }
        let mut centroids: Vec<SphericalGaussian> = Vec::with_capacity(n_clusters);

        let mut rng = rand::thread_rng();
        for row_i in rand::seq::sample_iter(&mut rng, 0..train.rows(), n_clusters).unwrap().iter() {
            centroids.push(SphericalGaussian::new(1.0, train.get_row(*row_i)));
        }

        Ok(centroids)
    }
}

#[test]
fn dim1() {
    let var = 5.5;
    let mu = 2.3;
    let x = 3.0;
    let sg = SphericalGaussian::new(var, &[mu]);
    let uni = 1.0 / (2.0 * f32::consts::PI * var).sqrt() * (-0.5 * (x - mu).powi(2) / var).exp();
    assert_eq!(uni, sg.eval(&[x]));
}

#[test]
fn em_blobs() {
    let train: DMatrix<f32> = DMatrix::from_csv("data/blobs.csv", 1, ',', Some(&[0, 1])).unwrap();
    let (centroids, weights, log_likelihood) = EMCLustering::fit(&train, Some(3), None, None).unwrap();
    println!("centroids = \n{:?}", centroids);
    println!("weights = \n{:?}", weights);
    println!("log_likelihood = \n{:?}", log_likelihood);
    assert!((log_likelihood - (-3.914224)).abs() < 1.0e-5);
}

#[test]
fn em_w_kmeans_blobs() {
    let train: DMatrix<f32> = DMatrix::from_csv("data/blobs.csv", 1, ',', Some(&[0, 1])).unwrap();
    let (centroids, weights, log_likelihood) = EMCLustering::fit_w_kmeans(&train, 3, None).unwrap();
    println!("centroids = \n{:?}", centroids);
    println!("weights = \n{:?}", weights);
    println!("log_likelihood = \n{:?}", log_likelihood);
    assert!((log_likelihood - (-3.914224)).abs() < 1.0e-5);

    let labels: DMatrix<u32> = DMatrix::from_csv("data/blobs.csv", 1, ',', Some(&[2])).unwrap();
    let (p_labels, p_prob) = EMCLustering::predict(&train, &centroids, &weights).unwrap();
}
