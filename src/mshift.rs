use rand;
use kdtree::KdTree;
use kdtree::distance::squared_euclidean;

use matrix::DMatrix;
use utils::vec_sub;


pub enum SeedOptions {
    TrainSet,
    RandomChoice(usize),
    Custom(DMatrix<f64>),
}

pub struct MeanShiftOptions {
    seed: SeedOptions,
    bandwidth: f64,
    eps: f64,
    max_iter: usize,
}

impl MeanShiftOptions {
    pub fn new() -> MeanShiftOptions {
        MeanShiftOptions{ seed: SeedOptions::TrainSet,
                          bandwidth: 1.0,
                          eps: 1.0e-3,
                          max_iter: 300,
                        }
    }

    pub fn seed(mut self, seed: SeedOptions) -> MeanShiftOptions {
        self.seed = seed;
        self
    }

    pub fn bandwidth(mut self, bandwidth: f64) -> MeanShiftOptions {
        self.bandwidth = bandwidth;
        self
    }

    pub fn eps(mut self, eps: f64) -> MeanShiftOptions {
        self.eps = eps;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> MeanShiftOptions {
        self.max_iter = max_iter;
        self
    }
}

pub struct MeanShift {
    centers: DMatrix<f64>,
    options: MeanShiftOptions,
}

impl MeanShift {
    pub fn new(options: MeanShiftOptions) -> MeanShift {
        MeanShift{ centers: DMatrix::new(),
                   options: options,
                 }
    }

    pub fn fit(&mut self, train: &DMatrix<f64>) -> Result<(), String> {
        let n_dim = train.cols();
        let n_points = train.rows();
        if n_dim == 0 { return Err("Dimension is zero".to_string()); }
        if n_points == 0 { return Err("no points".to_string()); }

        // Инициализируем центры кластеров
        let mut seeds = match self.options.seed {
            SeedOptions::TrainSet => { train.clone() }
            SeedOptions::RandomChoice(n_seeds) if n_seeds <= n_points=> {
                let mut seeds: DMatrix<f64> = DMatrix::new_zeros(n_seeds, n_dim);
                let mut rng = rand::thread_rng();
                for (i, row_i) in rand::seq::sample_iter(&mut rng, 0..train.rows(), n_seeds).unwrap().iter().enumerate() {
                    seeds.set_row(i, train.get_row(*row_i))?;
                }
                seeds
            },
            SeedOptions::Custom(ref opt_seeds) if opt_seeds.cols() == n_dim && opt_seeds.rows() > 0 => {
                opt_seeds.clone()
            },
            _ => { return Err("Invalid seeds from options".to_string()); },
        };
        let n_seeds = seeds.rows();

        let eps = self.options.eps * self.options.bandwidth;
        let mut points_within = vec![0; n_seeds];

        // Двигаем сиды к центрам плотности, используем "плоское" ядро
        {
            // Инициализируем kdtree точками выборки
            let mut kdtree_train = KdTree::new(n_dim);
            for i in 0..n_points {
                kdtree_train.add(train.get_row(i), i).map_err(|e| format!("Can't add point {}: {:?}", i, e))?;
            }

            for seed_id in 0..n_seeds {
                let seed = seeds.get_row_mut(seed_id);
                let mut seed_prev = vec![0.0; n_dim];
                let mut n_iter = 0;
                while n_iter < self.options.max_iter {
                    seed_prev.clone_from_slice(seed);
                    let locals = kdtree_train.within(seed, self.options.bandwidth.powi(2), &squared_euclidean)
                                       .map_err(|e| format!("Can't find neighbors (seed_id = {}, n_iter = {}): {:?}", seed_id, n_iter, e))?;
                    if locals.is_empty() {
                        break;
                    }
                    let mut mean = vec![0.0; n_dim];
                    locals.iter().for_each(|&(_, id)| mean.iter_mut().zip(train.get_row(*id).iter()).for_each(|(ai, bi)| *ai += *bi));
                    mean.iter_mut().for_each(|v| *v /= locals.len() as f64);
                    seed.clone_from_slice(&mean);
                    points_within[seed_id] = locals.len();
                    n_iter += 1;
                    let diff = vec_sub(seed, &seed_prev);
                    if squared_euclidean(&diff, &[0.0, 0.0]).sqrt() < eps { break; }
                }
            }
        }

        // Объединяем близкие сиды
        // Инициализируем kdtree точками сидов
        let mut kdtree_seeds = KdTree::new_with_capacity(n_dim, n_seeds);
        let mut ids: Vec<usize> = Vec::with_capacity(n_seeds);
        for i in (0..n_seeds).filter(|i| points_within[*i] > 0) {
            kdtree_seeds.add(seeds.get_row(i), i).map_err(|e| format!("Can't add seed {}: {:?}", i, e))?;
            ids.push(i);
        }
        if kdtree_seeds.size() == 0 { return Err("No active seeds!".to_string()); }

        // сортируем сиды по кол-ву точек рядом(points_within) по убыванию
        ids.sort_unstable_by(|a, b| points_within[*b].cmp(&points_within[*a]));

        self.centers = DMatrix::new_zeros(0, n_dim);
        for id in &ids {
            if points_within[*id] == 0 { continue; }
            let point = seeds.get_row(*id);
            let locals = kdtree_seeds.within(point, self.options.bandwidth.powi(2), &squared_euclidean)
                               .map_err(|e| format!("Can't find neighbor seeds (ids = {}): {:?}", id, e))?;
            self.centers.append_row(point);
            locals.iter().for_each(|&(_, i)| points_within[*i] = 0);
        }

        Ok(())
    }

    fn predict_common(&self, test: &DMatrix<f64>) -> Result<KdTree<usize, &[f64]>, String> {
        let n_dim = test.cols();
        let n_clusters = self.n_clusters();
        if n_dim != self.centers.cols() {
            return Err(format!("train set dim {} != test set dim {}", n_dim, self.centers.cols()));
        }

        let mut kdtree_centers = KdTree::new_with_capacity(n_dim, n_clusters);
        for i in 0..n_clusters {
            kdtree_centers.add(self.centers.get_row(i), i).map_err(|e| format!("Can't add center {}: {:?}", i, e))?;
        }
        Ok(kdtree_centers)
    }

    pub fn predict(&self, test: &DMatrix<f64>) -> Result<Vec<u32>, String> {
        let n_points = test.rows();
        let mut labels: Vec<u32> = Vec::with_capacity(n_points);
        if n_points == 0 { return Ok(labels); }

        let kdtree_centers = self.predict_common(test)?;
        for i in 0..n_points {
            let locals = kdtree_centers.nearest(test.get_row(i), 1, &squared_euclidean)
                               .map_err(|e| format!("Can't find neighbor center (i = {}): {:?}", i, e))?;
            locals.iter().for_each(|&(_, id)| labels.push(*id as u32));
        }
        debug_assert!(labels.len() == n_points);
        Ok(labels)
    }

    pub fn predict_w_bandwidth(&self, test: &DMatrix<f64>) -> Result<Vec<i32>, String> {
        let n_points = test.rows();
        let mut labels: Vec<i32> = Vec::with_capacity(n_points);
        if n_points == 0 { return Ok(labels); }

        let kdtree_centers = self.predict_common(test)?;
        for i in 0..n_points {
            let locals = kdtree_centers.nearest(test.get_row(i), 1, &squared_euclidean)
                               .map_err(|e| format!("Can't find neighbor center (i = {}): {:?}", i, e))?;
            for (dist, id) in locals {
                labels.push( if dist < self.options.bandwidth.powi(2) {
                    *id as i32
                } else {
                    -1
                });
            }
        }
        debug_assert!(labels.len() == n_points);
        Ok(labels)
    }

    pub fn n_clusters(&self) -> usize {
        self.centers.rows()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use utils::accuracy_perm;
    use utils::write_csv_col;

    #[test]
    fn blobs() {
        let train: DMatrix<f64> = DMatrix::from_csv("data/blobs.csv", 1, ',', Some(&[0, 1])).unwrap();
        let labels: DMatrix<u32> = DMatrix::from_csv("data/blobs.csv", 1, ',', Some(&[2])).unwrap();
        for bandwidth in &[1.0, 2.0] {
            let mut ms = MeanShift::new( MeanShiftOptions::new()
                                                   .bandwidth(*bandwidth)
                                       );
            ms.fit(&train).unwrap();
            println!("n_clusters = \n{:?}", ms.n_clusters());
            assert_eq!(ms.n_clusters(), if *bandwidth == 1.0 { 7 } else { 3 } );
            // println!("centers = \n{}", ms.centers);
            let plabels = ms.predict(&train).unwrap();
            let labels_ids: Vec<_> = (0..ms.n_clusters() as u32).collect();
            let accuracy = accuracy_perm(labels.data(), &plabels, &labels_ids);
            println!("accuracy_perm = {}", accuracy);
            assert!((accuracy -  if *bandwidth == 1.0 { 0.938 } else { 1.0 }).abs() < 1e-3);
            // write_csv_col("output/blobs_mean.csv", &plabels, None).unwrap();
        }
    }

    // #[test]
    // fn foursquare() {
    //     let filename = "/Users/dmitry/code/courses/coursera/machine_learning_yandex/unsupervised_learning_w1/checkins_proc.dat";
    //     let train: DMatrix<f64> = DMatrix::from_csv(filename, 1, ',', Some(&[0, 1])).unwrap();
    //     let mut ms = MeanShift::new( MeanShiftOptions::new()
    //                                        .bandwidth(0.3)
    //                                        .seed(SeedOptions::RandomChoice(1000))
    //                                );
    //     ms.fit(&train).unwrap();
    //     println!("n_clusters = \n{:?}", ms.n_clusters());
    //     println!("centers = \n{}", ms.centers);
    // }
}
