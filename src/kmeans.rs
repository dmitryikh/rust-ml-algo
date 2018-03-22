
use rand;
use rand::Rng;
use matrix::DMatrix;

pub struct KMeans
{ }

impl KMeans
{
    pub fn fit( train: &DMatrix<f32>,
            n_clusters_opt: Option<usize>,
            max_iter_opt: Option<usize>,
            n_init_opt: Option<usize>,
          ) -> Result<(DMatrix<f32>, Vec<u32>, f32), String> {

        let n_init = match n_init_opt {
            Some(i) if i == 0 => {return Err("n_init_opt can't be zero!".to_string());},
            Some(i) => i,
            None => 10,
        };

        let (mut centroids, mut samples_id, mut inertia) = KMeans::fit_one(train, None, n_clusters_opt, max_iter_opt)?;
        for _ in 1..n_init {
            let (centroids_tmp, samples_id_tmp, inertia_tmp) = KMeans::fit_one(train, None, n_clusters_opt, max_iter_opt)?;
            if inertia_tmp < inertia {
                centroids = centroids_tmp;
                inertia = inertia_tmp;
                samples_id = samples_id_tmp;
            }
        }
        Ok((centroids, samples_id, inertia))
    }

    pub fn fit_one( train: &DMatrix<f32>,
                    centroids_opt: Option<DMatrix<f32>>,
                    n_clusters_opt: Option<usize>,
                    max_iter_opt: Option<usize>,
                  ) -> Result<(DMatrix<f32>, Vec<u32>, f32), String> {
        let mut centroids;
        let n_clusters;
        match centroids_opt {
            Some(c) => {
                centroids = c;
                n_clusters = centroids.rows();
            },
            None => {
                n_clusters = match n_clusters_opt {
                    Some(n) => n,
                    None => {
                        return Err("both n_clusters_opt and centroids_opt are not porvided".to_string());
                    }
                };
                centroids = KMeans::take_random_centroids(train, n_clusters)?;
            }
        };
        let max_iter = match max_iter_opt {
            Some(i) => i,
            None => 100
        };

        let dist_sq_func = |p1: &[f32], p2: &[f32]| -> f32 {
            let mut dist = 0.0;
            for (p1_i, p2_i) in p1.iter().zip(p2.iter()) {
                dist += (p1_i - p2_i).powi(2);
            }
            dist
        };

        let mut inertia_prev = 0.0;

        let n_samples = train.rows();
        let mut samples_id = vec![0 as u32; n_samples];
        for _ in 0..max_iter {
            let mut is_changed = false;
            let mut inertia = 0.0;
            // 1. Считаем принадлежность точки к тому или иному кластеру
            for row in 0..n_samples {
                let train_row = train.get_row(row);
                let mut id_min: u32 = 0;
                let mut d_min = dist_sq_func(centroids.get_row(id_min as usize), train_row);
                for id in 1..centroids.rows() {
                    let d = dist_sq_func(centroids.get_row(id), train_row);
                    if d < d_min {
                        d_min = d;
                        id_min = id as u32;
                    }
                }
                inertia += d_min;
                if id_min != samples_id[row] {
                    is_changed = true;
                    samples_id[row] = id_min;
                }
            }

            if !is_changed || inertia == 0.0 { break; }
            if ((inertia - inertia_prev) / inertia).abs() <= 1e-4 { break; }
            inertia_prev = inertia;

            // 2. Пересчитываем центр кластера
            for id in 0..n_clusters {
                for v in centroids.get_row_mut(id) {
                    *v = 0.0
                }
            }

            let mut clusters_count = vec![0; n_clusters];
            for row in 0..n_samples {
                let train_row = train.get_row(row);
                let id_row = samples_id[row];
                for (c, p) in centroids.get_row_mut(id_row as usize).iter_mut().zip(train_row.iter()) {
                    *c += *p;
                }
                clusters_count[id_row as usize] += 1;
            }

            for id in 0..n_clusters {
                if clusters_count[id] != 0 {
                    for v in centroids.get_row_mut(id) {
                        *v /= clusters_count[id] as f32;
                    }
                }
            }
        }
        Ok((centroids, samples_id, inertia_prev))
    }

    pub fn gen_random_centroids(train: &DMatrix<f32>, n_clusters: usize) -> Result<DMatrix<f32>, String> {
        let dim = train.cols(); 
        if dim == 0 || n_clusters == 0 || train.rows() == 0 {
            return Err("incorrect dimensions".to_string());
        }
        let mut centroids: DMatrix<f32> = DMatrix::new_zeros(n_clusters, dim);
        // 1. Получить границы по каждой размерности
        let mut min_row = train.get_row(0).to_vec();
        let mut max_row = train.get_row(0).to_vec();
        
        for i in 0..train.rows() {
            let row = train.get_row(i);
            for j in 0..dim {
                if row[j] > max_row[j] {max_row[j] = row[j]}
                if row[j] < min_row[j] {min_row[j] = row[j]}
            }
        }

        // 2. Сгенерировать центры кластеров внутри диапазонов min_row, max_row
        for i in 0..n_clusters {
            let row = centroids.get_row_mut(i);
            for j in 0..dim {
                if min_row[j] == max_row[j] {
                    row[j] = min_row[j]
                } else {
                    row[j] = rand::thread_rng().gen_range(min_row[j], max_row[j]);
                }
            }
        }

        Ok(centroids)
    }

    pub fn take_random_centroids(train: &DMatrix<f32>, n_clusters: usize) -> Result<DMatrix<f32>, String> {
        let dim = train.cols(); 
        if dim == 0 || n_clusters == 0 || train.rows() == 0 {
            return Err("incorrect dimensions".to_string());
        }
        let mut centroids: DMatrix<f32> = DMatrix::new_zeros(n_clusters, dim);

        let mut rng = rand::thread_rng();
        for (i, row_i) in rand::seq::sample_iter(&mut rng, 0..train.rows(), n_clusters).unwrap().iter().enumerate() {
            centroids.set_row(i, train.get_row(*row_i))?;
        }

        Ok(centroids)
    }
}

#[test]
fn gen_random_centroids_zero() {
    let train: DMatrix<f32> = DMatrix::new_zeros(3, 3);
    let n_clusters = 2;
    let centroids = KMeans::gen_random_centroids(&train, n_clusters).unwrap();
    for i in 0..centroids.rows() {
        assert_eq!(centroids.get_row(i), [0.0, 0.0, 0.0]);
    }
}

#[test]
fn gen_random_centroids_random() {
    let mut train: DMatrix<f32> = DMatrix::new_zeros(0, 3);
    train.append_row(&[0.5, 1.0, 0.0]);
    train.append_row(&[1.0, 0.5, 0.5]);
    train.append_row(&[0.0, 0.0, 1.0]);

    let n_clusters = 10;
    let centroids = KMeans::gen_random_centroids(&train, n_clusters).unwrap();
    println!("Centriods:\n{}", centroids);
    for i in 0..centroids.rows() {
        for v in centroids.get_row(i) {
            assert!(*v >= 0.0 && *v <= 1.0);
        }
    }
}

#[test]
#[should_panic(expected = "incorrect dimensions")]
fn gen_random_centroids_panic() {
    let train: DMatrix<f32> = DMatrix::new_zeros(0, 3);
    KMeans::gen_random_centroids(&train, 2).unwrap();
}

#[test]
fn kmeans_blobs() {
    let train: DMatrix<f32> = DMatrix::from_csv("data/blobs.csv", 1, ',', Some(&[0, 1])).unwrap();
    let (centroids, ids, inertia) = KMeans::fit(&train, Some(3), None, None).unwrap();
    println!("centroids = \n{}", centroids);
    println!("ids = \n{:?}", ids);
    println!("inertia = \n{}", inertia);
}
