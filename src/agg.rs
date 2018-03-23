use matrix::DMatrix;

#[derive(Debug)]
struct AggCluster {
    left: usize,
    right: usize,
    dist: f64,
    n_points: usize,
}

#[derive(Debug, Copy, Clone)]
pub enum Linkage {
    Min,
    Max,
    Average,
    Center,
    Ward,
}

pub struct AgglomerativeOptions {
    linkage: Linkage
}

impl AgglomerativeOptions {
    pub fn new() -> AgglomerativeOptions {
        AgglomerativeOptions{ linkage: Linkage::Average }
    }

    pub fn linkage(mut self, linkage: Linkage) -> AgglomerativeOptions {
        self.linkage = linkage;
        self
    }
}

pub struct Agglomerative {
    clusters: Vec<AggCluster>,
    options: AgglomerativeOptions,
}

impl Agglomerative {
    pub fn new(options: AgglomerativeOptions) -> Agglomerative {
        Agglomerative{ clusters: Vec::new(), options: options }
    }

    pub fn fit(&mut self, train: &DMatrix<f64>) -> Result<(), String> {
        let n_points = train.rows();
        // TODO: n_points == 0
        self.clusters = Vec::with_capacity(n_points - 1);
        let mut is_top: Vec<bool> = Vec::with_capacity(2 * n_points - 1);
        for _ in 0..n_points {
            is_top.push(true);
        }
        if n_points == 1 { return Ok(()); }
        // верхний треугольник матрицы без диагонали
        let mut distances: Vec<f64> = Vec::with_capacity(n_points * (n_points - 1) / 2);
        let dist_index_f = |i, j| {
            // Нижняя треугольная матрица, например 4 x 4
            // * * * *
            // 0 * * *
            // 1 2 * *
            // 3 4 5 *
            // Всего элементов 4 * (4 - 1) /2 = 6
            let (i, j) = if i < j { (j, i) } else { (i, j) };
            debug_assert!(j < i);
            (i-1) * i / 2 + j
        };
        let dist_f = |i, j| train.get_row(i).iter()
                                     .zip(train.get_row(j).iter())
                                     .fold(0.0, |s, v| s + (*v.0 - *v.1).powi(2)).sqrt();
        // считаем исходные расстояния
        for i in 1..n_points {
            for j in 0..i {
                distances.push(dist_f(i, j));
            }
        }

        for it in 0..(n_points - 1) {
            // ищем минимальное расстояние
            let mut pair_min = (0, 0);  // пустая инициализация
            let mut dist_min = 0.0;
            let mut is_first = true;
            for i in 1..(n_points + it) {
                if !is_top[i] { continue; }
                for j in 0..i {
                    if !is_top[j] { continue; }
                    let dist = distances[dist_index_f(i, j)];
                    pair_min = if is_first {
                        is_first = false;
                        dist_min = dist;
                        (i, j)
                    } else {
                        if dist < dist_min  {
                            dist_min = dist;
                            (i, j)
                        } else {
                            pair_min
                        }
                    };
                }
            }
            debug_assert!(pair_min != (0, 0));

            // объединяем кластера
            let u = if (pair_min.0 < n_points) { 1 } else { self.clusters[pair_min.0 - n_points].n_points };
            let v = if (pair_min.1 < n_points) { 1 } else { self.clusters[pair_min.1 - n_points].n_points };
            self.clusters.push(AggCluster{ left: pair_min.0,
                                           right: pair_min.1,
                                           dist: dist_min,
                                           n_points: u + v,
                                         }
                              );
            is_top[pair_min.0] = false;
            is_top[pair_min.1] = false;
            is_top.push(true);

            let u = u as f64;
            let v = v as f64;
            let w = u + v;
            let ruv = distances[dist_index_f(pair_min.0, pair_min.1)];

            // считаем расстояния от нового кластера до всех остальных
            let i = n_points + it;
            for j in 0..i {
                if is_top[j] {
                    let (au, av, b, g) = match self.options.linkage {
                        Linkage::Min => (0.5, 0.5, 0.0, -0.5),
                        Linkage::Max => (0.5, 0.5, 0.0, 0.5),
                        Linkage::Average => (u / w, v / w, 0.0, 0.0),
                        Linkage::Center => {
                            let au = u / w;
                            let av = v / w;
                            (au, av, -au * av, 0.0)
                        }
                        Linkage::Ward => {
                            let s = if (j < n_points) { 1 } else { self.clusters[j - n_points].n_points };
                            let s = s as f64;
                            ((s + u) / (s + w), (s + v) / (s + w), -s / (s + w), 0.0)
                        }
                    };
                    let rus = distances[dist_index_f(pair_min.0, j)];
                    let rvs = distances[dist_index_f(pair_min.1, j)];
                    let distance = au * rus + av * rvs + b * ruv + g * (rus - rvs).abs();
                    distances.push(distance);
                } else {
                    distances.push(0.0);
                }
            }
        }
        Ok(())
    }

    pub fn predict(&self, n_clusters: u32) -> Result<Vec<u32>, String> {
        if self.clusters.is_empty() { return Err("clusters are empty".to_string()); }
        // Исходим из того, что было построено полное дерево
        let n_clusters = n_clusters as usize;
        let n_points = self.clusters.len() + 1;
        let top = 2 * n_points - 1 - (n_clusters - 1);
        let mut labels: Vec<u32> = vec![0; n_points];

        // Ищем корни кластеров
        let mut roots: Vec<usize> = Vec::with_capacity(n_clusters);
        for i in top..(2 * n_points - 1) {
            let cluster = &self.clusters[i - n_points];
            if cluster.left < top {
                roots.push(cluster.left);
            }
            if cluster.right < top {
                roots.push(cluster.right);
            }
        }
        debug_assert!(roots.len() == n_clusters);

        // раскручиваем корни кластеров до конкретных точек, расставляем метки
        let mut nodes: Vec<usize> = Vec::new();
        for (label, root) in roots.iter().enumerate() {
            nodes.clear();
            nodes.push(*root);
            while !nodes.is_empty() {
                let node  = nodes.pop().unwrap();
                if node < n_points {
                    labels[node] = label as u32;
                } else {
                    let cluster = &self.clusters[node - n_points];
                    nodes.push(cluster.left);
                    nodes.push(cluster.right);
                }
            }
        }
        Ok(labels)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use utils::{accuracy_perm, write_csv_col};

    #[test]
    fn blobs() {
        let train: DMatrix<f64> = DMatrix::from_csv("data/blobs.csv", 1, ',', Some(&[0, 1])).unwrap();
        let labels: DMatrix<u32> = DMatrix::from_csv("data/blobs.csv", 1, ',', Some(&[2])).unwrap();
        for linkage in &[Linkage::Max, Linkage::Average, Linkage::Center, Linkage::Ward] {
            let mut agg = Agglomerative::new( AgglomerativeOptions::new()
                                                  .linkage(*linkage)
                                            );
            agg.fit(&train).unwrap();
            let p_labels = agg.predict(3).unwrap();
            let accuracy = accuracy_perm(labels.data(), &p_labels, &[0, 1, 2]);
            println!("Linkage = {:?}, accuracy_perm = {}", linkage, accuracy);
            // write_csv_col("output/blobs_agg.csv", &p_labels, None).unwrap();
            assert_eq!(accuracy, 1.0);
        }
    }

    #[test]
    fn mouse() {
        let train: DMatrix<f64> = DMatrix::from_csv("data/mouse.csv", 40, ',', Some(&[0, 1])).unwrap();
        let labels: DMatrix<u32> = DMatrix::from_csv("data/mouse.csv", 40, ',', Some(&[2])).unwrap();
        for linkage in &[Linkage::Max, Linkage::Average, Linkage::Center, Linkage::Ward] {
            let mut agg = Agglomerative::new( AgglomerativeOptions::new()
                                                  .linkage(*linkage)
                                            );
            agg.fit(&train).unwrap();
            let p_labels = agg.predict(3).unwrap();
            let accuracy = accuracy_perm(labels.data(), &p_labels, &[0, 1, 2]);
            println!("Linkage = {:?}, accuracy_perm = {}", linkage, accuracy);
            // write_csv_col("output/blobs_agg.csv", &p_labels, None).unwrap();
            let lower_bound = match *linkage {
                Linkage::Max => 0.66,
                Linkage::Average => 0.88,
                Linkage::Center => 0.95,
                Linkage::Ward => 0.94,
                _ => {panic!("Don't know result");}
            };
            assert!(accuracy > lower_bound);
        }
    }
}
