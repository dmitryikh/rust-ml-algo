use kdtree::KdTree;
use kdtree::distance::squared_euclidean;

use matrix::DMatrix;

pub struct DbscanOption {
    eps: f64,
    min_samples: u32,
}

impl DbscanOption {
    pub fn new() -> DbscanOption {
        DbscanOption{ eps: 0.5, min_samples: 5 }
    }

    pub fn eps(mut self, eps: f64) -> DbscanOption {
        debug_assert!(eps > 0.0);
        self.eps = eps;
        self
    }

    pub fn min_samples(mut self, min_samples: u32) -> DbscanOption {
        debug_assert!(min_samples > 0);
        self.min_samples = min_samples;
        self
    }
}

#[derive(Clone, Debug)]
pub enum PointLabel {
    Unknown,
    Core(usize),
    Border(usize),
    Noise,
}

pub struct Dbscan {
    labels: Vec<PointLabel>,
    options: DbscanOption,
}

impl Dbscan {
    pub fn new(options: DbscanOption) -> Dbscan {
        Dbscan{ labels: Vec::new(), options }
    }

    pub fn fit(&mut self, train: &DMatrix<f64>) -> Result<(), String> {
        let n_dim = train.cols();
        let n_points = train.rows();
        if n_dim == 0 { return Err("Dimension is zero".to_string()); }
        if n_points == 0 { return Err("no points".to_string()); }

        let mut neighbors: Vec<Vec<usize>> = Vec::with_capacity(n_points);
        self.labels = vec![PointLabel::Unknown; n_points];
        // Инициализируем kdtree точками выборки
        let mut kdtree = KdTree::new(n_dim);
        for i in 0..n_points {
            kdtree.add(train.get_row(i), i).map_err(|e| format!("Can't add point {}: {:?}", i, e))?;
        }

        // Ищем `eps` соседей для всех точек, помечаем `Core` точки
        for i in 0..n_points {
            let locals = kdtree.within(train.get_row(i), self.options.eps.powi(2), &squared_euclidean)
                               .map_err(|e| format!("Can't find neighbors for point {}: {:?}", i, e))?;
            neighbors.push(locals.iter().map(|v| *v.1).collect());
            if neighbors[i].len() >= self.options.min_samples as usize {
                self.labels[i] = PointLabel::Core(0);
            }
        }

        // Расставляем номера компонент для `Core`
        self._label_core_components(&neighbors);

        // Размечаем остальные точки на `Border` и `Noise`
        for i in 0..n_points {
            if let PointLabel::Core(_) = self.labels[i] { continue; }
            let mut cur_id = None;
            for id in &neighbors[i] {
                if let PointLabel::Core(id) = self.labels[*id] {
                    cur_id = Some(id);
                    break;
                }
            }
            self.labels[i] = match cur_id {
                Some(id) => PointLabel::Border(id),
                None =>  PointLabel::Noise,
            };
        }
        Ok(())
    }

    pub fn labels(&self) -> Vec<i32> {
        let mut ilabels: Vec<i32> = Vec::with_capacity(self.labels.len());
        for label in &self.labels {
            ilabels.push( match label {
                &PointLabel::Noise => -1,
                &PointLabel::Border(id) => id as i32,
                &PointLabel::Core(id) => id as i32,
                &PointLabel::Unknown => { panic!("Why i'm here!"); },
            });
        }
        ilabels
    }

    fn _label_core_components(&mut self, neighbors: &Vec<Vec<usize>>) {
        // Используем Depth-first search с простановкой меток компонент для `Core` точек
        let n_points = neighbors.len();
        let mut visited = vec![false; n_points];
        let mut visit_stack: Vec<(usize /*точка*/, usize /*ребро*/)> = Vec::new();
        let mut cur_id = 0;

        for i in 0..n_points {
            if visited[i] { continue; }
            if let PointLabel::Core(ref mut id) = self.labels[i] {
                *id = cur_id;
            } else { continue; }
            visit_stack.push((i, 0));
            visited[i] = true;
            while let Some((v, e)) = visit_stack.pop() {
                if e + 1 < neighbors[v].len() {
                    visit_stack.push((v, e + 1));
                } else {
                    continue;
                }
                let w = neighbors[v][e];
                if visited[w] { continue; }
                if let PointLabel::Core(ref mut id) = self.labels[w] {
                    *id = cur_id;
                    visit_stack.push((w, 0));
                    visited[w] = true;
                }
            }
            cur_id += 1;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use utils::write_csv_col;

    #[test]
    fn kdtree() {
        let mut kdtree = KdTree::new(2);
        kdtree.add(&[0.0, 0.0], 0).unwrap();
        kdtree.add(&[0.1, 0.0], 1).unwrap();
        kdtree.add(&[0.1, 0.1], 2).unwrap();
        kdtree.add(&[0.0, 0.1], 3).unwrap();
        let res = kdtree.nearest(&[0.0, 0.01], 2, &squared_euclidean).unwrap();
        println!("res = {:?}", res);
        for (_, id) in res {
            assert!(*id == 0 || *id == 3);
        }

        let res = kdtree.within(&[0.0, 0.0], 0.01001, &squared_euclidean).unwrap();
        println!("res = {:?}", res);
        for (_, id) in res {
            assert!(*id == 0 || *id == 3 || *id == 1);
        }
    }

    #[test]
    fn mouse() {
        let train: DMatrix<f64> = DMatrix::from_csv("data/mouse.csv", 40, ',', Some(&[0, 1])).unwrap();
        let labels: DMatrix<u32> = DMatrix::from_csv("data/mouse.csv", 40, ',', Some(&[2])).unwrap();
        let mut dbs = Dbscan::new( DbscanOption::new()
                                     .eps(0.03)
                                     .min_samples(3)
                                 );
        dbs.fit(&train).unwrap();
        // println!("{:?}", dbs.labels);
        // write_csv_col("output/mouse_dbs.csv", &dbs.labels(), None).unwrap();
    }

    #[test]
    fn blobs() {
        let train: DMatrix<f64> = DMatrix::from_csv("data/blobs.csv", 1, ',', Some(&[0, 1])).unwrap();
        let labels: DMatrix<u32> = DMatrix::from_csv("data/blobs.csv", 1, ',', Some(&[2])).unwrap();
        let mut dbs = Dbscan::new( DbscanOption::new()
                                     .eps(0.3)
                                     .min_samples(3)
                                 );
        dbs.fit(&train).unwrap();
        // println!("{:?}", dbs.labels);
        // write_csv_col("output/blobs_dbs.csv", &dbs.labels(), None).unwrap();
    }
}
