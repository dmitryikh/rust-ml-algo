use rand::Rng;
use rand::distributions::Range;

use cart::{ClsTree, RegTree, CartOptions};
use matrix::DMatrix;
use utils::isaac_rng;

pub struct RandomForestOptions {
    /// Кол-во деревьев в ансамбле
    n_trees: usize,
    /// Опции построения дерева
    tree_options: CartOptions,
    /// Проводить bootstrap при генерации выборки для дерева
    bootstrap: bool,
    /// Проводить оценку точности на out of bag выборке
    out_of_bag: bool,
    /// Кол-во поток для обучения
    n_jobs: usize,
}

impl RandomForestOptions {
    pub fn new() -> RandomForestOptions {
        RandomForestOptions{ n_trees: 10,
                             tree_options: CartOptions::new(),
                             bootstrap: true,
                             out_of_bag: false,
                             n_jobs: 1,
                           }
    }

    pub fn n_trees(mut self, n_trees: usize) -> RandomForestOptions {
        self.n_trees = n_trees;
        self
    }

    pub fn tree_options(mut self, tree_options: CartOptions) -> RandomForestOptions {
        self.tree_options = tree_options;
        self
    }

    pub fn bootstrap(mut self, bootstrap: bool) -> RandomForestOptions {
        self.bootstrap = bootstrap;
        self
    }

    pub fn out_of_bag(mut self, out_of_bag: bool) -> RandomForestOptions {
        self.out_of_bag = out_of_bag;
        self
    }

    pub fn n_jobs(mut self, n_jobs: usize) -> RandomForestOptions {
        self.n_jobs = n_jobs;
        self
    }
}


pub struct ClsRandomForest {
    trees: Vec<ClsTree>,
    options: RandomForestOptions,
}


pub struct RegRandomForest {
    trees: Vec<RegTree>,
    options: RandomForestOptions,
}

impl ClsRandomForest {
    pub fn new(options: RandomForestOptions) -> ClsRandomForest {
        ClsRandomForest{ trees: Vec::new(), options: options }
    }

    pub fn fit(&mut self, train: &DMatrix<f64>, labels: &[u32]) -> Result<(), String> {
        self.trees.clear();
        let n_trees = self.options.n_trees;
        if n_trees == 0 { return Err("n_trees should be > 0".to_string()); }

        let mut rng = isaac_rng(self.options.tree_options.random_seed);
        // Заранее генерируем сиды для каждого дерева, чтобы в параллельной реализации была
        // детерменированность в случае фиксированого сида в опциях
        let tree_seeds: Vec<_> = (0..n_trees).map(|_| rng.next_u64()).collect();
        for tree_i in 0..self.options.n_trees {
            let ids = if self.options.bootstrap {
                // Почему то не работает
                // rng.sample_iter(&Range::new(0, train.rows())).take(train.rows()).collect()
                (0..train.rows()).map(|_| rng.gen_range(0, train.rows())).collect()
            } else {
                (0..train.rows()).collect()
            };
            let mut tree = ClsTree::new(self.options.tree_options.clone().random_seed(tree_seeds[tree_i]));
            tree.fit_with_ids(train, labels, ids)?;
            self.trees.push(tree);
        }
        Ok(())
    }

    pub fn predict_proba(&self, test: &DMatrix<f64>) -> Result<Vec<f64>, String> {
        if self.trees.is_empty() { return Err("No trees built".to_string()); }
        let n_trees = self.options.n_trees;

        let mut probs = vec![0.0; test.rows()];
        for tree in &self.trees {
            tree.predict_proba(test)?
                .iter().zip(probs.iter_mut())
                .for_each(|(p, p_sum)| *p_sum += *p / self.trees.len() as f64);
        }
        Ok(probs)
    }

    pub fn predict(&self, test: &DMatrix<f64>) -> Result<Vec<u32>, String> {
        let avgs = self.predict_proba(test)?;
        Ok(avgs.iter().map(|&avg| if avg >= 0.5 { 1 } else { 0 }).collect())
    }
}


impl RegRandomForest {
    pub fn new(options: RandomForestOptions) -> RegRandomForest {
        RegRandomForest{ trees: Vec::new(), options: options }
    }

    pub fn fit(&mut self, train: &DMatrix<f64>, labels: &[f64]) -> Result<(), String> {
        self.trees.clear();
        let n_trees = self.options.n_trees;
        if n_trees == 0 { return Err("n_trees should be > 0".to_string()); }

        let mut rng = isaac_rng(self.options.tree_options.random_seed);
        // Заранее генерируем сиды для каждого дерева, чтобы в параллельной реализации была
        // детерменированность в случае фиксированого сида в опциях
        let tree_seeds: Vec<_> = (0..n_trees).map(|_| rng.next_u64()).collect();
        for tree_i in 0..self.options.n_trees {
            let ids = if self.options.bootstrap {
                // Почему то не работает
                // rng.sample_iter(&Range::new(0, train.rows())).take(train.rows()).collect()
                (0..train.rows()).map(|_| rng.gen_range(0, train.rows())).collect()
            } else {
                (0..train.rows()).collect()
            };
            let mut tree = RegTree::new(self.options.tree_options.clone().random_seed(tree_seeds[tree_i]));
            tree.fit_with_ids(train, labels, ids)?;
            self.trees.push(tree);
        }
        Ok(())
    }

    pub fn predict(&self, test: &DMatrix<f64>) -> Result<Vec<f64>, String> {
        if self.trees.is_empty() { return Err("No trees built".to_string()); }
        let n_trees = self.options.n_trees;

        let mut avgs = vec![0.0; test.rows()];
        for tree in &self.trees {
            tree.predict(test)?
                .iter().zip(avgs.iter_mut())
                .for_each(|(p, p_sum)| *p_sum += *p / self.trees.len() as f64);
        }
        Ok(avgs)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use utils::{accuracy, accuracy_perm, write_csv_col, rmse_error, mae_error};
    use cart::SplitCriteria;

    #[test]
    fn sin() {
        let mut forest = RegRandomForest::new( RandomForestOptions::new()
                                                  .n_trees(10)
                                                  .bootstrap(false)
                                                  .tree_options( CartOptions::new()
                                                                     .max_depth(100)
                                                                     .min_in_leaf(1)
                                                                     .random_seed(42)
                                                                     .split_criterion(SplitCriteria::MSE)
                                                               )
                                             );
        let train_x: DMatrix<f64> = DMatrix::from_csv("data/sin.csv", 1, ',', Some(&[0])).unwrap();
        let train_y: DMatrix<f64> = DMatrix::from_csv("data/sin.csv", 1, ',', Some(&[1])).unwrap();
        forest.fit(&train_x, train_y.data()).unwrap();
        let pred_y = forest.predict(&train_x).unwrap();
        // write_csv_col("output/forest_sin.csv", &pred_y, None).unwrap();
        let rmse = rmse_error(train_y.data(), &pred_y);
        let mae = mae_error(train_y.data(), &pred_y);
        println!("RMSE = {}, MAE = {}", rmse, mae);
    }
}
