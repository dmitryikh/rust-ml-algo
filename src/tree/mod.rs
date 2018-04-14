mod impurity;

use std::usize;
use std::collections::VecDeque;
use rand::Rng;
use rand;

use matrix::DMatrix;
use utils::isaac_rng;
use self::impurity::{ImpurityGiniUpdater, ImpurityMSEUpdater, ImpurityUpdaterTrait};

#[derive(Debug)]
struct Node {
    /// Правило, по которому выбираем поддерево
    rule: Option<( usize, // Индекс признака, по которому происходит разделение
                   f64,   // Значение признака, по которому происходит разделение
                 )>,
    /// индексы lhs и rhs поддеревьев
    nodes: Option<(usize, usize)>,
    /// Кол-во прецендентов
    size: usize,
    /// Среднее значение предсказания
    avg: f64,
    /// Неоднородность выборки до разбиения
    impurity: f64,
    /// Неоднородность выборки после разбиения
    impurity_after: f64,
}

#[derive(Debug, Clone)]
pub enum SplitCriteria {
    Unset,
    Gini,
    Entropy,
    MSE,
    MAE
}

#[derive(Debug, Clone)]
pub enum SplitFeatures {
    /// Используем все признаки
    Full,
    /// Используем случайные N признаков
    Random(usize),
}

#[derive(Debug, Clone)]
pub struct CartOptions {
    /// Максимальная глубина дерева
    max_depth: usize,
    /// Минимальное кол-во объектов в листе
    min_in_leaf: usize,
    /// Порог неопределенности, ниже которого считаем выборку однородной
    min_impurity: f64,
    /// Критерий разбиения узла
    split_criterion: SplitCriteria,
    /// Опция выбора признаков для разбиения узла
    split_features: SplitFeatures,
    /// Настройка генератора случайных чисел
    pub random_seed: u64,
}

impl CartOptions {
    pub fn new() -> CartOptions {
        CartOptions{ max_depth: usize::MAX,
                     min_in_leaf: 1,
                     min_impurity: 1e-7,
                     split_criterion: SplitCriteria::Unset,
                     split_features: SplitFeatures::Full,
                     random_seed: 0,
                   }
    }

    pub fn max_depth(mut self, max_depth: usize) -> CartOptions {
        self.max_depth = max_depth;
        self
    }

    pub fn min_in_leaf(mut self, min_in_leaf: usize) -> CartOptions {
        self.min_in_leaf = min_in_leaf;
        self
    }

    pub fn min_impurity(mut self, min_impurity: f64) -> CartOptions {
        self.min_impurity = min_impurity;
        self
    }

    pub fn split_criterion(mut self, split_criterion: SplitCriteria) -> CartOptions {
        self.split_criterion = split_criterion;
        self
    }

    pub fn split_features(mut self, split_features: SplitFeatures) -> CartOptions {
        self.split_features = split_features;
        self
    }

    pub fn random_seed(mut self, random_seed: u64) -> CartOptions {
        self.random_seed = random_seed;
        self
    }
}

pub struct ClsTree {
    nodes: Vec<Node>,
    options: CartOptions,
}

pub struct RegTree {
    nodes: Vec<Node>,
    options: CartOptions,
}

impl ClsTree {
    pub fn new(options: CartOptions) -> ClsTree {
        ClsTree{ nodes: Vec::new(), options: options }
    }

    pub fn fit(&mut self, train: &DMatrix<f64>, labels: &[u32]) -> Result<(), String> {
        self.fit_with_ids(train, labels, (0..train.rows()).collect())
    }

    pub fn fit_with_ids(&mut self, train: &DMatrix<f64>, labels: &[u32], ids: Vec<usize>) -> Result<(), String> {
        let impurity_updater = match self.options.split_criterion {
            SplitCriteria::Gini => ImpurityGiniUpdater::new(labels, &ids),
            // SplitCriteria::Entropy => calc_impurity_entropy,
            _ => { return Err(format!("Wrong SplitCriteria: {:?}", self.options.split_criterion)); },
        };
        self.nodes = fit(train, labels, ids, &impurity_updater, &self.options)?;
        Ok(())
    }

    pub fn predict(&self, test: &DMatrix<f64>) -> Result<Vec<u32>, String> {
        let avgs = self.predict_proba(test)?;
        Ok(avgs.iter().map(|&avg| if avg >= 0.5 { 1 } else { 0 }).collect())
    }

    pub fn predict_proba(&self, test: &DMatrix<f64>) -> Result<Vec<f64>, String> {
        predict_avg(&self.nodes, test)
    }
}

impl RegTree {
    pub fn new(options: CartOptions) -> RegTree {
        RegTree{ nodes: Vec::new(), options: options }
    }

    pub fn fit(&mut self, train: &DMatrix<f64>, y: &[f64]) -> Result<(), String> {
        self.fit_with_ids(train, y, (0..train.rows()).collect())
    }

    pub fn fit_with_ids(&mut self, train: &DMatrix<f64>, y: &[f64], ids: Vec<usize>) -> Result<(), String> {
        let impurity_updater = match self.options.split_criterion {
            SplitCriteria::MSE => ImpurityMSEUpdater::new(y, &ids),
            // SplitCriteria::MAE => calc_impurity_mae,
            _ => { return Err(format!("Wrong SplitCriteria: {:?}", self.options.split_criterion)); },
        };
        self.nodes = fit(train, y, ids, &impurity_updater, &self.options)?;
        Ok(())
    }

    pub fn predict(&self, test: &DMatrix<f64>) -> Result<Vec<f64>, String> {
        predict_avg(&self.nodes, test)
    }
}

fn get_impurity_updater<L> ( labels: &[L],
                             ids: &'a mut [usize],
                             options: &CartOptions,
                           ) {
    
}

fn best_split<'a, L, R>( train: &DMatrix<f64>,
                         labels: &[L],
                         ids: &'a mut [usize],
                         impurity_updater: &'a ImpurityUpdaterTrait<'a, Label=L>,
                         parent: &mut Node,
                         options: &CartOptions,
                         rng: &mut R
                       )
    -> Option<(Node, Vec<usize>, Node, Vec<usize>)>
    where
    R: Rng,
{
    let n_samples = ids.len();
    debug_assert!(parent.size == n_samples);
    debug_assert!(parent.impurity == parent.impurity_after);
    debug_assert!(n_samples != 0);

    // если в выборке недостаточно элементов, то всё
    if n_samples <  2 * options.min_in_leaf { return None; }
    // если выборка однородная, то всё
    if parent.impurity < options.min_impurity { return None; }

    let n_dim = train.cols();
    let mut f_vals: Vec<f64> = Vec::with_capacity(n_samples);
    let mut lhs_ids_best = Vec::new();
    let mut rhs_ids_best = Vec::new();
    let mut lhs_impurity_best = 0.0;
    let mut rhs_impurity_best = 0.0;
    let mut lhs_avg_best = 0.0;
    let mut rhs_avg_best = 0.0;
    let mut found_best = false;

    // итерирумеся по номеру фич f_id
    let f_ids = match options.split_features {
        SplitFeatures::Full => { (0..n_dim).collect() },
        SplitFeatures::Random(n) => {
            match rand::seq::sample_iter(rng, 0..n_dim, n) {
                Ok(random_ids) => random_ids,
                Err(full_ids) => full_ids,
            }
        },
    };
    for f_id in f_ids {
        f_vals.clear();
        // сортируем ids по значению признака f_id
        ids.sort_unstable_by(|&a, &b| train.get_val(a, f_id).partial_cmp(&train.get_val(b, f_id)).unwrap());
        ids.iter().for_each(|&i| f_vals.push(train.get_val(i, f_id)));
        // составляем вектор из промежуточных значений
        // f_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let min_f_delta = 1.0e-7;
        // let mut threshold = f_vals[0];
        let mut pos_prev = 0;
        // let thresholds = Some(f_vals[0]).into_iter()
        //                      .chain(  // включаем первый элемент
        //                         f_vals[1..].iter().filter(|&&w|
        //                         {
        //                             if w > last_element {
        //                                 last_element = w;
        //                                 return true;
        //                             } else { return false; }
        //                         }).windows(2).map(|w| (w[1] + w[0]) / 2.0)
        //                      );
        // итерируясь по каждому промежуточному значению составляем lhs и rhs выборки
        // for threshold in thresholds {
        impurity_updater.reset(ids);
        for pos in 1..ids.len() {
            if f_vals[pos] - f_vals[pos_prev] < min_f_delta { continue; }
            let threshold = (f_vals[pos] + f_vals[pos_prev]) * 0.5;
            pos_prev = pos;
            impurity_updater.update(pos);
            // let (lhs_ids, rhs_ids) = split_by_rule(train, ids, f_id, threshold);
            // // выборки не могут быть пустыми, т.к. мы разбиваем thresholds, которые лежат внутри диапазона признака
            // debug_assert!(!lhs_ids.is_empty() && !rhs_ids.is_empty());
            if pos < options.min_in_leaf || (ids.len() - pos) < options.min_in_leaf { continue; }
            let (lhs_impurity, rhs_impurity) = impurity_updater.impurity_sides();
            let q = pos as f64 / n_samples as f64 * lhs_impurity
                  + (ids.len() - pos) as f64 / n_samples as f64 * rhs_impurity;
            if q < parent.impurity_after {
                parent.rule = Some((f_id, threshold));
                parent.impurity_after = q;
                lhs_ids_best = ids[0..pos].to_vec();
                rhs_ids_best = ids[pos..].to_vec();
                lhs_impurity_best = lhs_impurity;
                rhs_impurity_best = rhs_impurity;
                let (lhs_avg, rhs_avg) = impurity_updater.average_sides();
                lhs_avg_best = lhs_avg;
                rhs_avg_best = rhs_avg;
                found_best = true;
            }
        }
    }
    if (!found_best) { return None; }
    debug_assert!(parent.impurity > parent.impurity_after);  // рассчитываем только на улучшение!
    // println!("Best division on feature {} on value {}, \
    //           impurity was {}, impurity is {}",
    //          parent.rule.unwrap().0, parent.rule.unwrap().1, parent.impurity, parent.impurity_after
    //         );
    Some((Node{ rule: None,
                nodes: None,
                size: lhs_ids_best.len(),
                avg: lhs_avg_best,
                impurity: lhs_impurity_best,
                impurity_after: lhs_impurity_best
              },
          lhs_ids_best,
          Node{ rule: None,
                nodes: None,
                size: rhs_ids_best.len(),
                avg: rhs_avg_best,
                impurity: rhs_impurity_best,
                impurity_after: rhs_impurity_best
              },
          rhs_ids_best,
        ))
}

fn build_root<'a, L>(train: &DMatrix<f64>, labels: &[L], ids: &'a [usize], impurity_updater: &'a ImpurityUpdaterTrait<'a, Label=L>) -> Node
{
    impurity_updater.reset(ids);
    let impurity = impurity_updater.impurity();
    let avg = impurity_updater.average();
    let node = Node{ rule: None, nodes: None, size: ids.len(), avg: avg, impurity: impurity, impurity_after: impurity };
    node
}

// fn split_by_rule(train: &DMatrix<f64>, ids: &[usize], f_id: usize, threshold: f64) -> (Vec<usize>, Vec<usize>) {
//     let mut lhs_ids = Vec::new();
//     let mut rhs_ids = Vec::new();
//     for &i in ids {
//         if train.get_val(i, f_id) <= threshold {
//             lhs_ids.push(i);
//         } else {
//             rhs_ids.push(i);
//         }
//     }
//     (lhs_ids, rhs_ids)
// }
// 
fn fit<L>(train: &DMatrix<f64>, labels: &[L], ids: Vec<usize>, impurity_updater: &ImpurityUpdaterTrait<Label=L>, options: &CartOptions)
    -> Result<Vec<Node>, String>
{
    let n_samples = ids.len();
    if n_samples == 0 { return Err("set is empty".to_string()); }
    if options.max_depth == 0 {
        return Err(format!("max_depth should be >= 1, got {}", options.max_depth));
    }
    if train.cols() == 0 {
        return Err("No features in train (0 columns)".to_string());
    }
    let mut rng = isaac_rng(options.random_seed);

    let mut nodes = Vec::new();
    let mut next_nodes: VecDeque<(usize /*n_id*/, Vec<usize> /*ids*/)> = VecDeque::new();
    let mut depth = 2;
    let node = build_root(train, labels, &ids, impurity_updater);
    nodes.push(node);
    next_nodes.push_back((nodes.len() - 1, ids));

    while depth <= options.max_depth && !next_nodes.is_empty(){
        let mut new_next_nodes: VecDeque<(usize, Vec<usize>)> = VecDeque::new();
        for (node_id, mut ids) in next_nodes.drain(..) {
            if let Some((lhs_node, lhs_ids, rhs_node, rhs_ids)) =
                   best_split(train, labels, &mut ids, impurity_updater, &mut nodes[node_id], options, &mut rng)
            {
                    let lhs_node_id = nodes.len();
                    nodes.push(lhs_node);
                    new_next_nodes.push_back((lhs_node_id, lhs_ids));
                    let rhs_node_id = nodes.len();
                    nodes.push(rhs_node);
                    new_next_nodes.push_back((rhs_node_id, rhs_ids));
                    nodes[node_id].nodes = Some((lhs_node_id, rhs_node_id));
            }
        }
        next_nodes = new_next_nodes;
        depth += 1;
    }
    Ok(nodes)
}

fn predict_avg(nodes: &Vec<Node>, test: &DMatrix<f64>) -> Result<Vec<f64>, String>
{
    if nodes.is_empty() { return Err("No tree build".to_string()); }
    let mut labels = Vec::with_capacity(test.rows());
    for i in 0..test.rows() {
        let mut n_id = 0;
        while let Some((f_id, threshold)) = nodes[n_id].rule {
            if let Some((lhs_id, rhs_id)) = nodes[n_id].nodes {
                n_id = if test.get_val(i, f_id) <= threshold { lhs_id } else { rhs_id };
            } else {
                return Err("Inconsistent tree".to_string());
            }
        }
        labels.push(nodes[n_id].avg);
    }
    Ok(labels)
}

#[cfg(test)]
mod test {
    use super::*;
    use utils::{accuracy, accuracy_perm, write_csv_col, rmse_error, mae_error};

    #[test]
    fn blobs() {
        let train: DMatrix<f64> = DMatrix::from_csv("data/blobs.csv", 1, ',', Some(&[0, 1])).unwrap();
        let labels: DMatrix<u32> = DMatrix::from_csv("data/blobs.csv", 1, ',', Some(&[2])).unwrap();
        let labels_2: Vec<u32> = labels.data().iter().map(|&id| if id > 0 { 1 } else { 0 }).collect();
        let mut cart = ClsTree::new( CartOptions::new()
                                         .max_depth(10)
                                         .split_criterion(SplitCriteria::Gini)
                                   );
        cart.fit(&train, &labels_2).unwrap();
        cart.predict(&train).unwrap();
        let p_labels = cart.predict(&train).unwrap();
        let accuracy = accuracy(&labels_2, &p_labels);
        println!("accuracy = {}", accuracy);
        assert_eq!(accuracy, 1.0);
    }

    #[test]
    fn mouse() {
        let train: DMatrix<f64> = DMatrix::from_csv("data/mouse.csv", 40, ',', Some(&[0, 1])).unwrap();
        let labels: DMatrix<u32> = DMatrix::from_csv("data/mouse.csv", 40, ',', Some(&[2])).unwrap();
        let labels_2: Vec<u32> = labels.data().iter().map(|&id| if id > 0 { 1 } else { 0 }).collect();
        let mut cart = ClsTree::new( CartOptions::new()
                                         .max_depth(10)
                                         .split_criterion(SplitCriteria::Entropy)
                                   );
        cart.fit(&train, &labels_2).unwrap();
        let p_labels = cart.predict(&train).unwrap();
        let accuracy = accuracy(&labels_2, &p_labels);
        println!("accuracy = {}", accuracy);
        assert_eq!(accuracy, 1.0);
    }

    #[test]
    #[should_panic(expected = "set is empty")]
    fn empty_train() {
        ClsTree::new( CartOptions::new()
                         .split_criterion(SplitCriteria::Gini)
                    ).fit(&DMatrix::new_zeros(0, 2), &[]).unwrap();
    }

    #[test]
    #[should_panic(expected = "No features in train")]
    fn zero_dim() {
        ClsTree::new( CartOptions::new()
                         .split_criterion(SplitCriteria::Gini)
                    ).fit(&DMatrix::new_zeros(2, 0), &[0, 0]).unwrap();
    }

    #[test]
    fn one_point() {
        let mut cart = ClsTree::new( CartOptions::new()
                                         .split_criterion(SplitCriteria::Entropy)
                                   );
        cart.fit(&DMatrix::new_zeros(1, 2), &[1]).unwrap();
        let mut test: DMatrix<f64> = DMatrix::new_zeros(0, 2);
        test.append_row(&[1.0, 0.5]);
        test.append_row(&[0.0, 0.0]);
        let labels = cart.predict(&test).unwrap();
        assert!(cart.nodes.len() == 1);
        assert_eq!(cart.nodes[0].rule, None);
        assert_eq!(cart.nodes[0].nodes, None);
        assert_eq!(cart.nodes[0].size, 1);
        assert_eq!(cart.nodes[0].avg, 1.0);
        assert_eq!(cart.nodes[0].impurity, 0.0);
        assert_eq!(cart.nodes[0].impurity_after, 0.0);
        assert_eq!(labels, [1, 1]);
    }

    #[test]
    fn sin() {
        let mut cart = RegTree::new( CartOptions::new()
                                         .max_depth(4)
                                         .split_criterion(SplitCriteria::MSE)
                                   );
        let train_x: DMatrix<f64> = DMatrix::from_csv("data/sin.csv", 1, ',', Some(&[0])).unwrap();
        let train_y: DMatrix<f64> = DMatrix::from_csv("data/sin.csv", 1, ',', Some(&[1])).unwrap();
        cart.fit(&train_x, train_y.data()).unwrap();
        // println!("Nodes: {:?}", cart.nodes);
        let pred_y = cart.predict(&train_x).unwrap();
        // write_csv_col("output/sin.csv", &pred_y, None).unwrap();
        let rmse = rmse_error(train_y.data(), &pred_y);
        let mae = mae_error(train_y.data(), &pred_y);
        println!("RMSE = {}, MAE = {}", rmse, mae);
    }
}
