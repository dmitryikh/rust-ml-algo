
pub fn entropy(p_vec: &[f64]) -> f64 {
    debug_assert!(p_vec.iter().any(|&p| p >= 0.0 && p <= 1.0));
    p_vec.iter().fold(0.0, |s, &p| s - if p != 0.0 { p * p.log2() } else { 0.0 })
}

pub fn entropy_bin(p: f64) -> f64 {
    entropy(&[p, 1.0 - p])
}

pub fn gini(p_vec: &[f64]) -> f64 {
    debug_assert!(p_vec.iter().any(|&p| p >= 0.0 && p <= 1.0));
    p_vec.iter().fold(1.0, |s, &p| s - p.powi(2))
}

pub fn gini_bin(p: f64) -> f64 {
    debug_assert!(p >= 0.0 && p <= 1.0);
    2.0 * p * (1.0 - p)
}

pub fn calc_impurity_gini(labels: &[u32], ids: &[usize], pos: usize) -> (f64 /*lhs_avg*/, f64 /*lhs_impurity*/,
                                                         f64 /*rhs_avg*/, f64 /*rhs_impurity*/) {
    debug_assert!(pos > 0 && pos < ids.len());

    let lhs_avg = ids[0..pos].iter().fold(0.0, |sum, &i| sum + labels[i] as f64) / pos as f64;
    let rhs_avg = ids[pos..].iter().fold(0.0, |sum, &i| sum + labels[i] as f64) / (ids.len() - pos) as f64;
    (lhs_avg, gini_bin(lhs_avg), rhs_avg, gini_bin(rhs_avg))
}

fn calc_impurity_entropy(labels: &[u32], ids: &[usize]) -> (f64 /*avg*/, f64 /*impurity*/) {
    if ids.is_empty() {
        (0.0, 0.0)
    } else {
        let avg = ids.iter().fold(0.0, |sum, &i| sum + labels[i] as f64) / ids.len() as f64;
        (avg, entropy_bin(avg))
    }
}

pub fn calc_impurity_mse(y: &[f64], ids: &[usize], pos: usize) -> (f64 /*lhs_avg*/, f64 /*lhs_impurity*/,
                                                               f64 /*rhs_avg*/, f64 /*rhs_impurity*/) {
    debug_assert!(pos < ids.len());
    let (lhs_avg, lhs_mse) = if pos == 0 {
        (0.0, 0.0)
    } else {
        let lhs_avg = ids[0..pos].iter().fold(0.0, |sum, &i| sum + y[i]) / pos as f64;
        let lhs_mse = ids[0..pos].iter().fold(0.0, |sum, &i| sum + (y[i] - lhs_avg).powi(2)) / pos as f64;
        (lhs_avg, lhs_mse)
    };


    let rhs_avg = ids[pos..].iter().fold(0.0, |sum, &i| sum + y[i]) / (ids.len() - pos) as f64;
    let rhs_mse = ids[pos..].iter().fold(0.0, |sum, &i| sum + (y[i] - rhs_avg).powi(2)) / (ids.len() - pos) as f64;

    (lhs_avg, lhs_mse, rhs_avg, rhs_mse)
}

struct AverageUpdater {
    lhs_sum: f64,
    sum: f64,
    pos: usize,
    len: usize,
}

impl AverageUpdater
{
    fn new<L>(labels: &[L], ids: &[usize]) -> Self
        where f64: From<L>,
        L: Clone,
    {
        let mut s = AverageUpdater { lhs_sum: 0.0,
                                     sum: 0.0,
                                     pos: 0,
                                     len: ids.len(),
                                   };
        s.reset(labels, ids);
        s
    }

    fn reset<L>(&mut self, labels: &[L], ids: &[usize])
        where f64: From<L>,
        L: Clone,
    {
        self.lhs_sum = 0.0;
        self.sum = ids.iter().fold(0.0, |sum, &i| sum + f64::from(labels[i].clone()));
        self.pos = 0;
        self.len = ids.len();
    }

    fn update<L>(&mut self, new_pos: usize, labels: &[L], ids: &[usize])
        where f64: From<L>,
        L: Clone,
    {
        self.lhs_sum += ids[self.pos..new_pos].iter()
                            .fold(0.0, |sum, &i| sum + f64::from(labels[i].clone()));
        self.pos = new_pos;
    }

    fn average(&self) -> f64 {
        self.sum / self.len as f64
    }

    fn average_sides(&self) -> (f64, f64) {
        let lhs_avg = if self.pos == 0 {
                          0.0
                      } else {
                          self.lhs_sum / self.pos as f64
                      };
        let rhs_avg = (self.sum - self.lhs_sum) / (self.len - self.pos) as f64;
        (lhs_avg, rhs_avg)
    }
}

pub trait ImpurityUpdaterTrait {
    type Label;

    fn reset(&mut self, labels: &[Self::Label], ids: &[usize]);
    fn update(&mut self, new_pos: usize, labels: &[Self::Label], ids: &[usize]);
    fn impurity(&self) -> f64;
    fn average(&self) -> f64;
    fn impurity_sides(&self) -> (f64, f64);
    fn average_sides(&self) -> (f64, f64);
}

pub struct ImpurityMSEUpdater {
    avg_updater: AverageUpdater,
    lhs_sum_sq: f64,
    sum_sq: f64,
}

impl ImpurityMSEUpdater {
    pub fn new(labels: &[f64], ids: &[usize]) -> Self {
        let mut s = ImpurityMSEUpdater{ avg_updater: AverageUpdater::new(labels, ids),
                                        lhs_sum_sq: 0.0,
                                        sum_sq: 0.0,
                                      };
        s.reset(labels, ids);
        s
    }
}

impl ImpurityUpdaterTrait for ImpurityMSEUpdater {
    type Label = f64;

    fn reset(&mut self, labels: &[f64], ids: &[usize]) {
        self.avg_updater.reset(labels, ids);
        self.lhs_sum_sq = 0.0;
        self.sum_sq = ids.iter().fold(0.0, |sq_sum, &i| sq_sum + labels[i].powi(2));
    }

    fn update(&mut self, new_pos: usize, labels: &[f64], ids: &[usize]) {
        let pos = self.avg_updater.pos;
        self.lhs_sum_sq += ids[pos..new_pos].iter()
                            .fold(0.0, |sq_sum, &i| sq_sum + labels[i].powi(2));
        self.avg_updater.update(new_pos, labels, ids);
    }

    fn impurity(&self) -> f64 {
        self.sum_sq / self.avg_updater.len as f64 - self.avg_updater.average().powi(2)
    }

    fn average(&self) -> f64 {
        self.avg_updater.average()
    }

    fn impurity_sides(&self) -> (f64, f64) {
        let (lhs_avg, rhs_avg) = self.avg_updater.average_sides();
        let pos = self.avg_updater.pos;
        let lhs_mse = if pos == 0 {
                          0.0
                      } else {
                          self.lhs_sum_sq / pos as f64 - lhs_avg.powi(2)
                      };
        let rhs_mse = (self.sum_sq - self.lhs_sum_sq) / (self.avg_updater.len - pos) as f64 - rhs_avg.powi(2);
        (lhs_mse, rhs_mse)
    }

    fn average_sides(&self) -> (f64, f64) {
        self.avg_updater.average_sides()
    }
}

pub struct ImpurityGiniUpdater {
    avg_updater: AverageUpdater,
}

impl ImpurityGiniUpdater {
    pub fn new(labels: &[u32], ids: &[usize]) -> Self {
        let mut s = ImpurityGiniUpdater{ avg_updater: AverageUpdater::new(labels, ids) };
        s.reset(labels, ids);
        s
    }
}

impl ImpurityUpdaterTrait for ImpurityGiniUpdater {
    type Label = u32;

    fn reset(&mut self, labels: &[u32], ids: &[usize]) {
        self.avg_updater.reset(labels, ids);
    }

    fn update(&mut self, new_pos: usize, labels: &[u32], ids: &[usize]) {
        self.avg_updater.update(new_pos, labels, ids);
    }

    fn impurity(&self) -> f64 {
        let avg = self.avg_updater.average();
        gini_bin(avg)
    }

    fn average(&self) -> f64 {
        self.avg_updater.average()
    }

    fn impurity_sides(&self) -> (f64, f64) {
        let (lhs_avg, rhs_avg) = self.avg_updater.average_sides();
        (gini_bin(lhs_avg), gini_bin(rhs_avg))
    }

    fn average_sides(&self) -> (f64, f64) {
        self.avg_updater.average_sides()
    }
}

pub struct ImpurityEntropyUpdater {
    avg_updater: AverageUpdater,
}

impl ImpurityEntropyUpdater {
    pub fn new(labels: &[u32], ids: &[usize]) -> Self {
        let mut s = ImpurityEntropyUpdater{ avg_updater: AverageUpdater::new(labels, ids) };
        s.reset(labels, ids);
        s
    }
}

impl ImpurityUpdaterTrait for ImpurityEntropyUpdater {
    type Label = u32;

    fn reset(&mut self, labels: &[u32], ids: &[usize]) {
        self.avg_updater.reset(labels, ids);
    }

    fn update(&mut self, new_pos: usize, labels: &[u32], ids: &[usize]) {
        self.avg_updater.update(new_pos, labels, ids);
    }

    fn impurity(&self) -> f64 {
        let avg = self.avg_updater.average();
        entropy_bin(avg)
    }

    fn average(&self) -> f64 {
        self.avg_updater.average()
    }

    fn impurity_sides(&self) -> (f64, f64) {
        let (lhs_avg, rhs_avg) = self.avg_updater.average_sides();
        (entropy_bin(lhs_avg), entropy_bin(rhs_avg))
    }

    fn average_sides(&self) -> (f64, f64) {
        self.avg_updater.average_sides()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn gini_impurity_test() {
        assert_eq!(gini_bin(0.0), 0.0);
        assert_eq!(gini_bin(1.0), 0.0);
        assert_eq!(gini_bin(0.5), 0.5);
        assert!((gini_bin(0.7) - 0.42).abs() < 1.0e-6);
    }

    #[test]
    fn entropy_test() {
        assert_eq!(entropy_bin(0.0), 0.0);
        assert_eq!(entropy_bin(1.0), 0.0);
        assert_eq!(entropy_bin(0.5), 1.0);
        assert!((entropy_bin(0.7) - 0.88129).abs() < 1.0e-5);
    }

    #[test]
    fn impurity_mse_updater_cases() {
        let labels = [0.0, 0.0, 0.0, 1.0, 2.0, 0.0];
        let ids = [0, 1, 2, 3, 4, 5];
        let mut mse = ImpurityMSEUpdater::new(&labels, &ids);
        assert_eq!(mse.impurity(), 14.0 / 24.0);
        assert_eq!(mse.impurity_sides(), (0.0, 14.0 / 24.0));
        mse.update(0, &labels, &ids);
        assert_eq!(mse.impurity_sides(), (0.0, 14.0 / 24.0));
        mse.update(1, &labels, &ids);
        assert_eq!(mse.impurity_sides(), (0.0, 16.0 / 25.0));
        mse.update(1, &labels, &ids);
        assert_eq!(mse.impurity_sides(), (0.0, 16.0 / 25.0));
        mse.update(2, &labels, &ids);
        assert_eq!(mse.impurity_sides(), (0.0, 11.0 / 16.0));
        mse.update(4, &labels, &ids);
        assert_eq!(mse.impurity_sides(), (3.0 / 16.0, 1.0));
        mse.update(5, &labels, &ids);
        assert_eq!(mse.impurity_sides(), (16.0 / 25.0, 0.0));
    }

    #[test]
    fn impurity_gini_updater_cases() {
        let labels = [0, 0, 0, 1, 2, 0];
        let ids = [0, 1, 2, 3, 4, 5];
        let mut gini = ImpurityGiniUpdater::new(&labels, &ids);
        assert_eq!(gini.impurity(), gini_bin(0.5));
        assert_eq!(gini.impurity_sides(), (0.0, gini_bin(0.5)));
        gini.update(0, &labels, &ids);
        assert_eq!(gini.impurity_sides(), (0.0, gini_bin(0.5)));
        gini.update(1, &labels, &ids);
        assert_eq!(gini.impurity_sides(), (0.0, gini_bin(0.6)));
        gini.update(1, &labels, &ids);
        assert_eq!(gini.impurity_sides(), (0.0, gini_bin(0.6)));
        gini.update(2, &labels, &ids);
        assert_eq!(gini.impurity_sides(), (0.0, gini_bin(0.75)));
        gini.update(4, &labels, &ids);
        assert_eq!(gini.impurity_sides(), (gini_bin(0.25), gini_bin(1.0)));
        gini.update(5, &labels, &ids);
        assert_eq!(gini.impurity_sides(), (gini_bin(0.6), 0.0));
    }
}
