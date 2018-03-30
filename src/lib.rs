extern crate rand;
extern crate permutohedron;
extern crate kdtree;

pub mod utils;
pub mod matrix;
pub mod kmeans;
pub mod em;
pub mod agg;
pub mod dbscan;
pub mod mshift;
pub mod cart;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
